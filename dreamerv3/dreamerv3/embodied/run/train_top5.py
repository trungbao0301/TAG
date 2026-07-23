import json
import re
from collections import deque

import embodied
import numpy as np


def train_top5(agent, env, replay, logger, args):

    top_count = 10
    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    print("Logdir", logdir)
    should_expl = embodied.when.Until(args.expl_until)
    should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
    should_save = embodied.when.Every(args.save_every, initial=False)
    should_sync = embodied.when.Every(args.sync_every)
    step = logger.step
    updates = embodied.Counter()
    metrics = embodied.Metrics()
    success_history = deque(maxlen=100)
    print("Observation space:", embodied.format(env.obs_space), sep="\n")
    print("Action space:", embodied.format(env.act_space), sep="\n")

    timer = embodied.Timer()
    timer.wrap("agent", agent, ["policy", "train", "report", "save"])
    timer.wrap("env", env, ["step"])
    timer.wrap("replay", replay, ["add", "save"])
    timer.wrap("logger", logger, ["write"])

    nonzeros = set()

    def episode_result(ep):
        score = float(ep["reward"].astype(np.float64).sum())
        duration_sec = float(
            np.asarray(ep.get("log_elapsed_sec", [np.inf])).reshape(-1)[-1]
        )
        success = bool(np.asarray(ep.get("log_success", [0.0])).reshape(-1)[-1])
        return score, duration_sec, success

    def rank_key(item):
        # Score is primary. For equal scores, a faster episode ranks higher.
        duration_sec = item.get("duration_sec")
        duration_sec = float(duration_sec) if duration_sec is not None else np.inf
        return (-round(float(item["score"]), 6), duration_sec)

    def per_episode(ep):
        length = len(ep["reward"]) - 1
        score, duration_sec, success = episode_result(ep)
        success_history.append(float(success))
        success_values = np.asarray(success_history, dtype=np.float32)
        success_rate_20 = float(success_values[-20:].mean())
        success_rate_50 = float(success_values[-50:].mean())
        success_rate_100 = float(success_values.mean())
        sum_abs_reward = float(np.abs(ep["reward"]).astype(np.float64).sum())
        logger.add(
            {
                "length": length,
                "score": score,
                "duration_sec": duration_sec,
                "success": float(success),
                "success_rate_20": success_rate_20,
                "success_rate_50": success_rate_50,
                "success_rate_100": success_rate_100,
                "sum_abs_reward": sum_abs_reward,
                "reward_rate": (np.abs(ep["reward"]) >= 0.5).mean(),
            },
            prefix="episode",
        )
        print(
            f"Episode has {length} steps, return {score:.6f}, "
            f"time {duration_sec:.3f}s, success={success}."
        )
        stats = {}
        for key in args.log_keys_video:
            if key in ep:
                stats[f"policy_{key}"] = ep[key]
        for key, value in ep.items():
            if not args.log_zeros and key not in nonzeros and (value == 0).all():
                continue
            nonzeros.add(key)
            if re.match(args.log_keys_sum, key):
                stats[f"sum_{key}"] = ep[key].sum()
            if re.match(args.log_keys_mean, key):
                stats[f"mean_{key}"] = ep[key].mean()
            if re.match(args.log_keys_max, key):
                stats[f"max_{key}"] = ep[key].max(0).mean()
        metrics.add(stats, prefix="stats")

    driver = embodied.Driver(env)
    driver.on_episode(lambda ep, worker: per_episode(ep))
    driver.on_step(lambda tran, _: step.increment())

    def add_replay(ep, worker):
        for i in range(len(ep["reward"])):
            trn = {k: v[i] for k, v in ep.items() if not k.startswith("log_")}
            replay.add(trn, worker)

    driver.on_episode(add_replay)

    checkpoint = embodied.Checkpoint(logdir / "checkpoint.ckpt", parallel=False)
    timer.wrap("checkpoint", checkpoint, ["save", "load"])
    checkpoint.step = step
    checkpoint.agent = agent
    checkpoint.replay = replay
    final_checkpoint = embodied.Checkpoint(
        logdir / "checkpoint.ckpt", log=True, parallel=False
    )
    final_checkpoint.step = step
    final_checkpoint.agent = agent
    final_checkpoint.replay = replay
    loaded_existing = False
    if args.from_checkpoint:
        checkpoint.load(args.from_checkpoint)
        loaded_existing = True
    elif checkpoint.exists():
        checkpoint.load()
        loaded_existing = True
    else:
        checkpoint.save()

    top_checkpoint = embodied.Checkpoint(log=False, parallel=False)
    top_checkpoint.step = step
    top_checkpoint.agent = agent
    top_checkpoint.replay = replay
    top10 = []
    top10_path = logdir / "checkpoint_top10.json"
    legacy_top5_path = logdir / "checkpoint_top5.json"
    load_path = top10_path if top10_path.exists() else legacy_top5_path
    if load_path.exists():
        try:
            top10[:] = json.loads(load_path.read())
            top10[:] = sorted(top10, key=rank_key)[:top_count]
            print(f"Loaded existing top-{top_count} list with {len(top10)} entries.")
        except Exception as exc:
            print(f"Could not load existing top-{top_count} list: {exc}")

    print(f"Replay size after checkpoint load: {len(replay)}")
    print("Prefill train dataset.")
    if loaded_existing:
        print("Prefill uses loaded checkpoint policy.")
        prefill_policy = lambda *a: agent.policy(*a, mode="train")
    else:
        print("Prefill uses random policy because no checkpoint was loaded.")
        random_agent = embodied.RandomAgent(env.act_space)
        prefill_policy = random_agent.policy
    while len(replay) < max(args.batch_steps, args.train_fill):
        driver(prefill_policy, steps=100)
    logger.add(metrics.result())
    logger.write()

    dataset = agent.dataset(replay.dataset)
    state = [None]
    batch = [None]
    episode = embodied.Counter()

    def train_step(ep, worker):
        for _ in range(should_train(step)):
            with timer.scope("dataset"):
                batch[0] = next(dataset)
            outs, state[0], mets = agent.train(batch[0], state[0])
            metrics.add(mets, prefix="train")
            if "priority" in outs:
                replay.prioritize(outs["key"], outs["priority"])
            updates.increment()
        agent.sync()
        agg = metrics.result()
        report = agent.report(batch[0])
        report = {k: v for k, v in report.items() if "train/" + k not in agg}
        logger.add(agg)
        logger.add(report, prefix="report")
        logger.add(replay.stats, prefix="replay")
        logger.add(timer.stats(), prefix="timer")
        logger.write(fps=True)
        episode.increment()

    def save_top10_checkpoint(ep, worker):
        score, duration_sec, success = episode_result(ep)
        current = {
            "score": score,
            "duration_sec": duration_sec,
            "success": success,
            "step": int(step),
            "episode": int(episode),
        }
        candidates = sorted(top10 + [current], key=rank_key)[:top_count]
        if current not in candidates:
            return

        filename = (
            f"checkpoint_top_score{score:.6f}_"
            f"time{duration_sec:.3f}s_"
            f"step{int(step)}_episode{int(episode)}.ckpt"
        ).replace("-", "m")
        current["path"] = filename
        top_path = logdir / filename
        print(
            f"New top-{top_count} episode score {score:.6f}, "
            f"time {duration_sec:.3f}s, success={success}; "
            f"saving checkpoint: {top_path}"
        )
        top_checkpoint.save(top_path)

        old_paths = {item.get("path") for item in top10 if item.get("path")}
        keep_paths = {item.get("path") for item in candidates if item.get("path")}
        for old_path in old_paths - keep_paths:
            try:
                (logdir / old_path).remove()
            except FileNotFoundError:
                pass

        top10[:] = candidates
        (logdir / "checkpoint_top10.json").write(
            json.dumps(top10, indent=2, sort_keys=True) + "\n",
            mode="w",
        )
        lines = []
        for index, item in enumerate(top10, 1):
            item_duration = item.get("duration_sec")
            time_text = (
                f"{float(item_duration):.3f}s"
                if item_duration is not None
                else "unknown"
            )
            lines.append(
                f"{index}. score={item['score']:.6f} "
                f"time={time_text} success={item.get('success', 'unknown')} "
                f"step={item['step']} episode={item['episode']} "
                f"path={item.get('path', '')}"
            )
        text = "\n".join(lines) + "\n"
        (logdir / "checkpoint_top10.txt").write(text, mode="w")
        # Keep the old filenames useful for existing monitoring commands.
        (logdir / "checkpoint_top5.txt").write(
            "\n".join(lines[:5]) + "\n", mode="w"
        )

    driver.on_episode(train_step)
    driver.on_episode(save_top10_checkpoint)

    print("Start training loop.")
    policy = lambda *args: agent.policy(
        *args, mode="explore" if should_expl(step) else "train"
    )
    try:
        while step < args.steps:
            driver(policy, episodes=1)
            if should_save(episode):
                checkpoint.save()
    except KeyboardInterrupt:
        print("Interrupted by user; saving final checkpoint before exit.")
    finally:
        replay.save(wait=True)
        final_checkpoint.save()
        logger.write()
        print(
            f"Final checkpoint saved: {logdir / 'checkpoint.ckpt'} "
            f"at step {int(step)}"
        )

#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay_dir", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--task", default="thomas")
    parser.add_argument("--max_frames", type=int, default=0)
    args = parser.parse_args()

    replay_dir = Path(args.replay_dir).expanduser()
    out_root = Path(args.out_root).expanduser()
    files = sorted(replay_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"no .npz files found in {replay_dir}")

    episodes = []
    actions = []
    rewards = []
    total = 0
    episode_id = 0

    for path in files:
        with np.load(path) as data:
            if "action" not in data or "reward" not in data:
                print(f"skip {path}: missing action/reward")
                continue
            action = np.nan_to_num(data["action"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            reward = np.nan_to_num(data["reward"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            action = np.clip(action, -1.0, 1.0)
            reward = np.clip(reward, -1000.0, 1000.0)

            n = min(len(action), len(reward))
            if args.max_frames and total + n > args.max_frames:
                n = args.max_frames - total
            if n <= 0:
                break

            action = action[:n]
            reward = reward[:n]
            is_first = data["is_first"][:n] if "is_first" in data else np.zeros(n, dtype=bool)

        ep = np.empty(n, dtype=np.int64)
        for i in range(n):
            if total > 0 or i > 0:
                if bool(is_first[i]):
                    episode_id += 1
            ep[i] = episode_id

        episodes.append(torch.from_numpy(ep))
        actions.append(torch.from_numpy(action))
        rewards.append(torch.from_numpy(reward))
        total += n
        print(f"processed {path.name}; total_steps={total}; episodes={episode_id + 1}")
        if args.max_frames and total >= args.max_frames:
            break

    out_root.mkdir(parents=True, exist_ok=True)
    demo = {
        "episode": torch.cat(episodes, dim=0).to(torch.int64),
        "action": torch.cat(actions, dim=0).to(torch.float32),
        "reward": torch.cat(rewards, dim=0).to(torch.float32),
    }
    out_path = out_root / f"{args.task}.pt"
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(demo, tmp)
    tmp.replace(out_path)

    tasks_json = out_root / "tasks.json"
    tasks_json.write_text(
        '{\n'
        f'  "{args.task}": {{\n'
        '    "action_dim": 2\n'
        '  }\n'
        '}\n'
    )
    print(f"saved {out_path}")
    print(f"saved {tasks_json}")
    print({k: tuple(v.shape) for k, v in demo.items()})


if __name__ == "__main__":
    main()

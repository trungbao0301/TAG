#!/usr/bin/env python3
"""Copy a DreamerV3 checkpoint keeping only the agent weights.

`train_top5` stores three things in checkpoint.ckpt: `step`, `agent`, and
`replay`. Resuming loads all three, so a run whose replay is contaminated
(recorded across two different estimator calibrations, say) drags that data
back in even with a fresh logdir.

`Checkpoint.load()` with no `keys=` iterates over whatever the file contains,
so dropping `step` and `replay` here is enough -- the learner then starts at
step 0 with an empty replay but the policy it already learned:

    ./run_server_dreamer_stuck_gpu0.sh --run.from_checkpoint <output>

Keys are dropped rather than the whole file re-saved so the agent blob is
passed through byte-identically.
"""
import argparse
import pickle
import sys

DROP = ("replay", "step")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", help="checkpoint.ckpt to read")
    ap.add_argument("output", help="policy-only checkpoint to write")
    ap.add_argument(
        "--keep",
        default="agent",
        help="comma-separated keys to keep (default: agent)",
    )
    args = ap.parse_args()

    keep = [k.strip() for k in args.keep.split(",") if k.strip()]
    with open(args.source, "rb") as fh:
        data = pickle.loads(fh.read())

    missing = [k for k in keep if k not in data]
    if missing:
        print(
            f"checkpoint has no {missing}; it contains {sorted(data)}",
            file=sys.stderr,
        )
        return 1

    # _timestamp is required: load() reads it to print the checkpoint's age.
    out = {k: data[k] for k in keep}
    out["_timestamp"] = data["_timestamp"]

    with open(args.output, "wb") as fh:
        fh.write(pickle.dumps(out))

    dropped = sorted(set(data) - set(out))
    print(f"kept {sorted(keep)}, dropped {dropped}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

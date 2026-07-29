#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def save_shard(frames, outdir, task, index):
    if not frames:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    tensor = torch.cat(frames, dim=0).contiguous()
    path = outdir / f"{task}_shard{index:04d}.pt"
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"frames": tensor}, tmp)
    tmp.replace(path)
    print(f"saved {path} {tuple(tensor.shape)}")


def image_batch_to_frames(images, size):
    if images.ndim != 4:
        raise ValueError(f"expected image array with 4 dims, got {images.shape}")
    if images.shape[-1] == 1:
        images = np.repeat(images, 3, axis=-1)
    elif images.shape[-1] != 3:
        raise ValueError(f"expected 1 or 3 image channels, got {images.shape}")

    tensor = torch.from_numpy(np.ascontiguousarray(images))
    tensor = tensor.permute(0, 3, 1, 2).to(torch.uint8)
    if tensor.shape[-2:] == (size, size):
        return tensor

    resized = F.interpolate(
        tensor.to(torch.float32),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )
    return resized.clamp(0, 255).to(torch.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay_dir", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--task", default="thomas")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--shard_size", type=int, default=2048)
    parser.add_argument("--max_frames", type=int, default=0)
    args = parser.parse_args()

    replay_dir = Path(args.replay_dir).expanduser()
    outdir = Path(args.out_root).expanduser() / args.task
    files = sorted(replay_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"no .npz files found in {replay_dir}")

    pending = []
    pending_count = 0
    shard_index = 0
    total = 0

    for path in files:
        with np.load(path) as data:
            if "image" not in data:
                print(f"skip {path}: no image key")
                continue
            frames = image_batch_to_frames(data["image"], args.size)

        if args.max_frames and total + frames.shape[0] > args.max_frames:
            frames = frames[: args.max_frames - total]
        if frames.shape[0] == 0:
            break

        start = 0
        while start < frames.shape[0]:
            need = args.shard_size - pending_count
            chunk = frames[start : start + need]
            pending.append(chunk)
            pending_count += chunk.shape[0]
            start += chunk.shape[0]
            total += chunk.shape[0]

            if pending_count == args.shard_size:
                save_shard(pending, outdir, args.task, shard_index)
                shard_index += 1
                pending = []
                pending_count = 0

        print(f"processed {path.name}; total_frames={total}")
        if args.max_frames and total >= args.max_frames:
            break

    save_shard(pending, outdir, args.task, shard_index)
    print(f"done: total_frames={total}, out_root={outdir.parent}")


if __name__ == "__main__":
    main()

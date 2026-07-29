"""Train and export a small marble heatmap model from click labels."""

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("marble_detector.onnx"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--input-width", type=int, default=320)
    parser.add_argument("--input-height", type=int, default=200)
    parser.add_argument(
        "--output-stride",
        type=int,
        choices=(4, 8),
        default=8,
        help="Heatmap pixel stride. Use 4 for more precise marble centers.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--label-limit",
        type=int,
        default=None,
        help="Use only the first N real labels, allowing later labels to remain an independent test set.",
    )
    args = parser.parse_args()

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset, random_split
    except (ImportError, OSError) as error:
        raise SystemExit(
            "PyTorch is required only for training. Use a working PyTorch environment "
            f"(the exported ONNX detector does not require PyTorch): {error}"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    class Labels(Dataset):
        def __init__(self, root, label_limit=None):
            self.root = root
            with (root / "labels.csv").open(newline="") as handle:
                self.rows = list(csv.DictReader(handle))
            if label_limit is not None:
                self.rows = self.rows[:label_limit]
            # Each visible frame also produces one hard negative by removing
            # the clicked marble with local inpainting. This lets an initial
            # model learn "not visible" even when the first capture session
            # contains only positive clicks. Real absent/occluded frames are
            # still preferred and remain part of self.rows when available.
            self.samples = [(row, False) for row in self.rows]
            self.samples.extend(
                (row, True) for row in self.rows if row["visible"] == "1"
            )

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            row, synthetic_negative = self.samples[index]
            image = cv2.imread(str(self.root / "images" / row["filename"]))
            if image is None:
                raise FileNotFoundError(row["filename"])
            source_h, source_w = image.shape[:2]
            visible = bool(int(row["visible"]))
            x = float(row["x_px"]) / source_w if visible else 0.0
            y = float(row["y_px"]) / source_h if visible else 0.0
            if synthetic_negative:
                mask = np.zeros((source_h, source_w), dtype=np.uint8)
                radius = max(8, round(min(source_h, source_w) * 0.03))
                cv2.circle(
                    mask,
                    (round(x * source_w), round(y * source_h)),
                    radius,
                    255,
                    -1,
                )
                image = cv2.inpaint(image, mask, radius, cv2.INPAINT_TELEA)
                visible = False

            if random.random() < 0.5:
                image = image[:, ::-1].copy()
                x = 1.0 - x
            if random.random() < 0.5:
                image = image[::-1].copy()
                y = 1.0 - y
            gain = random.uniform(0.75, 1.25)
            bias = random.uniform(-20.0, 20.0)
            image = np.clip(image.astype(np.float32) * gain + bias, 0, 255)
            image = cv2.resize(image, (args.input_width, args.input_height))
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
            image = np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1))

            out_h = args.input_height // args.output_stride
            out_w = args.input_width // args.output_stride
            target = np.zeros((1, out_h, out_w), np.float32)
            if visible:
                cx = x * out_w
                cy = y * out_h
                yy, xx = np.mgrid[:out_h, :out_w]
                target[0] = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 1.25**2))
            return torch.from_numpy(image), torch.from_numpy(target)

    class HeatmapNet(nn.Module):
        def __init__(self):
            super().__init__()

            def block(in_ch, out_ch, stride=2):
                return nn.Sequential(
                    nn.Conv2d(
                        in_ch, out_ch, 3, stride=stride, padding=1, bias=False
                    ),
                    nn.BatchNorm2d(out_ch),
                    nn.SiLU(),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.SiLU(),
                )

            self.network = nn.Sequential(
                block(3, 24),
                block(24, 48),
                block(48, 80, stride=2 if args.output_stride == 8 else 1),
                nn.Conv2d(80, 1, 1),
            )

        def forward(self, image):
            return self.network(image)

    dataset = Labels(args.dataset, args.label_limit)
    if len(dataset) < 50:
        raise SystemExit(
            "Collect at least 50 labels; 500+ across lighting/occlusion is recommended."
        )
    validation_count = max(1, round(0.2 * len(dataset)))
    train_set, validation_set = random_split(
        dataset,
        [len(dataset) - validation_count, validation_count],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HeatmapNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([20.0], device=device))

    best_loss = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), targets)
            loss.backward()
            optimizer.step()
            train_loss += float(loss) * images.shape[0]
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for images, targets in validation_loader:
                images, targets = images.to(device), targets.to(device)
                validation_loss += float(loss_fn(model(images), targets)) * images.shape[0]
        train_loss /= len(train_set)
        validation_loss /= len(validation_set)
        print(f"epoch={epoch:03d} train={train_loss:.5f} val={validation_loss:.5f}")
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.cpu().eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, args.input_height, args.input_width)
    torch.onnx.export(
        model,
        dummy,
        str(args.output),
        input_names=["image"],
        output_names=["heatmap_logits"],
        opset_version=17,
    )
    print(f"Exported {args.output} (best validation loss {best_loss:.5f})")


if __name__ == "__main__":
    main()

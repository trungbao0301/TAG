#!/usr/bin/env python3
from pathlib import Path


PATCH = '''
if os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes") or os.environ.get("WANDB_MODE", "").lower() == "disabled":
    class _WandbImage:
        def __init__(self, *args, **kwargs):
            pass
    def _wandb_noop(*args, **kwargs):
        return None
    wandb.init = _wandb_noop
    wandb.log = _wandb_noop
    wandb.Image = _WandbImage
'''


def main():
    for name in ["train_tokenizer.py", "train_dynamics.py"]:
        path = Path(name)
        text = path.read_text()
        if PATCH.strip() in text:
            continue
        backup = path.with_suffix(path.suffix + ".bak_wandb_noop")
        if not backup.exists():
            backup.write_text(text)
        text = text.replace("import wandb\n", "import wandb\n" + PATCH, 1)
        path.write_text(text)
        print(f"patched {path}")


if __name__ == "__main__":
    main()

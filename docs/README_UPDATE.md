# TAG website complete package

This package is ready to extract into the existing TAG Git repository. It was built from the files supplied in the ChatGPT conversation, so you do not need the original files on the Ubuntu machine.

## Included

- `docs/index.html` research-project webpage
- `docs/assets/videos/tag_real.mp4` — rotated real TAG demo
- `docs/assets/videos/tag_sim_full_success.gif` — supplied animated Isaac Sim rollout
- `docs/assets/paper/TAG_draft.pdf` — Overleaf PDF
- `docs/assets/paper/main.tex` — Overleaf LaTeX source
- Hardware, real/digital map, Isaac Sim, MuJoCo, pendulum, live-view, reward, success-rate, and duration figures from the Overleaf package

## Install

Download `TAG_site_complete_patch.zip`, then run on the local Ubuntu machine:

```bash
cd ~/TAG
unzip -o ~/Downloads/TAG_site_complete_patch.zip -d ~/TAG

# Optional local preview
python3 -m http.server 8000 -d docs
```

Open `http://localhost:8000`, then press Ctrl+C in the terminal when finished.

## Push to GitHub

```bash
cd ~/TAG
git status --short -- docs README_UPDATE.md
git add docs README_UPDATE.md
git commit -m "Upgrade TAG website with demos and paper assets"
git push
```

GitHub Pages should redeploy automatically from `main /docs`.

## Note on the supplied simulation asset

The supplied `tag_sim_full_success.gif` is a real animated GIF (640×400, about 24 seconds, 257 frames) and is included directly in the website package. It will animate automatically in the Isaac Sim demo panel.

# TAG website — latest Overleaf version

This package was rebuilt from the latest supplied Overleaf archive (`TAG__testbed_for_autonomous_games.zip`, Aug. 10, 2026), not the older DreamerV3-only manuscript.

It includes:
- current manuscript PDF and `main.tex`
- all figures from the latest Overleaf archive
- real TAG MP4 demo
- animated Isaac Sim GIF demo
- website content reflecting the latest paper: integrated Isaac Sim, DreamerV3 maze control, Furuta/TQC control, moving-board curriculum, sim-to-real transfer, ESP32 deployment, results, and generalization/future work

## Install into the repository

```bash
cd ~/TAG
unzip -o ~/Downloads/TAG_site_latest_overleaf.zip -d ~/TAG
python3 -m http.server 8000 -d docs
```

Open http://localhost:8000 to preview.

Then:

```bash
git add docs README_WEBSITE.md
git commit -m "Update TAG website from latest manuscript"
git push
```

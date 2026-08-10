# TAG project website — deploy with GitHub Pages

This folder is designed to be copied into the existing `docs/` folder of:

`https://github.com/trungbao0301/TAG`

The site uses only static HTML/CSS/JS, so GitHub Pages can host it directly.

## 1. Copy into your repo

From the root of your local TAG repository:

```bash
# Keep your existing docs/path_grid.png and other docs files.
cp /path/to/TAG_website/index.html docs/index.html
cp /path/to/TAG_website/assets/site.css docs/assets/site.css
cp /path/to/TAG_website/assets/site.js docs/assets/site.js
cp /path/to/TAG_website/assets/tag-pipeline.svg docs/assets/tag-pipeline.svg
cp /path/to/TAG_website/assets/dreamer-loop.svg docs/assets/dreamer-loop.svg

mkdir -p docs/assets/videos
```

Or copy the files manually in VS Code.

## 2. Add videos

The website automatically replaces each placeholder when the matching file exists:

```text
docs/assets/videos/tag_real.mp4
docs/assets/videos/tag_isaac.mp4
docs/assets/videos/sim_rollout.mp4
docs/assets/videos/sim_to_real.mp4
docs/assets/videos/tag_nominal.mp4
docs/assets/videos/tag_disturbance.mp4
docs/assets/videos/tag_adapted.mp4
```

Recommended encoding for web:
- MP4 / H.264
- 720p or 1080p
- muted clips work best for autoplay
- keep each short (roughly 5–20 s) and looping

Example FFmpeg compression:

```bash
ffmpeg -i input.mp4 -c:v libx264 -crf 24 -preset medium -movflags +faststart -an output.mp4
```

## 3. Edit the author list / paper / results

Open `docs/index.html` and search for:

- `Trung Bao Truong`
- `Paper · Coming Soon`
- `@article{tag2026`
- `Success Rate`

Those are the main fields to replace before publication.

## 4. Commit and push

```bash
git add docs/index.html docs/assets
git commit -m "Add TAG project website"
git push
```

## 5. Enable GitHub Pages

In GitHub:

**Repository → Settings → Pages**

Choose:

- Source: **Deploy from a branch**
- Branch: **main**
- Folder: **/docs**

Save.

GitHub will publish the page at a URL similar to:

`https://trungbao0301.github.io/TAG/`

## 6. Optional: make a short project URL

If you later want something like `tag-testbed.github.io`, create a separate GitHub organization/repository. For now, `/TAG/` is simpler and keeps the website next to the code.

## Notes

- `path_grid.png` is already present in your repository's `docs/` folder and the page uses it automatically.
- No web framework is required.
- The design is inspired by clean academic project pages such as SimDist and Diffusion Policy Policy Optimization, but the code/layout here is original for TAG.

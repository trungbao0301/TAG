(function () {
  const mediaFiles = {
    "real-hero": "assets/videos/tag_real.mp4",
    "sim-hero": "assets/videos/tag_isaac.mp4",
    "sim-rollout": "assets/videos/sim_rollout.mp4",
    "sim-real": "assets/videos/sim_to_real.mp4",
    "nominal": "assets/videos/tag_nominal.mp4",
    "disturbance": "assets/videos/tag_disturbance.mp4",
    "adapted": "assets/videos/tag_adapted.mp4"
  };

  // If a requested MP4 exists on GitHub Pages, swap the placeholder for a looping video.
  // A HEAD request may be blocked on some local file:// previews, so placeholders remain there.
  Object.entries(mediaFiles).forEach(([key, src]) => {
    const target = document.querySelector(`[data-media="${key}"]`);
    if (!target || location.protocol === "file:") return;

    fetch(src, { method: "HEAD" }).then((res) => {
      if (!res.ok) return;
      const video = document.createElement("video");
      video.src = src;
      video.autoplay = true;
      video.loop = true;
      video.muted = true;
      video.playsInline = true;
      video.controls = true;
      video.setAttribute("aria-label", key.replaceAll("-", " "));
      video.style.width = "100%";
      video.style.height = "100%";
      video.style.objectFit = "cover";
      video.style.position = "absolute";
      video.style.inset = "0";
      target.replaceChildren(video);
    }).catch(() => {});
  });
})();

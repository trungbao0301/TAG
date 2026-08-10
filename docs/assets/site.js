(function () {
  const simTarget = document.querySelector('[data-sim-media]');
  if (!simTarget || location.protocol === 'file:') return;

  const candidates = [
    'assets/videos/tag_sim_full_success.gif',
    'assets/videos/tag_isaac.mp4'
  ];

  const tryCandidate = (index) => {
    if (index >= candidates.length) return;
    const src = candidates[index];
    fetch(src, { method: 'HEAD' }).then((res) => {
      if (!res.ok) return tryCandidate(index + 1);
      if (src.endsWith('.gif')) {
        const img = document.createElement('img');
        img.src = src;
        img.alt = 'TAG Isaac Sim successful rollout';
        img.style.width = '100%';
        img.style.height = '100%';
        img.style.position = 'absolute';
        img.style.inset = '0';
        img.style.objectFit = 'contain';
        simTarget.replaceChildren(img);
      } else {
        const video = document.createElement('video');
        video.src = src;
        video.autoplay = true;
        video.loop = true;
        video.muted = true;
        video.playsInline = true;
        video.controls = true;
        video.style.width = '100%';
        video.style.height = '100%';
        video.style.position = 'absolute';
        video.style.inset = '0';
        video.style.objectFit = 'contain';
        simTarget.replaceChildren(video);
      }
    }).catch(() => tryCandidate(index + 1));
  };

  tryCandidate(0);
})();

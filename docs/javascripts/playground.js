// Interactive tiling playground for the docs.
// The grid logic mirrors plakakia.tiling.tile_starts exactly.
(function () {
  "use strict";

  const COLORS = ["#009688", "#f57c00", "#9c27b0", "#2196f3"];

  function tileStarts(length, tile, step, coverEdges) {
    if (length < tile) return [];
    const starts = [];
    for (let s = 0; s <= length - tile; s += step) starts.push(s);
    if (coverEdges && starts[starts.length - 1] + tile < length) starts.push(length - tile);
    return starts;
  }

  // Total length covered by the union of [s, s + tile) intervals.
  function coveredLength(starts, tile) {
    let total = 0;
    let end = 0;
    starts.forEach((s) => {
      total += Math.max(0, s + tile - Math.max(s, end));
      end = Math.max(end, s + tile);
    });
    return total;
  }

  function init(root) {
    if (root.dataset.ready) return;
    root.dataset.ready = "1";
    const W = Number(root.dataset.width);
    const H = Number(root.dataset.height);
    const canvas = root.querySelector("canvas");
    const ctx = canvas.getContext("2d");
    const img = new Image();
    const $ = (name) => root.querySelector(`[name="${name}"]`);
    const tileIn = $("tile");
    const stepIn = $("step");
    const coverIn = $("cover");

    function draw() {
      const tile = Number(tileIn.value);
      const step = Number(stepIn.value);
      const cover = coverIn.checked;
      root.querySelector(".v-tile").textContent = tile;
      root.querySelector(".v-step").textContent = step;

      const xs = tileStarts(W, tile, step, cover);
      const ys = tileStarts(H, tile, step, cover);
      const k = canvas.width / W;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (img.complete) ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      // Pixels right of / below the last tile are dropped: shade them red.
      const coverX = xs.length ? Math.min(W, xs[xs.length - 1] + tile) : 0;
      const coverY = ys.length ? Math.min(H, ys[ys.length - 1] + tile) : 0;
      ctx.fillStyle = "rgba(229, 57, 53, 0.55)";
      ctx.fillRect(coverX * k, 0, (W - coverX) * k, canvas.height);
      ctx.fillRect(0, coverY * k, coverX * k, (H - coverY) * k);

      ys.forEach((y, r) => {
        xs.forEach((x, c) => {
          const color = COLORS[(r % 2) * 2 + (c % 2)];
          ctx.fillStyle = color + "26";
          ctx.fillRect(x * k, y * k, tile * k, tile * k);
          ctx.strokeStyle = color;
          ctx.lineWidth = 2;
          ctx.strokeRect(x * k + 1, y * k + 1, tile * k - 2, tile * k - 2);
        });
      });

      const n = xs.length * ys.length;
      const dropped = 1 - (coveredLength(xs, tile) * coveredLength(ys, tile)) / (W * H);
      const overlap = step < tile ? Math.round((100 * (tile - step)) / tile) : 0;
      root.querySelector(".s-n").textContent = n;
      root.querySelector(".s-overlap").textContent = overlap + "%";
      root.querySelector(".s-dropped").textContent = (100 * dropped).toFixed(1) + "%";
      root.querySelector("code.s-code").textContent =
        `tiles, coords = tile_image(image, tile_size=${tile}, step_size=${step}` +
        (cover ? ",\n                           cover_edges=True)" : ")") +
        `\n# image ${H}x${W} -> ${n} tiles of ${tile}x${tile}`;
    }

    [tileIn, stepIn, coverIn].forEach((el) => el.addEventListener("input", draw));
    img.addEventListener("load", draw);
    img.src = root.dataset.src;
    draw();
  }

  function boot() {
    document.querySelectorAll(".plk-playground").forEach(init);
  }

  // Material's instant loading exposes document$; fall back to DOMContentLoaded.
  if (typeof document$ !== "undefined") {
    document$.subscribe(boot);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();

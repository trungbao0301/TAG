#!/usr/bin/env python3
"""Draw a path by clicking on a to-scale board outline in a browser, and
write it to the waypoints file env_ballplate.py reads (TAG_BP_PATH_FILE).

Stdlib only -- no Flask, no extra install, since getting even numpy onto a
plain machine already took a fight with this project's Python environments.

    python3 tools/draw_path_server.py --path-file ~/tag_ballplate_path.json

Open the printed URL (tunnel it like TensorBoard if the server is remote),
click points on the board, hit Send. env_ballplate.py re-reads the file every
episode reset, so the policy starts replaying the new path on its next reset
-- no restart needed on either side.
"""
import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BOARD_WIDTH_M = 0.259
BOARD_HEIGHT_M = 0.229

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Draw a path for TAG</title>
<style>
  body {{ font-family: sans-serif; background: #222; color: #eee; }}
  canvas {{ background: #333; border: 2px solid #666; cursor: crosshair; }}
  button {{ font-size: 16px; padding: 8px 16px; margin: 4px; }}
  #status {{ margin-top: 8px; }}
</style></head>
<body>
<h2>Draw a path (click to add points, in order)</h2>
<canvas id="board" width="{width_px}" height="{height_px}"></canvas><br>
<button onclick="undo()">Undo</button>
<button onclick="clearAll()">Clear</button>
<button onclick="send()">Send to policy</button>
<div id="status">{num_points} point(s) currently saved.</div>
<script>
const W_M = {board_w}, H_M = {board_h};
const canvas = document.getElementById('board');
const ctx = canvas.getContext('2d');
let points = {initial_points};  // board metres, lower-left origin

function toPx(p) {{
  return [p[0] / W_M * canvas.width, (1 - p[1] / H_M) * canvas.height];
}}
function toM(px, py) {{
  return [px / canvas.width * W_M, (1 - py / canvas.height) * H_M];
}}
function draw() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#888'; ctx.strokeRect(1, 1, canvas.width - 2, canvas.height - 2);
  if (points.length === 0) return;
  ctx.strokeStyle = '#4ac0ff'; ctx.lineWidth = 2; ctx.beginPath();
  let [x0, y0] = toPx(points[0]); ctx.moveTo(x0, y0);
  for (const p of points.slice(1)) {{ const [x, y] = toPx(p); ctx.lineTo(x, y); }}
  ctx.stroke();
  points.forEach((p, i) => {{
    const [x, y] = toPx(p);
    ctx.fillStyle = i === 0 ? '#2ecc71' : '#4ac0ff';
    ctx.beginPath(); ctx.arc(x, y, 4, 0, 2 * Math.PI); ctx.fill();
  }});
}}
canvas.addEventListener('click', (e) => {{
  const rect = canvas.getBoundingClientRect();
  points.push(toM(e.clientX - rect.left, e.clientY - rect.top));
  draw();
}});
function undo() {{ points.pop(); draw(); }}
function clearAll() {{ points = []; draw(); }}
function send() {{
  if (points.length < 2) {{ alert('Draw at least 2 points first.'); return; }}
  fetch('/send', {{method: 'POST', body: JSON.stringify({{waypoints: points}})}})
    .then(r => r.text())
    .then(t => document.getElementById('status').innerText = t);
}}
draw();
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    path_file = None
    board_width = BOARD_WIDTH_M
    board_height = BOARD_HEIGHT_M
    px_per_m = 2500

    def _load_current(self):
        try:
            with open(self.path_file, "r", encoding="utf-8") as f:
                return json.load(f).get("waypoints", [])
        except (OSError, ValueError, json.JSONDecodeError):
            return []

    def do_GET(self):
        if self.path != "/":
            self.send_response(404)
            self.end_headers()
            return
        points = self._load_current()
        body = PAGE.format(
            width_px=int(self.board_width * self.px_per_m),
            height_px=int(self.board_height * self.px_per_m),
            board_w=self.board_width,
            board_h=self.board_height,
            num_points=len(points),
            initial_points=json.dumps(points),
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/send":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length))
            waypoints = payload["waypoints"]
            assert isinstance(waypoints, list) and len(waypoints) >= 2
            for x, y in waypoints:
                assert 0.0 <= x <= self.board_width and 0.0 <= y <= self.board_height
        except Exception as exc:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(f"Rejected: {exc}".encode("utf-8"))
            return

        tmp_path = f"{self.path_file}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump({"waypoints": waypoints}, f)
        os.replace(tmp_path, self.path_file)  # atomic, so a reset never reads a half-write

        message = f"Saved {len(waypoints)} points -> {self.path_file}. Takes effect next episode reset."
        print("[draw_path_server]", message)
        body = message.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # the default logs every request to stderr; too noisy for click-by-click drawing


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--path-file", required=True,
                    help="must match TAG_BP_PATH_FILE on the env's side")
    p.add_argument("--board-width", type=float, default=BOARD_WIDTH_M)
    p.add_argument("--board-height", type=float, default=BOARD_HEIGHT_M)
    args = p.parse_args()

    Handler.path_file = os.path.expanduser(args.path_file)
    Handler.board_width = args.board_width
    Handler.board_height = args.board_height

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[draw_path_server] http://{args.host}:{args.port}  ->  {Handler.path_file}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

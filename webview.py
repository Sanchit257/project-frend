"""Live demo view: the annotated camera feed plus Frend's activity log, served
to any browser on the network (e.g. the MacBook at http://192.168.2.2:8080).

The detection loop hands each processed frame to `publish()`; it is JPEG-
encoded at most a few times a second so the stream never slows detection.
"""

import collections
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

from detector import PROXIMITY_NAMES, draw_detections

log = logging.getLogger("frend.web")

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Frend live view</title>
<style>
body{margin:0;background:#111;color:#eee;font:15px system-ui,sans-serif;display:flex;gap:16px;padding:16px;box-sizing:border-box;height:100vh}
#feed{flex:3;display:flex;flex-direction:column;gap:8px}#feed img{width:100%;border-radius:8px;background:#000}
#side{flex:2;display:flex;flex-direction:column;gap:12px;min-width:0}
h2{margin:0;font-size:18px}.box{background:#1c1c1c;border-radius:8px;padding:10px;overflow:auto}
#objs div{padding:2px 0}#log{flex:1;font:13px ui-monospace,monospace;white-space:pre-wrap}
.k{color:#888}.far{color:#4c4}.close{color:#fc3}.veryclose{color:#f44}
</style></head><body>
<div id="feed"><h2>Frend - live camera &amp; detections</h2><img src="/stream">
<div class="k">Boxes: <span class="far">green = far</span>, <span class="close">yellow = close</span>,
<span class="veryclose">red = very close</span>. Blue frame = possible wall check zone.</div></div>
<div id="side"><div class="box"><h2>Status</h2><div id="stat"></div></div>
<div class="box"><h2>Objects now</h2><div id="objs"></div></div>
<div class="box" id="log"></div></div>
<script>
async function tick(){try{const s=await (await fetch('/state')).json();
document.getElementById('stat').innerHTML=`Detector: ${s.fps.toFixed(1)} FPS<br>Camera motion: ${s.motion.toFixed(1)} ${s.moving?'(moving - alerts paused)':''}<br>Wall check: edges ${s.wall_edges.toFixed(3)}, texture ${s.wall_texture.toFixed(1)}<br>Alerts: ${s.alerts?'on':'off'}`;
document.getElementById('objs').innerHTML=s.objects.map(o=>`<div class="${o.proximity.replace(' ','')}">${o.label} - ${o.position}, ${o.proximity} (${o.confidence})</div>`).join('')||'<span class="k">nothing</span>';
const l=document.getElementById('log');l.textContent=s.log.join('\\n');l.scrollTop=l.scrollHeight;}catch(e){}}
setInterval(tick,500);tick();
</script></body></html>"""


class _LogTap(logging.Handler):
    """Copies Frend's own INFO log lines (heard commands, alerts, answers)."""

    def __init__(self, lines):
        super().__init__(logging.INFO)
        self.lines = lines

    def emit(self, record):
        if record.name.startswith("frend"):
            self.lines.append(time.strftime("%H:%M:%S ") + record.getMessage())


class LiveView:
    def __init__(self, port=8080, max_fps=8):
        self.port = port
        self.min_interval = 1.0 / max_fps
        self._jpeg = None
        self._jpeg_id = 0
        self._last_encode = 0.0
        self._cond = threading.Condition()
        self.state = {"fps": 0.0, "motion": 0.0, "moving": False, "wall_edges": 0.0,
                      "wall_texture": 0.0, "alerts": True, "objects": []}
        self.log_lines = collections.deque(maxlen=200)
        logging.getLogger().addHandler(_LogTap(self.log_lines))
        self._server = None

    def start(self):
        view = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                if self.path == "/":
                    self._send(200, "text/html", PAGE.encode())
                elif self.path == "/state":
                    body = dict(view.state, log=list(view.log_lines)[-60:])
                    self._send(200, "application/json", json.dumps(body).encode())
                elif self.path == "/stream":
                    self._stream()
                else:
                    self._send(404, "text/plain", b"not found")

            def _send(self, code, ctype, body):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _stream(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                last = 0
                try:
                    while True:
                        with view._cond:
                            view._cond.wait_for(lambda: view._jpeg_id != last, timeout=5)
                            jpeg, last = view._jpeg, view._jpeg_id
                        if jpeg is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self._server = ThreadingHTTPServer(("0.0.0.0", self.port), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, name="webview", daemon=True).start()
        log.info("Live view at http://<pi-address>:%d (e.g. http://192.168.2.2:%d)",
                 self.port, self.port)
        return self

    def publish(self, frame, detections, *, fps, motion, moving, walls, alerts):
        self.state.update(
            fps=fps, motion=motion, moving=moving, alerts=alerts,
            wall_edges=walls.edge_score, wall_texture=walls.texture_score,
            objects=[{"label": d.label, "position": d.position,
                      "proximity": PROXIMITY_NAMES[d.proximity],
                      "confidence": round(d.confidence, 2)} for d in detections])
        now = time.monotonic()
        if now - self._last_encode < self.min_interval:
            return
        self._last_encode = now
        out = draw_detections(frame, detections)
        h, w = out.shape[:2]
        cv2.rectangle(out, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (255, 120, 0), 1)
        if moving:
            cv2.putText(out, "CAMERA MOVING", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with self._cond:
                self._jpeg = buf.tobytes()
                self._jpeg_id += 1
                self._cond.notify_all()

    def stop(self):
        if self._server:
            self._server.shutdown()

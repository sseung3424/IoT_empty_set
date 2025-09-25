# web_stream.py
from flask import Flask, Response, render_template_string
from fall_detection import mjpeg_generator

app = Flask(__name__)

INDEX = """
<!doctype html><html><head><meta charset="utf-8"><title>Pi Live</title>
<style>body{margin:0;background:#111;color:#eee;font-family:sans-serif}
.wrap{max-width:960px;margin:0 auto;padding:8px}</style></head>
<body><div class="wrap">
<h2>Live</h2>
<img src="/video.mjpg" style="width:100%;max-width:960px"/>
</div></body></html>
"""

@app.route("/")
def index():
    return render_template_string(INDEX)

@app.route("/video.mjpg")
def video():
    return Response(mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

def start_https_server(host="0.0.0.0", port=8443, cert="cert.pem", key="key.pem"):
    print(f"[https] serving at https://{host}:{port}")
    app.run(host=host, port=port, ssl_context=(cert, key), threaded=True)

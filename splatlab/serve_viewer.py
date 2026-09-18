"""Serve only the viewer directory on localhost. Ctrl+C stops it.

python -m splatlab.serve_viewer
"""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port",type=int,default=8765)
    args=parser.parse_args()
    directory=Path(__file__).resolve().parents[1]/"viewer"
    if not (directory/"node_modules"/"@sparkjsdev"/"spark").exists():
        parser.error("Install viewer dependencies first: cd viewer; npm ci")
    server=ThreadingHTTPServer(("127.0.0.1",args.port),partial(Handler,directory=str(directory)))
    print(f"Open http://127.0.0.1:{args.port}/  (Ctrl+C to stop)",flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__ == "__main__": main()

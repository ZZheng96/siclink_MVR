"""Interactive spike-glyph viewer. Run: .venv\\Scripts\\python.exe -B test\\92_viewer\\server.py"""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import viewer_data as v  # noqa: E402

HOST, PORT = "127.0.0.1", 8765
BIN_MS = v.bin_ms()
_png_lock = threading.Lock()
_png_cache = {}


def meta():
    blocks = v.catalog()["blocks"]
    test = {}
    for b in blocks:
        if b["split"] == "test":
            test.setdefault(b["task"], []).append({"mouse": b["mouse"], "session": b["session"], "n_samples": b["n_samples"]})
    for t in test:
        test[t].sort(key=lambda x: (x["mouse"], x["session"]))
    return {"bin_ms": BIN_MS, "n_pictures": int(len(v.pictures())), "test": test,
            "train": [{"mouse": m, "session": s} for m, s in v.train_keys()],
            "window_ms": v.catalog()["window_ms"], "sample_rate_hz": v.catalog()["sample_rate_hz"]}


def heldout_list(task):
    h = v.heldout(task)
    keep = ("row", "mouse", "session", "target", "ocf", "oracle_ocf", "pred_index", "correct", "sample_ids")
    return {"task": task, "sequence_ok": h["sequence_ok"], "mean": h["mean"], "recorded_mean": h["recorded_mean"],
            "oracle_mean": h["oracle_mean"], "has_oracle": h["oracle"] is not None,
            "unresolved": sum(r["trials"] is None for r in h["rows"]),
            "rows": [{k: r[k] for k in keep} | {"resolved": r["trials"] is not None} for r in h["rows"]]}


def heldout_row(task, row):
    h = v.heldout(task)
    r = h["rows"][row]
    out = dict(r)
    out.update(task=task, bin_ms=BIN_MS[task], n_units=v.load_block("train", "train", r["mouse"], r["session"])["n_units"])
    if r["trials"] is None:
        out["events"] = None
        out["message"] = "无法确定来源 trial"
    else:
        out["events"], _ = v.trial_events("train", "train", r["mouse"], r["session"], r["trials"])
    return out


def test_list(task, mouse, session):
    d = v.load_block("test", task, mouse, session)
    sids = [str(s) for s in d["sample_ids"]]
    reps = np.diff(d["group_ptr"]).tolist()
    return {"task": task, "mouse": mouse, "session": session, "n_units": d["n_units"],
            "samples": [{"index": i, "sample_id": s, "n_trials": n,
                         "has_png": (v.SUBMISSION_PNG / task / f"{s}.png").exists()} for i, (s, n) in enumerate(zip(sids, reps))]}


def test_sample(task, mouse, session, index):
    trials = v.sample_trials("test", task, mouse, session, index)
    events, n_units = v.trial_events("test", task, mouse, session, trials)
    sid = str(v.load_block("test", task, mouse, session)["sample_ids"][index])
    return {"task": task, "mouse": mouse, "session": session, "index": index, "sample_id": sid, "n_units": n_units,
            "bin_ms": BIN_MS[task], "trials": trials, "events": events,
            "png": f"/img/submission?task={task}&sid={sid}" if (v.SUBMISSION_PNG / task / f"{sid}.png").exists() else None}


def train_glyph(glyph, with_events):
    blocks = v.train_trials_for_glyph(glyph)
    for b in blocks:
        b["n_units"] = v.load_block("train", "train", b["mouse"], b["session"])["n_units"]
        if with_events:
            b["events"], _ = v.trial_events("train", "train", b["mouse"], b["session"], b["trials"])
    return {"glyph": glyph, "blocks": blocks}


def image(kind, q):
    key = (kind, tuple(sorted((k, tuple(x)) for k, x in q.items())))
    with _png_lock:
        if key in _png_cache:
            return _png_cache[key]
    if kind == "glyph":
        data = v.png_bytes(v.glyph_display(v.pictures()[int(q["i"][0])]))
    elif kind == "submission":
        task, sid = q["task"][0], q["sid"][0]
        if task not in v.TASKS or "/" in sid or "\\" in sid or ".." in sid:
            raise KeyError("bad sid")
        data = (v.SUBMISSION_PNG / task / f"{sid}.png").read_bytes()
    else:
        task, row = q["task"][0], int(q["row"][0])
        h = v.heldout(task)
        if kind == "gt":
            data = v.png_bytes(v.glyph_display(h["gt"][row]))
        elif kind == "pred":
            data = v.png_bytes(v.glyph_display(h["pred"][row]))
        elif kind == "oracle" and h["oracle"] is not None:
            data = v.png_bytes(v.glyph_display(h["oracle"][row]))
        elif kind == "diff":
            data = v.png_bytes(v.diff_image(h["gt"][row], h["pred"][row]))
        elif kind == "odiff" and h["oracle"] is not None:
            data = v.png_bytes(v.diff_image(h["gt"][row], h["oracle"][row]))
        else:
            raise KeyError(kind)
    with _png_lock:
        _png_cache[key] = data
    return data


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache" if ctype.startswith("text/html") else "max-age=3600")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        p = url.path
        try:
            if p in ("/", "/index.html"):
                return self.send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
            if p.startswith("/img/"):
                return self.send(200, image(p[5:], q), "image/png")
            task = q.get("task", [None])[0]
            if task is not None and task not in v.TASKS:
                raise KeyError("task")
            if p == "/api/meta":
                data = meta()
            elif p == "/api/heldout":
                data = heldout_list(task)
            elif p == "/api/heldout_row":
                data = heldout_row(task, int(q["row"][0]))
            elif p == "/api/test":
                data = test_list(task, q["mouse"][0], q["session"][0])
            elif p == "/api/test_sample":
                data = test_sample(task, q["mouse"][0], q["session"][0], int(q["index"][0]))
            elif p == "/api/train_glyph":
                data = train_glyph(int(q["glyph"][0]), q.get("events", ["1"])[0] == "1")
            else:
                return self.send(404, b'{"error":"not found"}', "application/json")
            body = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            return self.send(200, body, "application/json; charset=utf-8")
        except (KeyError, IndexError, ValueError, FileNotFoundError) as e:
            body = json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False).encode("utf-8")
            return self.send(400, body, "application/json; charset=utf-8")


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"viewer: http://{HOST}:{PORT}/", flush=True)
    threading.Thread(target=lambda: [v.heldout(t) for t in v.TASKS], daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

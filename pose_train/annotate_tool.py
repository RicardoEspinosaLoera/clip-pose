"""
annotate_tool.py — minimal annotation GUI for the real evaluation frames (protocol section 3).

Per frame: a clip bounding box (2 clicks), three keypoints (jaw tip, spring end, coil centre),
and four attributes (held by applier, jaws open, partially occluded, blurry). One JSON per frame
in real_eval/annotations/<annotator>/<frame>.json; resumable; the detector box is shown as a hint.

Keys
  left click   : place the next point (box corner 1, box corner 2, tip, spring, coil)
  u            : undo last point          r : reset frame
  h / o / p / b: toggle held / open / occluded / blurry
  n / space    : save and next            p (prev) : BACKSPACE
  q            : save and quit
Usage: python annotate_tool.py --annotator ricardo [--start 0]
"""

import os
import json
import argparse

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "real_eval"))
STEPS = ["box corner 1", "box corner 2", "jaw TIP", "SPRING end", "COIL centre"]
VIEW_W = 1600


class State:
    def __init__(self):
        self.pts, self.attrs = [], dict(held=False, open=False, occluded=False, blurry=False)
        self.zoom_box = None


def draw(img, st, hint, name, idx, total):
    vis = img.copy()
    if hint is not None:
        x, y, w, h = hint
        cv2.rectangle(vis, (x, y), (x + w, y + h), (80, 80, 80), 1)
    cols = [(0, 255, 255), (0, 255, 255), (0, 0, 255), (255, 0, 255), (255, 128, 0)]
    for i, p in enumerate(st.pts):
        cv2.circle(vis, (int(p[0]), int(p[1])), 5, cols[i], -1)
    if len(st.pts) >= 2:
        (x0, y0), (x1, y1) = st.pts[0], st.pts[1]
        cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 255), 2)
    if len(st.pts) >= 4:
        cv2.line(vis, (int(st.pts[2][0]), int(st.pts[2][1])), (int(st.pts[3][0]), int(st.pts[3][1])), (255, 255, 255), 1)
    nxt = STEPS[len(st.pts)] if len(st.pts) < len(STEPS) else "done: n = next"
    txt = f"[{idx + 1}/{total}] {name} | next: {nxt} | held(h)={int(st.attrs['held'])} open(o)={int(st.attrs['open'])} occl(p)={int(st.attrs['occluded'])} blur(b)={int(st.attrs['blurry'])}"
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(vis, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotator", required=True)
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()
    frames = json.load(open(os.path.join(OUT, "frames.json")))
    out_dir = os.path.join(OUT, "annotations", a.annotator)
    os.makedirs(out_dir, exist_ok=True)
    idx = a.start
    while idx < len(frames) and os.path.exists(os.path.join(out_dir, frames[idx]["file"] + ".json")):
        idx += 1
    cv2.namedWindow("annotate", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("annotate", VIEW_W, int(VIEW_W * 9 / 16))
    st = State()
    scale = {"s": 1.0}

    def on_mouse(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN and len(st.pts) < len(STEPS):
            st.pts.append((x / scale["s"], y / scale["s"]))

    cv2.setMouseCallback("annotate", on_mouse)
    while 0 <= idx < len(frames):
        fr = frames[idx]
        img = cv2.imread(os.path.join(OUT, "frames", fr["file"]))
        H, W = img.shape[:2]
        scale["s"] = VIEW_W / W
        st = State()
        prev = os.path.join(out_dir, fr["file"] + ".json")
        if os.path.exists(prev):
            d = json.load(open(prev))
            st.pts = [tuple(p) for p in d["points"]]
            st.attrs = d["attrs"]
        while True:
            vis = draw(img, st, fr.get("hint_box"), fr["file"], idx, len(frames))
            cv2.imshow("annotate", cv2.resize(vis, (VIEW_W, int(H * scale["s"]))))
            k = cv2.waitKey(30) & 0xFF
            if k == ord("u") and st.pts:
                st.pts.pop()
            elif k == ord("r"):
                st = State()
            elif k == ord("h"):
                st.attrs["held"] = not st.attrs["held"]
            elif k == ord("o"):
                st.attrs["open"] = not st.attrs["open"]
            elif k == ord("p"):
                st.attrs["occluded"] = not st.attrs["occluded"]
            elif k == ord("b"):
                st.attrs["blurry"] = not st.attrs["blurry"]
            elif k in (ord("n"), ord(" "), ord("q"), 8):
                if len(st.pts) == len(STEPS):
                    (x0, y0), (x1, y1) = st.pts[0], st.pts[1]
                    rec = dict(file=fr["file"], src=fr["src"], video=fr["video"], width=W, height=H,
                               box=[min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)],
                               points={"tip": st.pts[2], "spring": st.pts[3], "coil": st.pts[4]},
                               attrs=st.attrs, annotator=a.annotator)
                    json.dump(rec, open(prev, "w"), indent=1)
                if k == ord("q"):
                    cv2.destroyAllWindows()
                    return
                idx += -1 if k == 8 else 1
                break
    cv2.destroyAllWindows()
    print("done")


if __name__ == "__main__":
    main()

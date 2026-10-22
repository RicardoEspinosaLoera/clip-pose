"""
annotate_tool.py — manual annotation of the real evaluation frames (protocol section 3).

Per frame: a clip bounding box, three keypoints (jaw TIP = midpoint of the two jaw tips,
SPRING end, COIL centre) and four attributes (held by applier, jaws open, partially occluded,
blurry). One JSON per frame in real_eval/annotations/<annotator>/<frame>.json; resumable.

Workflow per frame
  1. The detector box is PRE-LOADED (yellow). If it is right, go on; if not, press `c` and
     click two opposite corners of the clip box.
  2. The view ZOOMS onto the box automatically. Click TIP, SPRING, COIL (points are always
     marked by hand: never pre-filled, to avoid anchoring bias).
  3. Toggle the attributes, then `n` to save and go to the next frame.

Keys
  left click : next point            u : undo last point       r : reset frame (box pre-loaded again)
  c          : clear box, click a new one                      z : toggle zoom / full frame
  x          : back end hidden (skips the third point)
  e          : EXCLUDE frame (no clip, or clip impossible to annotate) -> saved as excluded, then next
  h / o / p / b / m : held / open / occluded / blurry / more than one clip in the frame
  several clips: annotate ONE -> the one held by the applier; else the pre-loaded box if it is a clip; else the most visible
  n or space : save + next           BACKSPACE : save + previous           q : save + quit
Usage: python annotate_tool.py --annotator ricardo [--start N] [--no-prefill]
       (without --start: resumes at the first frame not yet annotated; --start N opens frame N even if annotated)
"""

import os
import json
import argparse

import cv2
import numpy as np

from real_points import norm_points, KEYS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "real_eval"))
STEPS = ["box corner 1", "box corner 2", "1 jaw TIP (end of the jaws)", "2 clip HEAD centre (blue body)",
         "3 BACK end (x = hidden)"]
COLS = [(0, 255, 255), (0, 255, 255), (0, 0, 255), (255, 128, 0), (255, 0, 255)]   # tip red, head blue, back magenta
VIEW_W, VIEW_H = 1600, 900
ZOOM_MARGIN = 2.2          # zoom window = box size x margin (keeps context around the clip)


class State:
    def __init__(self, hint=None, prefill=True):
        self.pts = []
        self.attrs = dict(held=False, open=False, occluded=False, blurry=False, multi=False)
        if prefill and hint is not None:
            x, y, w, h = hint
            self.pts = [(float(x), float(y)), (float(x + w), float(y + h))]
        self.zoom = True
        self.excluded = False


def view_transform(st, W, H):
    """Affine (scale s, offset ox, oy): view = (img - o) * s. Zoom on the box once it exists."""
    if st.zoom and len(st.pts) >= 2:
        (x0, y0), (x1, y1) = st.pts[0], st.pts[1]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        bw, bh = max(abs(x1 - x0), 20), max(abs(y1 - y0), 20)
        cw = max(bw * ZOOM_MARGIN, bh * ZOOM_MARGIN * VIEW_W / VIEW_H, 240)
        s = VIEW_W / cw
        return s, cx - (VIEW_W / 2) / s, cy - (VIEW_H / 2) / s
    s = min(VIEW_W / W, VIEW_H / H)
    return s, 0.0, 0.0


def render(img, st, name, idx, total):
    H, W = img.shape[:2]
    s, ox, oy = view_transform(st, W, H)
    A = np.array([[s, 0, -ox * s], [0, s, -oy * s]], np.float64)
    vis = cv2.warpAffine(img, A, (VIEW_W, VIEW_H), flags=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 40, 40))

    def P(p):
        return int(round((p[0] - ox) * s)), int(round((p[1] - oy) * s))
    if len(st.pts) >= 2:
        cv2.rectangle(vis, P(st.pts[0]), P(st.pts[1]), (0, 255, 255), 1)
    if len(st.pts) >= 4:
        cv2.line(vis, P(st.pts[2]), P(st.pts[3]), (255, 255, 255), 1)
    for i, p in enumerate(st.pts):
        if i < 2 or p is None:
            continue
        cv2.circle(vis, P(p), 7, (0, 0, 0), 3)
        cv2.circle(vis, P(p), 7, COLS[i], 1)
        cv2.circle(vis, P(p), 1, COLS[i], -1)
    nxt = STEPS[len(st.pts)] if len(st.pts) < len(STEPS) else "done -> n = save + next"
    if getattr(st, "excluded", False) and not st.pts:
        nxt = "EXCLUDED (no clip). Click to annotate it anyway, or n = next"
    a = st.attrs
    l1 = f"[{idx + 1}/{total}] {name}   zoom x{s:.1f}   next: {nxt}"
    l2 = (f"held(h)={int(a['held'])}  open(o)={int(a['open'])}  occl(p)={int(a['occluded'])}  blur(b)={int(a['blurry'])}  multi(m)={int(a['multi'])}"
          f"   |  c new box  u undo  r reset  z zoom  e no clip  q quit")
    cv2.rectangle(vis, (0, 0), (VIEW_W, 56), (0, 0, 0), -1)
    cv2.putText(vis, l1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(vis, l2, (10, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return vis, (s, ox, oy)


def load_previous(path):
    d = json.load(open(path))
    if d.get("excluded"):
        st = State(prefill=False)
        st.excluded = True
        return st
    x, y, w, h = d["box"]
    st = State(prefill=False)
    P = norm_points(d["points"])
    st.pts = [(x, y), (x + w, y + h)] + [tuple(P[k]) if P[k] is not None else None for k in KEYS]
    st.attrs = dict(dict(held=False, open=False, occluded=False, blurry=False, multi=False), **d["attrs"])
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotator", required=True)
    ap.add_argument("--start", type=int, default=None, help="open frame N (1-based, as shown in the window), annotated or not")
    ap.add_argument("--no-prefill", action="store_true", help="do not pre-load the detector box")
    a = ap.parse_args()
    frames = json.load(open(os.path.join(OUT, "frames.json")))
    out_dir = os.path.join(OUT, "annotations", a.annotator)
    os.makedirs(out_dir, exist_ok=True)
    if a.start is not None:                      # jump to an exact frame (also to review an annotated one)
        idx = min(max(a.start - 1, 0), len(frames) - 1)
    else:                                        # resume at the first frame not yet annotated
        idx = 0
        while idx < len(frames) and os.path.exists(os.path.join(out_dir, frames[idx]["file"] + ".json")):
            idx += 1
    done = sum(os.path.exists(os.path.join(out_dir, f["file"] + ".json")) for f in frames)
    print(f"{done}/{len(frames)} frames already annotated by {a.annotator}; starting at {idx + 1}")
    cv2.namedWindow("annotate", cv2.WINDOW_AUTOSIZE)
    view = {"T": (1.0, 0.0, 0.0)}
    holder = {}

    def on_mouse(ev, x, y, flags, _):
        st = holder.get("st")
        if st is not None and ev == cv2.EVENT_LBUTTONDOWN and len(st.pts) < len(STEPS):
            s, ox, oy = view["T"]
            st.pts.append((x / s + ox, y / s + oy))

    cv2.setMouseCallback("annotate", on_mouse)
    while 0 <= idx < len(frames):
        fr = frames[idx]
        img = cv2.imread(os.path.join(OUT, "frames", fr["file"]))
        H, W = img.shape[:2]
        prev = os.path.join(out_dir, fr["file"] + ".json")
        hint = None if a.no_prefill else fr.get("hint_box")
        st = load_previous(prev) if os.path.exists(prev) else State(hint)
        holder["st"] = st
        while True:
            vis, view["T"] = render(img, st, fr["file"], idx, len(frames))
            cv2.imshow("annotate", vis)
            k = cv2.waitKey(30) & 0xFF
            if k == ord("u") and st.pts:
                st.pts.pop()
            elif k == ord("r"):
                st = State(hint)
                holder["st"] = st
            elif k == ord("c"):
                st.pts = []
            elif k == ord("x") and len(st.pts) == 4:
                st.pts.append(None)
            elif k == ord("e"):
                json.dump(dict(file=fr["file"], src=fr["src"], video=fr["video"], width=W, height=H, excluded=True,
                               reason="no clip / not annotatable", annotator=a.annotator), open(prev, "w"), indent=1)
                print(f"  {fr['file']}: excluded")
                idx += 1
                break
            elif k == ord("z"):
                st.zoom = not st.zoom
            elif k in (ord("h"), ord("o"), ord("p"), ord("b"), ord("m")):
                key = {ord("h"): "held", ord("o"): "open", ord("p"): "occluded", ord("b"): "blurry", ord("m"): "multi"}[k]
                st.attrs[key] = not st.attrs[key]
            elif k in (ord("n"), ord(" "), ord("q"), 8):
                if len(st.pts) == len(STEPS):
                    (x0, y0), (x1, y1) = st.pts[0], st.pts[1]
                    rec = dict(file=fr["file"], src=fr["src"], video=fr["video"], width=W, height=H,
                               box=[min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)],
                               points={"tip": list(st.pts[2]), "head": list(st.pts[3]),
                                       "back": list(st.pts[4]) if st.pts[4] is not None else None}, scheme=2,
                               attrs=st.attrs, annotator=a.annotator, box_prefilled=hint is not None)
                    json.dump(rec, open(prev, "w"), indent=1)
                elif k in (ord("n"), ord(" ")) and len(st.pts) > 2:
                    print(f"  {fr['file']}: incomplete ({len(st.pts) - 2}/3 points), not saved")
                if k == ord("q"):
                    cv2.destroyAllWindows()
                    return
                idx += -1 if k == 8 else 1
                break
    cv2.destroyAllWindows()
    done = sum(os.path.exists(os.path.join(out_dir, f["file"] + ".json")) for f in frames)
    print(f"done: {done}/{len(frames)} frames annotated -> {out_dir}")


if __name__ == "__main__":
    main()

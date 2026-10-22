"""
make_annotation_kit.py — zip for external annotators: the annotation tool, the 200 real frames,
the guide and one-click launchers. Existing annotations are NOT included (independent annotation).

Usage: python make_annotation_kit.py [--out C:/Users/ricar/realac_bundle]
"""

import os
import time
import zipfile
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
KIT = os.path.join(ROOT, "real_eval", "kit")
REF_IMG = os.path.join(ROOT, "GT_Generator", "_keypoints_check.jpg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.expanduser("~"), "realac_bundle"))
    ap.add_argument("--set", default=os.path.join(ROOT, "real_eval"), help="frame set to ship (real_eval or real_eval_v2)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    name = f"kit_anotacion_clips_{time.strftime('%Y%m%d')}"
    zp = os.path.join(a.out, name + ".zip")
    frames_dir = os.path.join(a.set, "frames")
    frames = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith(".jpg"))
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in sorted(os.listdir(KIT)):
            p = os.path.join(KIT, fn)
            if fn.endswith(".py"):
                continue
            if fn.endswith(".bat"):                       # cmd.exe needs CRLF for labels / goto
                txt = open(p, encoding="utf-8").read().replace("\r\n", "\n").replace("\n", "\r\n")
                z.writestr(f"{name}/{fn}", txt)
            else:
                z.write(p, f"{name}/{fn}")
        z.write(os.path.join(HERE, "annotate_tool.py"), f"{name}/pose_train/annotate_tool.py")
        z.write(os.path.join(HERE, "real_points.py"), f"{name}/pose_train/real_points.py")
        z.write(os.path.join(a.set, "frames.json"), f"{name}/real_eval/frames.json")
        for fn in frames:
            z.write(os.path.join(frames_dir, fn), f"{name}/real_eval/frames/{fn}")
    print(f"{zp}  ({os.path.getsize(zp) / 1e6:.1f} MB, {len(frames)} frames, no existing annotations)")


if __name__ == "__main__":
    main()

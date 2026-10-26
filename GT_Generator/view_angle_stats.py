"""
view_angle_stats.py -- how foreshortened are the synthetic clips? (pose prior vs the real frames)
For every GT JSON, the clip's long axis (model x; jaws at x=0, spring at x=15.46 in clip_lazic_std.obj, so
-x points to the jaws) is rotated into the camera frame with clip.pose_se3 and its angle to the view axis
(+Z) is reported: 90 deg = the clip lies in the image plane (seen side-on, the whole blade length visible),
0 deg = seen end-on. Also the share of samples whose jaws point away from the camera (BLADE_AWAY).
25 Oct 2026: out_lclip_tune (18 samples) gave median 35 deg, only 11 % above 60 deg, whereas the real_eval_v2
frames show the clip side-on in most frames -> the real views are out of the synthetic distribution.
Usage: python view_angle_stats.py <set_dir> [<set_dir> ...]
"""
import os
import sys
import json
import glob
import numpy as np
from scipy.spatial.transform import Rotation as Rr


def stats(d):
    angs, tip_away = [], []
    for j in sorted(glob.glob(os.path.join(d, "[0-9]*.json"))):
        try:
            p = json.load(open(j))["clip"]["pose_se3"]
        except (KeyError, ValueError):
            continue
        w, x, y, z = p["quaternion_wxyz"]
        R = Rr.from_quat([x, y, z, w]).as_matrix()
        jaws = R @ np.array([-1.0, 0.0, 0.0])
        angs.append(np.degrees(np.arccos(abs(jaws[2]))))
        tip_away.append(jaws[2] > 0)
    a = np.array(angs)
    if not len(a):
        return f"{d}: no pose JSONs"
    hist = np.histogram(a, bins=[0, 15, 30, 45, 60, 75, 90])[0] / len(a) * 100
    return (f"{os.path.basename(os.path.normpath(d))}: n={len(a)}  axis-vs-view angle p25 {np.percentile(a, 25):.0f} "
            f"med {np.median(a):.0f} p75 {np.percentile(a, 75):.0f} deg | in-plane (>60): {(a > 60).mean() * 100:.0f}% | "
            f"jaws away from camera: {np.mean(tip_away) * 100:.0f}%\n    bins 0-15 15-30 30-45 45-60 60-75 75-90 deg: "
            + " ".join(f"{h:4.0f}%" for h in hist))


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(stats(d))

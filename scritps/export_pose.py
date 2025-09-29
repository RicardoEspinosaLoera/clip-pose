#!/usr/bin/env python3
"""
Export regressor predictions on a folder to a CSV.
Outputs: stem, r11...r33, tx, ty, tz
"""
import argparse, glob, os, csv, torch
from torch.utils.data import DataLoader
from src.datasets import TripletDataset
from src.models import Regressor
from src.camera import sixd_to_rotmat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/val")
    ap.add_argument("--weights", required=True, help="Trained model checkpoint")
    ap.add_argument("--out_csv", default="preds.csv")
    args = ap.parse_args()

    ds = TripletDataset(args.data_root)
    dl = DataLoader(ds, batch_size=8, shuffle=False)
    model = Regressor().cuda()
    model.load_state_dict(torch.load(args.weights))
    model.eval()

    with open(args.out_csv,"w",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stem"] + [f"r{i}{j}" for i in range(3) for j in range(3)] + ["tx","ty","tz"])
        for batch in dl:
            I = batch["image"].cuda()
            r6, t = model(I)
            R = sixd_to_rotmat(r6).cpu().numpy()
            t = t.cpu().numpy()
            for i, stem in enumerate(batch["stem"]):
                row = [stem] + R[i].flatten().tolist() + t[i].tolist()
                writer.writerow(row)

if __name__ == "__main__":
    main()

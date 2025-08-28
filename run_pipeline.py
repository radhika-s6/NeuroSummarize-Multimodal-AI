# run_pipeline.py
from src.summarize import Summarizer
import argparse
import os
import json
from pathlib import Path

def process_and_save_predictions(in_dir="data/reports", out_dir="data/gold/pending", clin_model=None, lay_model=None, device=None):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    summarizer = Summarizer(clinical_model=clin_model or "sshleifer/distilbart-cnn-12-6",
                            lay_model=lay_model or "facebook/bart-large-cnn",
                            device=device)
    n = 0
    for fname in sorted(os.listdir(in_dir)):
        if not fname.lower().endswith(".txt"):
            continue
        in_path = os.path.join(in_dir, fname)
        with open(in_path, "r", encoding="utf-8") as fh:
            text = fh.read()
        base = os.path.splitext(fname)[0]
        out = summarizer.summarize_both(text)
        obj = {
            "id": base,
            "report": text,
            "clinical_summary_model": out.get("clinical_summary", ""),
            "lay_summary_model": out.get("lay_summary", ""),
            "pred_entities": out.get("pred_entities", [])
        }
        with open(os.path.join(out_dir, f"{base}_pending.json"), "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2)
        n += 1
    print(f"Saved {n} prediction files to {out_dir}")


def main(engine="tesseract", device=None, clin_model=None, lay_model=None, save_predictions=False):
    if save_predictions:
        process_and_save_predictions(in_dir="data/reports",
                                     out_dir="data/gold/pending",
                                     clin_model=clin_model,
                                     lay_model=lay_model,
                                     device=device)
    else:
        print("No action requested. Use --save_predictions to generate prediction JSONs for manual annotation.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="tesseract")
    ap.add_argument("--device", default=None)
    ap.add_argument("--clin_model", default=None)
    ap.add_argument("--lay_model", default=None)
    ap.add_argument("--save_predictions", action="store_true", help="Run summarizer on data/reports and save JSONs to data/gold/pending")
    args = ap.parse_args()
    main(**vars(args))

import os, pandas as pd
from pathlib import Path

DATA_ROOT = r"D:\MSc_AI\MSc_Project\NeuroSummarize\data"
OUT_DIR = "data/reports"

def find_participants_tsv(root):
    """Return the first participants.tsv or participants.tsv.gz found"""
    for dirpath, _, files in os.walk(root):
        for fname in files:
            if fname.lower() in ("participants.tsv", "participants.tsv.gz"):
                return os.path.join(dirpath, fname)
    return None

def main():
    part_file = find_participants_tsv(DATA_ROOT)
    if part_file is None:
        raise FileNotFoundError(f"participants.tsv not found under {DATA_ROOT}")

    print(f"Using participants file: {part_file}")
    part = pd.read_csv(part_file, sep="\t", compression="infer")

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    for _, row in part.iterrows():
        sid = str(row.get("participant_id", row.get("subject_id","sub-xxx")))
        dx  = row.get("diagnosis", row.get("group","unknown"))
        age = row.get("age","NA")
        sex = row.get("sex","NA")
        text = (
            f"Patient: {sid}\n"
            f"Modality: MRI (structural + fMRI where available)\n"
            f"Diagnosis/Group: {dx}\n"
            f"Age: {age}; Sex: {sex}\n"
            f"Findings: No acute intracranial hemorrhage reported in dataset metadata.\n"
            f"Regions of interest: frontal lobe, temporal lobe (based on study tasks/phenotypes).\n"
            f"Recommendation: Correlate imaging with clinical phenotype.\n"
        )
        with open(os.path.join(OUT_DIR, f"{sid}.txt"), "w", encoding="utf-8") as f:
            f.write(text)

    print(f"Generated {len(part)} templated reports in {OUT_DIR}")

if __name__ == "__main__":
    main()

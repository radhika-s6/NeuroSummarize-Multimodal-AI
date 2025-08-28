# src/evaluate_models.py
"""
Evaluation driver to run multiple summarizers (HF, openai, ollama) on a dataset,
compute ROUGE-L, BLEU, entity-level P/R/F1 (RadGraph-style), hallucination rate,
Likert clinical utility averages, and create comparative plots.

Use:
    python -m src.evaluate_models --data_dir data/gold --models "openai:gpt-4o,sshleifer/distilbart-cnn-12-6" --output_dir results/eval --max_samples 200
"""
import os
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple
import csv

# metrics
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics import precision_recall_fscore_support

# plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statistics

# summarizer
from src.summarize import Summarizer

# embedding-based utils
_embedder = None
try:
    from sentence_transformers import SentenceTransformer, util
    _embedder = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    _embedder = None
    # warn later if used

nltk.download('punkt', quiet=True)


def load_gold_dataset(data_dir: Path) -> List[Dict]:
    """
    Load gold dataset items from a directory.
    Supports .json / .JSON / .jsonl / .JSONL files.
    Recursively searches all subdirectories.
    """
    items = []
    # search recursively for any .json or .jsonl file (case-insensitive)
    for p in data_dir.rglob("*"):
        if p.suffix.lower() == ".json":
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    items.append(json.load(fh))
            except Exception as e:
                print(f"[WARN] Could not load {p}: {e}")
        elif p.suffix.lower() == ".jsonl":
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            items.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] Could not load {p}: {e}")
    print(f"Loaded {len(items)} items from {data_dir}")
    return items


def normalize_text(t: str) -> str:
    if t is None:
        return ""
    return " ".join(nltk.word_tokenize(t.lower()))


def rouge_l_score(ref: str, hyp: str) -> float:
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    score = scorer.score(ref or "", hyp or "")
    return float(score['rougeL'].fmeasure)


def bleu_score(ref: str, hyp: str) -> float:
    ref_tokens = nltk.word_tokenize((ref or "").lower())
    hyp_tokens = nltk.word_tokenize((hyp or "").lower())
    if len(hyp_tokens) == 0:
        return 0.0
    chencherry = SmoothingFunction()
    return float(sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=chencherry.method1))


# entity helpers (RadGraph-aware matching)
def _normalize_ent_text(t: str) -> str:
    if not t:
        return ""
    return " ".join(t.lower().strip().split())


def _radgraph_type_normalize(t: str) -> str:
    if not t:
        return "UNKNOWN"
    s = str(t).lower()
    if "anat" in s or "anatom" in s:
        return "ANATOMY"
    if "obs" in s or "observ" in s or "finding" in s:
        return "OBSERVATION"
    if "later" in s or "left" in s or "right" in s:
        return "LATERALITY"
    return s.upper()


def entity_level_scores(gold_entities: List[Dict], pred_entities: List[Dict]) -> Tuple[float, float, float]:
    # normalize lists of dicts
    gold_items = []
    for g in (gold_entities or []):
        txt = g.get("text", "") if isinstance(g, dict) else str(g)
        gold_items.append({"text": _normalize_ent_text(txt), "type": _radgraph_type_normalize(g.get("type", "") if isinstance(g, dict) else "") , "span": g.get("span") if isinstance(g, dict) else None})
    pred_items = []
    for p in (pred_entities or []):
        if isinstance(p, dict):
            txt = p.get("text", "")
            typ = p.get("type", "")
            span = p.get("span", None)
        else:
            txt = str(p)
            typ = ""
            span = None
        pred_items.append({"text": _normalize_ent_text(txt), "type": _radgraph_type_normalize(typ), "span": span})

    if not gold_items and not pred_items:
        return 1.0, 1.0, 1.0

    gold_matched = set()
    pred_matched = set()

    # embeddings precompute if possible
    use_emb = False
    if _embedder and gold_items and pred_items:
        try:
            gold_texts = [g["text"] for g in gold_items]
            pred_texts = [p["text"] for p in pred_items]
            gold_emb = _embedder.encode(gold_texts, convert_to_tensor=True)
            pred_emb = _embedder.encode(pred_texts, convert_to_tensor=True)
            use_emb = True
        except Exception:
            use_emb = False

    for i, p in enumerate(pred_items):
        best_j = None
        best_score = 0.0
        for j, g in enumerate(gold_items):
            if j in gold_matched:
                continue
            # coarse type must match (or at least not be contradictory)
            if p["type"] != "UNKNOWN" and g["type"] != "UNKNOWN" and p["type"] != g["type"]:
                continue
            # prefer exact span match
            if p.get("span") and g.get("span") and p["span"] == g["span"]:
                best_j = j
                best_score = 1.0
                break
            # exact text match
            if p["text"] and g["text"] and p["text"] == g["text"]:
                best_j = j
                best_score = max(best_score, 0.95)
                continue
            # embedding similarity
            if use_emb:
                sim = float(util.cos_sim(pred_emb[i], gold_emb[j]).item())
                if sim > best_score:
                    best_score = sim
                    best_j = j
            else:
                # jaccard on token sets
                set_p = set(p["text"].split())
                set_g = set(g["text"].split())
                if set_p or set_g:
                    jacc = len(set_p & set_g) / max(1, len(set_p | set_g))
                else:
                    jacc = 0.0
                if jacc > best_score:
                    best_score = jacc
                    best_j = j
        # threshold decision
        thresh = 0.65 if not use_emb else 0.70
        if best_j is not None and (best_score >= thresh or best_score >= 0.95):
            gold_matched.add(best_j)
            pred_matched.add(i)

    tp = len(pred_matched)
    fp = len(pred_items) - tp
    fn = len(gold_items) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0
    return float(precision), float(recall), float(f1)


def parse_entities_from_summary(summary_text: str, pred_entities: List[Dict] = None) -> List[Dict]:
    # prefer structured pred_entities from summarizer
    if pred_entities:
        parsed = []
        for e in pred_entities:
            if isinstance(e, dict):
                parsed.append({"text": e.get("text", ""), "type": e.get("type", ""), "span": e.get("span", None)})
            else:
                parsed.append({"text": str(e), "type": "", "span": None})
        return parsed
    if not summary_text:
        return []
    chunks = [c.strip() for c in summary_text.replace(';', ',').split(',') if c.strip()]
    return [{'text': c, 'type': '', 'span': None} for c in chunks[:50]]


def hallucination_rate(gold_entities: List[Dict], pred_entities: List[Dict], emb_threshold: float = 0.7) -> float:
    if not pred_entities:
        return 0.0
    # embedding semantic-check if possible
    try:
        if _embedder and gold_entities:
            gold_texts = [g.get("text", "") for g in gold_entities]
            pred_texts = [p.get("text", "") for p in pred_entities]
            gold_emb = _embedder.encode(gold_texts, convert_to_tensor=True)
            pred_emb = _embedder.encode(pred_texts, convert_to_tensor=True)
            hallucinated = 0
            for i in range(len(pred_texts)):
                sims = util.cos_sim(pred_emb[i], gold_emb)
                max_sim = float(sims.max()) if len(gold_emb) > 0 else 0.0
                if max_sim < emb_threshold:
                    hallucinated += 1
            return float(hallucinated) / len(pred_texts)
    except Exception:
        pass
    # fallback: strict normalized text non-overlap
    gold_norm = set([_normalize_ent_text(g.get("text", "")) for g in gold_entities or []])
    pred_norm = [_normalize_ent_text(p.get("text", "")) for p in pred_entities or []]
    hallucinated = sum(1 for t in pred_norm if t not in gold_norm)
    return float(hallucinated) / len(pred_norm)


def evaluate_one_model_on_dataset(model_id: str, items: List[Dict], summarizer,
                                  max_samples: int = None, save_dir: Path = None) -> Dict:
    """
    Evaluate a summarizer on a list of items, returning aggregated metrics.

    Defensive / robustness improvements:
    - Initialize `row` early so it cannot raise UnboundLocalError.
    - Use safe lookups for etype score values.
    - Capture and continue on per-item summarization exceptions.
    - When saving CSV, include the union of all keys (so dynamic `entity_f_*` columns are preserved).
    """
    results_rows = []
    agg = {
        'model': model_id,
        'n': 0,
        'rouge_l_clin': [], 'bleu_clin': [],
        'rouge_l_lay': [], 'bleu_lay': [],
        'entity_p': [], 'entity_r': [], 'entity_f': [],
        'hall_rate': [], 'likert': []
    }

    for idx, item in enumerate(items):
        if max_samples and idx >= max_samples:
            break

        # --- Basic case fields ---
        doc_id = item.get("id", f"case_{idx}")
        text = item.get("report", "")
        gold_clin = item.get("clinical_summary_gold", "")
        gold_lay = item.get("lay_summary_gold", "")
        gold_entities = item.get("entities_gold", [])

        # Likert field (support two keys)
        likert_score = item.get("clinical_utility_score", item.get("clinical_utility_likert", None))
        if likert_score is not None:
            try:
                likert_score = float(likert_score)
            except Exception:
                likert_score = None

        # Initialize row early to avoid UnboundLocalError in any code path
        row: Dict[str, object] = {"model": model_id, "id": doc_id}

        # --- Summarize (defensive) ---
        try:
            out = summarizer.summarize_both(text)
        except Exception as e:
            # If summarizer fails for this item, record the error and continue
            row.update({
                "rouge_l_clin": None, "bleu_clin": None,
                "rouge_l_lay": None, "bleu_lay": None,
                "entity_p": None, "entity_r": None, "entity_f": None,
                "hallucination_rate": None,
                "clinical_utility_likert": likert_score,
                "pred_entities": json.dumps({"error": str(e)})
            })
            results_rows.append(row)
            # Do not include this item's metrics in the aggregates (alternatively, you could include zeros)
            continue

        pred_clin = out.get("clinical_summary", "")
        pred_lay = out.get("lay_summary", "")
        # prefer structured pred_entities returned by Summarizer
        pred_entities = out.get("pred_entities")
        if pred_entities is None:
            pred_entities = parse_entities_from_summary(pred_clin, None)

        # --- Compute metrics (assume functions exist in module scope) ---
        r_clin = rouge_l_score(gold_clin, pred_clin)
        b_clin = bleu_score(gold_clin, pred_clin)
        r_lay = rouge_l_score(gold_lay, pred_lay)
        b_lay = bleu_score(gold_lay, pred_lay)

        p, r, f = entity_level_scores(gold_entities, pred_entities)
        etype_scores = entity_level_scores_by_type(gold_entities, pred_entities)
        hrate = hallucination_rate(gold_entities, pred_entities)

        # --- Populate main row fields ---
        # Use json.dumps for pred_entities to make CSV-friendly
        try:
            pred_entities_serialized = json.dumps(pred_entities)
        except Exception:
            # If pred_entities is not JSON-serializable directly, coerce to string
            pred_entities_serialized = json.dumps({"value": str(pred_entities)})

        row.update({
            "rouge_l_clin": r_clin,
            "bleu_clin": b_clin,
            "rouge_l_lay": r_lay,
            "bleu_lay": b_lay,
            "entity_p": p,
            "entity_r": r,
            "entity_f": f,
            "hallucination_rate": hrate,
            "clinical_utility_likert": likert_score,
            "pred_entities": pred_entities_serialized
        })

        # --- Safely add entity-level f-scores (dynamic keys) ---
        if isinstance(etype_scores, dict):
            for etype, vals in etype_scores.items():
                key = f"entity_f_{str(etype).lower()}"
                # vals might be None or missing 'f'
                if isinstance(vals, dict):
                    row[key] = vals.get("f", None)
                else:
                    row[key] = None

        results_rows.append(row)

        # --- Update aggregates ---
        agg['n'] += 1
        if r_clin is not None:
            agg['rouge_l_clin'].append(r_clin)
        if b_clin is not None:
            agg['bleu_clin'].append(b_clin)
        if r_lay is not None:
            agg['rouge_l_lay'].append(r_lay)
        if b_lay is not None:
            agg['bleu_lay'].append(b_lay)
        if p is not None:
            agg['entity_p'].append(p)
        if r is not None:
            agg['entity_r'].append(r)
        if f is not None:
            agg['entity_f'].append(f)
        if hrate is not None:
            agg['hall_rate'].append(hrate)
        if likert_score is not None:
            agg['likert'].append(likert_score)

    # Save per-case CSV if requested (ensure union of all dynamic columns included)
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        per_case_csv = save_dir / f"{model_id.replace(':', '_')}_per_case.csv"
        if results_rows:
            # compute union of all keys for fieldnames to include dynamic entity_f_* columns
            all_keys = set()
            for r in results_rows:
                all_keys.update(r.keys())
            # preserve a stable order: put common fields first if present
            common_order = ["model", "id", "rouge_l_clin", "bleu_clin", "rouge_l_lay", "bleu_lay",
                            "entity_p", "entity_r", "entity_f", "hallucination_rate",
                            "clinical_utility_likert", "pred_entities"]
            # remaining keys (like entity_f_xxx)
            remaining = sorted(k for k in all_keys if k not in common_order)
            fieldnames = [k for k in common_order if k in all_keys] + remaining

            per_case_csv.parent.mkdir(parents=True, exist_ok=True)

            with open(per_case_csv, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results_rows)

    # Optionally produce summary statistics (means) while leaving the agg lists intact
    def _mean_or_none(lst):
        try:
            return statistics.mean(lst) if lst else None
        except Exception:
            return None

    agg_summary = {
        'model': model_id,
        'n': agg['n'],
        'mean_rouge_l_clin': _mean_or_none(agg['rouge_l_clin']),
        'mean_bleu_clin': _mean_or_none(agg['bleu_clin']),
        'mean_rouge_l_lay': _mean_or_none(agg['rouge_l_lay']),
        'mean_bleu_lay': _mean_or_none(agg['bleu_lay']),
        'mean_entity_p': _mean_or_none(agg['entity_p']),
        'mean_entity_r': _mean_or_none(agg['entity_r']),
        'mean_entity_f': _mean_or_none(agg['entity_f']),
        'mean_hall_rate': _mean_or_none(agg['hall_rate']),
        'mean_likert': _mean_or_none(agg['likert']),
    }

    # Return both the raw agg (with lists) and a compact summary for convenience
    return {"agg": agg, "summary": agg_summary}

    def _mean(lst):
        return float(sum(lst) / len(lst)) if lst else 0.0

        summary = {
            "model": model_id,
            "n": agg['n'],
            "rouge_l_clin_mean": _mean(agg['rouge_l_clin']),
            "bleu_clin_mean": _mean(agg['bleu_clin']),
            "rouge_l_lay_mean": _mean(agg['rouge_l_lay']),
            "bleu_lay_mean": _mean(agg['bleu_lay']),
            "entity_p_mean": _mean(agg['entity_p']),
            "entity_r_mean": _mean(agg['entity_r']),
            "entity_f_mean": _mean(agg['entity_f']),
            "hallucination_rate_mean": _mean(agg['hall_rate']),
            "clinical_utility_likert_mean": _mean(agg['likert'])
        }
        if save_dir:
            with open(save_dir / f"{model_id.replace(':','_')}_summary.json", "w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2)
        return summary

def entity_level_scores_by_type(gold_entities, pred_entities):
    """
    Returns F1 scores broken down by entity type (ANATOMY, OBSERVATION, LATERALITY).
    """
    types = ["ANATOMY", "OBSERVATION", "LATERALITY"]
    results = {t: {"p":0, "r":0, "f":0} for t in types}

    for t in types:
        g = [e for e in gold_entities if _radgraph_type_normalize(e.get("type","")) == t]
        p = [e for e in pred_entities if _radgraph_type_normalize(e.get("type","")) == t]
        prec, rec, f1 = entity_level_scores(g, p)
        results[t] = {"p": prec, "r": rec, "f": f1}
    return results


# plotting helpers
def plot_grouped_metrics(summaries: List[Dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(summaries)
    for c in ["rouge_l_clin_mean","bleu_clin_mean","entity_f_mean","hallucination_rate_mean","clinical_utility_likert_mean"]:
        if c not in df.columns:
            df[c] = 0.0
    df = df.fillna(0.0)
    df["hallucination_inv"] = 1.0 - df["hallucination_rate_mean"]
    metrics = [("ROUGE-L","rouge_l_clin_mean"),("BLEU","bleu_clin_mean"),("Entity F1","entity_f_mean"),("Hallucination (inv)","hallucination_inv"),("Utility","clinical_utility_likert_mean")]
    fig, ax = plt.subplots(figsize=(10,5))
    x = np.arange(len(df))
    width = 0.15
    for i, (_, col) in enumerate(metrics):
        ax.bar(x + i*width, df[col].values, width, label=metrics[i][0])
    ax.set_xticks(x + width * (len(metrics)-1) / 2)
    ax.set_xticklabels(df['model'].values, rotation=30, ha='right')
    ax.set_ylabel("Metric (normalized)")
    ax.set_title("Model comparison — grouped metrics")
    ax.legend()
    plt.tight_layout()
    fpath = out_dir / "grouped_metrics.png"
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    return fpath


def plot_metric_distributions(per_case_csv_paths: List[Path], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    dfs = []
    for p in per_case_csv_paths:
        try:
            df = pd.read_csv(p)
            df['model'] = p.stem.rsplit('_per_case',1)[0]
            dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return {}
    big = pd.concat(dfs, ignore_index=True)
    figs = {}
    for metric in ['entity_f','rouge_l_clin','bleu_clin','hallucination_rate']:
        plt.figure(figsize=(8,4))
        ax = plt.gca()
        big.boxplot(column=[metric], by='model', ax=ax)
        plt.title(f"{metric} per-model distribution")
        plt.suptitle("")
        plt.ylabel(metric)
        plt.xticks(rotation=30)
        fname = out_dir / f"{metric}_boxplot.png"
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()
        figs[metric] = fname
    return figs


def plot_radar_normalized(summaries: List[Dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(summaries).set_index('model')
    labels = ["ROUGE-L","BLEU","Entity F1","Hallucination(inv)","Utility(norm)"]
    model_names = df.index.tolist()
    values = []
    for m in model_names:
        row = df.loc[m]
        r = float(row.get("rouge_l_clin_mean",0))
        b = float(row.get("bleu_clin_mean",0))
        e = float(row.get("entity_f_mean",0))
        h = 1.0 - float(row.get("hallucination_rate_mean",1.0))
        u = float(row.get("clinical_utility_likert_mean",0))/5.0
        values.append([r,b,e,h,u])
    num_metrics = len(labels)
    angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, polar=True)
    for idx, val in enumerate(values):
        v = val + val[:1]
        ax.plot(angles, v, linewidth=1, label=model_names[idx])
        ax.fill(angles, v, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.legend(loc='upper right', bbox_to_anchor=(1.3,1.05))
    plt.title("Model comparison (radar)")
    fpath = out_dir / "radar_normalized.png"
    plt.tight_layout()
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    return fpath


# CLI entrypoint
def main(args):
    data_dir = Path(args.data_dir)
    items = load_gold_dataset(data_dir)
    print(f"Loaded {len(items)} eval items from {data_dir}")

    models = [m.strip() for m in args.models.split(",")]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    per_case_csvs = []
    for model_id in models:
        print("Evaluating model:", model_id)
        s = Summarizer(clinical_model=model_id, lay_model=model_id, device=args.device)
        model_out_dir = output / model_id.replace(":", "_")
        model_out_dir.mkdir(parents=True, exist_ok=True)
        summary = evaluate_one_model_on_dataset(model_id, items, s,
                                               max_samples=args.max_samples,
                                               save_dir=model_out_dir)
        all_summaries.append(summary)
        per_case = model_out_dir / f"{model_id.replace(':','_')}_per_case.csv"
        if per_case.exists():
            per_case_csvs.append(per_case)

    # aggregate CSV + JSON
    if all_summaries:
        aggregate_csv = output / "aggregate_summary.csv"
        pd.DataFrame(all_summaries).to_csv(aggregate_csv, index=False)
        with open(output / "aggregate_summary.json", "w", encoding="utf-8") as fh:
            json.dump(all_summaries, fh, indent=2)

    # plots
    plots_dir = output / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    if all_summaries:
        plot_grouped_metrics(all_summaries, plots_dir)
        plot_radar_normalized(all_summaries, plots_dir)
    if per_case_csvs:
        plot_metric_distributions(per_case_csvs, plots_dir)

    print("Evaluation finished. Results in:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with gold .json / .jsonl files.")
    parser.add_argument("--models", type=str, required=True, help="Comma-separated model IDs (HF or openai:..., ollama:...).")
    parser.add_argument("--output_dir", type=str, default="results/eval")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    main(args)

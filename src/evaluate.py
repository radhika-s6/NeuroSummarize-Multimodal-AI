from pathlib import Path
import json
import pandas as pd
from typing import List, Dict, Tuple
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

nltk.download('punkt', quiet=True)

def load_gold_predictions(gold_dir: str, pred_dir: str) -> List[Dict]:
    """
    Returns a list of dictionaries with gold and predicted summaries/entities
    """
    items = []
    gold_dir = Path(gold_dir)
    pred_dir = Path(pred_dir)
    for gold_file in gold_dir.glob("*.json"):
        base = gold_file.stem
        pred_file = pred_dir / f"{base}.json"
        if not pred_file.exists():
            continue
        with open(gold_file, "r", encoding="utf-8") as f:
            gold = json.load(f)
        with open(pred_file, "r", encoding="utf-8") as f:
            pred = json.load(f)
        items.append({"id": base, "gold": gold, "pred": pred})
    return items

# Metrics Computation
def normalize_text(text: str) -> str:
    if not text: return ""
    return " ".join(nltk.word_tokenize(text.lower()))

def rouge_l_score(ref: str, hyp: str) -> float:
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return scorer.score(ref, hyp)['rougeL'].fmeasure

def bleu_score(ref: str, hyp: str) -> float:
    ref_tokens = nltk.word_tokenize(ref.lower())
    hyp_tokens = nltk.word_tokenize(hyp.lower())
    if len(hyp_tokens) == 0: return 0.0
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=SmoothingFunction().method1)

def entity_f1(gold_entities: List[str], pred_entities: List[str]) -> float:
    gold_set = set(map(str.lower, gold_entities))
    pred_set = set(map(str.lower, pred_entities))
    tp = len(gold_set & pred_set)
    precision = tp / (len(pred_set) or 1e-9)
    recall = tp / (len(gold_set) or 1e-9)
    f1 = 2*precision*recall/(precision+recall) if precision+recall>0 else 0.0
    return f1

def hallucination_rate(gold_entities: List[str], pred_entities: List[str]) -> float:
    gold_set = set(map(str.lower, gold_entities))
    hallucinated = [e for e in pred_entities if e.lower() not in gold_set]
    return len(hallucinated) / (len(pred_entities) or 1e-9)

# Evaluation Wrapper
def evaluate_models(model_outputs: Dict[str, Dict], gold_summary: str,
                    gold_entities: List[str], human_scores_dict: Dict[str,List[int]]):
    """
    model_outputs: {"model_name": {"summary": str, "entities": List[str]}}
    Returns: metrics_dict = {"model_name": {"rouge_l":..., "bleu":..., "entity_f1":..., "hallucination":..., "human_score":...}}
    """
    metrics_dict = {}
    for model, outputs in model_outputs.items():
        summary = outputs.get("summary", "")
        entities = outputs.get("entities", [])
        rouge = rouge_l_score(gold_summary, summary)
        bleu = bleu_score(gold_summary, summary)
        f1 = entity_f1(gold_entities, entities)
        hall = hallucination_rate(gold_entities, entities)
        human_score = sum(human_scores_dict.get(model, [])) / (len(human_scores_dict.get(model, [])) or 1)
        metrics_dict[model] = {"rouge_l": rouge, "bleu": bleu, "entity_f1": f1,
                               "hallucination": hall, "human_score": human_score}
    return metrics_dict

def aggregate_metrics(metrics_dict: Dict[str, Dict]) -> pd.DataFrame:
    """
    metrics_dict: {patient_id: {model_name: {metric: value}}}
    Returns DataFrame with mean ± std per model for each metric.
    """
    rows = []
    models = list(next(iter(metrics_dict.values())).keys())
    for model in models:
        model_metrics = {metric: [] for metric in next(iter(metrics_dict.values()))[model]}
        for patient_id in metrics_dict:
            for metric, value in metrics_dict[patient_id][model].items():
                model_metrics[metric].append(value)
        row = {"model": model}
        for metric, vals in model_metrics.items():
            row[f"{metric}_mean"] = np.mean(vals)
            row[f"{metric}_std"] = np.std(vals)
        rows.append(row)
    return pd.DataFrame(rows)

# Visualization
def plot_model_comparison(metrics_dict: Dict[str, Dict], output_dir):
    """
    Generates bar charts with error bars for key metrics.
    """
    df = aggregate_metrics(metrics_dict)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Color-code models
    model_colors = {"BART": "red", "T5": "blue", "GPT-4": "green"}

    # Metrics to plot
    plot_metrics = [
        ("rouge_l_clin", "ROUGE-L (Clinical)"),
        ("bleu_clin", "BLEU (Clinical)"),
        ("rouge_l_lay", "ROUGE-L (Lay)"),
        ("bleu_lay", "BLEU (Lay)"),
        ("entity_f", "Entity F1"),
        ("hallucination_rate", "Hallucination Rate"),
        ("human_likert", "Clinical Utility (Likert)")
    ]

    for metric, title in plot_metrics:
        means = [df.loc[df.model==m, f"{metric}_mean"].values[0] for m in df.model]
        stds = [df.loc[df.model==m, f"{metric}_std"].values[0] for m in df.model]
        plt.figure(figsize=(8,5))
        sns.barplot(x=list(df.model), y=means, yerr=stds,
                    palette=[model_colors.get(m, "gray") for m in df.model])
        plt.ylabel(title)
        plt.title(f"{title} by Model")
        plt.ylim(0,1 if "rate" not in metric.lower() else 0.5)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}_comparison.png", dpi=300)
        plt.close()
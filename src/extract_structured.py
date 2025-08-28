import json, re
from typing import Dict, List

import spacy
from spacy.language import Language

# Try scispaCy first (if installed), else fall back to en_core_web_sm
def _load_nlp() -> Language:
    try:
        return spacy.load("en_core_sci_sm")
    except Exception:
        return spacy.load("en_core_web_sm")

nlp = _load_nlp()

# vocab lists remain as high-recall fallbacks
MODALITIES = ["MRI","fMRI","CT","PET","SPECT","DTI","MRA","MRV","SWI","FLAIR","T1","T2"]
BRAIN_REGIONS = [
    "frontal lobe","parietal lobe","temporal lobe","occipital lobe","insula","cingulate",
    "thalamus","hypothalamus","hippocampus","amygdala","basal ganglia","caudate","putamen",
    "globus pallidus","brainstem","midbrain","pons","medulla","cerebellum","corpus callosum",
    "ventricles","white matter","grey matter"
]
FINDING_KEYWORDS = [
    "lesion","infarct","hemorrhage","tumor","mass","edema","atrophy","ischemia","enhancement",
    "demyelination","microbleeds","calcification","midline shift","hydrocephalus"
]

def _find_any(text: str, vocab: List[str]) -> List[str]:
    t = text.lower()
    return sorted({v for v in vocab if v.lower() in t})

def _ner_candidates(text: str) -> Dict[str, List[str]]:
    """Pull candidates via NER labels + matcher-like regex."""
    doc = nlp(text)
    ents = [e.text for e in doc.ents]
    # crude boosts via regex
    laterality = []
    if re.search(r"\bleft\b", text, re.I): laterality.append("left")
    if re.search(r"\bright\b", text, re.I): laterality.append("right")
    if re.search(r"\bmidline\b", text, re.I): laterality.append("midline")
    return {
        "ents": ents,
        "laterality": sorted(set(laterality))
    }

def extract_structured(text: str) -> Dict:
    ner = _ner_candidates(text)
    # union rule-based lists with NER-driven hints
    modalities = sorted(set(_find_any(text, MODALITIES)))
    regions = sorted(set(_find_any(text, BRAIN_REGIONS) + [e for e in ner["ents"] if e.lower() in " ".join(BRAIN_REGIONS)]))
    findings = sorted(set(_find_any(text, FINDING_KEYWORDS)))
    return {
        "modality": modalities,
        "regions": regions,
        "findings": findings,
        "laterality": ner["laterality"]
    }

def save_json(obj: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

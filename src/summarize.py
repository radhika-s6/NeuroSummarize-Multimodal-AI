# src/summarize.py
from typing import Dict, Optional, Tuple, Callable, List
import os

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # Hugging Face
from src.utils.config_loader import load_config, get_api_keys

_cfg = load_config()
_api_keys = get_api_keys(_cfg)

OPENAI_API_KEY = _api_keys.get("openai")
GROQ_API_KEY   = _api_keys.get("groq")

try:
    from openai import OpenAI  # openai>=1.x
except Exception:
    OpenAI = None

try:
    import ollama  # local LLaMA via Ollama
except Exception:
    ollama = None

try:
    from groq import Groq  # local LLaMA via Ollama
except Exception:
    groq = None

# Keep your existing defaults
DEFAULT_FAST = "sshleifer/distilbart-cnn-12-6"
DEFAULT_QUALITY = "facebook/bart-large-cnn"

# Use your existing structured extractor to build predicted entities
from src.extract_structured import extract_structured


def _is_openai(model_id: str) -> bool:
    return isinstance(model_id, str) and model_id.startswith("openai:")


def _is_ollama(model_id: str) -> bool:
    return isinstance(model_id, str) and model_id.startswith("ollama:")


def _strip_prefix(model_id: str) -> str:
    """openai:gpt-3.5-turbo -> gpt-3.5-turbo; ollama:llama3 -> llama3"""
    return model_id.split(":", 1)[1] if ":" in model_id else model_id


def _hf_generator(model_id: str, device: Optional[str]) -> Tuple[Callable[[str, int], str], object]:
    """Returns (generate_fn, tok) for HF seq2seq models (existing behavior)."""
    tok = AutoTokenizer.from_pretrained(model_id)
    mod = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    if device:
        mod.to(device)

    def _gen(text: str, max_new_tokens: int = 220) -> str:
        inputs = tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024   # cap input length (BART/T5 default is usually 1024)
            )

        if device:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        out = mod.generate(
            **inputs,
            max_length=256,   # cap output summary length
            num_beams=4,
            early_stopping=True
            )

        return tok.decode(out[0], skip_special_tokens=True)

    return _gen, tok


def _openai_generator(model_id: str, system_prompt: str) -> Callable[[str, int], str]:
    """OpenAI chat completion generator for summaries."""
    if OpenAI is None:
        raise ImportError("openai package not installed. Add to requirements and pip install.")
    api_key = OPENAI_API_KEY
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY env var not set.")
    client = OpenAI(api_key=api_key)

    def _gen(text: str, max_new_tokens: int = 220) -> str:
        resp = client.chat.completions.create(
            model=_strip_prefix(model_id),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.2,
            max_tokens=max_new_tokens
        )
        return resp.choices[0].message.content.strip()

    return _gen


def _ollama_generator(model_id: str, system_prompt: str) -> Callable[[str, int], str]:
    """Local LLaMA-3 via Ollama (requires 'ollama run llama3' pulled)."""
    if ollama is None:
        raise ImportError("ollama package not installed. Add to requirements and pip install.")

    model = _strip_prefix(model_id)

    def _gen(text: str, max_new_tokens: int = 220) -> str:
        prompt = f"{system_prompt}\n\nUSER REPORT:\n{text}"
        out = ollama.generate(model=model, prompt=prompt, options={"num_predict": max_new_tokens})
        return out.get("response", "").strip()

    return _gen


def _is_groq(model_id: str) -> bool:
    return isinstance(model_id, str) and model_id.startswith("groq:")


def _groq_generator(model_id, system_prompt="You are a medical assistant"):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    clean_model = model_id.replace("groq:", "")

    def _gen(text: str, max_new_tokens: int = 220) -> str:
        resp = client.chat.completions.create(
            model=clean_model,   # correct name: "llama3-8b-8192"
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            max_tokens=max_new_tokens,
            temperature=0.2
        )
        return resp.choices[0].message.content.strip()
    return _gen



class Summarizer:
    """
    Backward-compatible:
      - If you pass HF IDs (your current default), we use your existing HF path.
      - If you pass 'openai:<model>' or 'ollama:<model>', we call that backend.
    """

    def __init__(
        self,
        clinical_model: str = DEFAULT_FAST,
        lay_model: str = DEFAULT_QUALITY,
        device: str = None
    ):
        self.device = device
        self._clin_backend = self._build_backend(clinical_model, which="clinical")
        self._lay_backend  = self._build_backend(lay_model, which="lay")

    @staticmethod
    def _system_prompt(which: str) -> str:
        if which == "clinical":
            return (
                "You are a neuroradiology assistant. Produce a concise, technically correct "
                "clinical summary suitable for radiologists. No speculation. Use precise terms."
            )
        else:
            return (
                "You are a patient educator. Produce a short, clear layperson summary without jargon. "
                "Explain findings simply and avoid scary language. No new facts beyond the text."
            )

    def _build_backend(self, model_id: str, which: str):
        # Auto-wrap: if someone passes "gpt-3.5-turbo" without prefix
        if model_id in ["gpt-3.5-turbo", "gpt-4", "gpt-4o"]:
            model_id = "openai:" + model_id

        if _is_openai(model_id):
            return ("openai", _openai_generator(model_id, self._system_prompt(which)), model_id)
        if _is_ollama(model_id):
            return ("ollama", _ollama_generator(model_id, self._system_prompt(which)), model_id)
        if _is_groq(model_id):
            return ("groq", _groq_generator(model_id, self._system_prompt(which)), model_id)

        # Default Hugging Face path
        gen, tok = _hf_generator(model_id, self.device)
        return ("hf", (gen, tok), model_id)

    def _gen(self, text: str, which: str, max_new_tokens: int = 220) -> str:
        backend, obj, model_id = self._clin_backend if which == "clinical" else self._lay_backend

        if backend in ["openai", "ollama", "groq"]:
            gen = obj
            return gen(text, max_new_tokens=max_new_tokens)

        # HF path (existing)
        gen, _tok = obj
        return gen(text, max_new_tokens=max_new_tokens)

    def summarize_both(self, text: str) -> Dict[str, object]:
        """
        Returns a dict:
          {
            "clinical_summary": str,
            "lay_summary": str,
            "pred_entities": [ {'type':'region'|'finding'|'laterality', 'text': '...'}, ... ]
          }
        """
        clinical = self._gen(text, "clinical")
        lay = self._gen(text, "lay")

        # Produce structured predicted entities using your extractor on the clinical summary
        try:
            struct = extract_structured(clinical)
            pred_entities: List[dict] = []
            for r in struct.get("regions", []):
                pred_entities.append({"type": "region", "text": r})
            for f in struct.get("findings", []):
                pred_entities.append({"type": "finding", "text": f})
            for l in struct.get("laterality", []):
                pred_entities.append({"type": "laterality", "text": l})
        except Exception:
            # graceful fallback: empty list
            pred_entities = []

        return {"clinical_summary": clinical, "lay_summary": lay, "pred_entities": pred_entities}

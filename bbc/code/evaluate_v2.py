from __future__ import annotations

"""
Indonesian/English → Batak evaluation pipeline.
Based on FLORES-200 evaluation pipeline.
Data source is from NusaX

Usage:
    # Checkpoint selection (dev split):
    uv run python evaluate.py --split dev --models nllb qwen_batak

    # Final reporting (devtest split — run once, after model selection):
    uv run python evaluate.py --split devtest --models nllb nllb_batak qwen

    # Single model, cached hypotheses only:
    uv run python evaluate.py --split devtest --models nllb --no-comet
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Callable

import torch
from sacrebleu.metrics import BLEU, CHRF

PROJECT_DIR = os.environ.get("PROJECT_DIR", "/dss/dsshome1/0C/go93pec2/evaluation")

DEV_DIR = Path(PROJECT_DIR) / "data" / "dev"
DEVTEST_DIR = Path(PROJECT_DIR) / "data" / "devtest"
FLORES_DIR = Path(PROJECT_DIR) / "data"
RESULTS_DIR = Path(PROJECT_DIR) / "outputs" / "results"

DEV_DIR.mkdir(parents=True, exist_ok=True)
DEVTEST_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PYTORCH_MPS_HIGH_WATERMARK_RATIO = 0.0
# COMET is optional; gracefully degrade if unavailable.
try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False


# ── device ────────────────────────────────────────────────────────────────────


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def safe_dtype(device: str) -> torch.dtype:
    """float16 hangs on MPS for some ops; use float32 there."""
    return torch.float16 if device == "cuda" else torch.float32


def free_cache(device: str) -> None:
    """Release device memory after each batch to prevent MPS stalls."""
    if device == "mps":
        torch.mps.empty_cache()
    # For CUDA, we avoid emptying cache every batch to prevent severe performance drops.
    # PyTorch handles CUDA memory pooling efficiently during the loop.


# ── FLORES-200 loader ─────────────────────────────────────────────────────────


def load_flores(
    split: str, 
    nusax=False,
    src_lang: str = "eng_Latn",
    tgt_lang: str = "bbc_Latn",
) -> tuple[list[str], list[str]]:
    """Return (source_sentences, target_references) for the given split.

    Reads from local files downloaded from https://tinyurl.com/flores200dataset.
    """
    split_dir = FLORES_DIR / split
    ext = f".{split}"

    def read_lang(lang: str) -> list[str]:
        path = split_dir / f"{lang}{ext}"
        return path.read_text(encoding="utf-8").splitlines()

    lang_map = {
        "eng_Latn": "en",
        "ind_Latn": "id",
        "bbc_Latn": "bbc"
    }

    if nusax:
        sources = read_lang(lang_map[src_lang])
        references = read_lang(lang_map[tgt_lang])
    else:
        sources = read_lang(src_lang)
        references = read_lang(tgt_lang)
        
    assert len(sources) == len(references), "Source/reference length mismatch"
    return sources, references


# ── metrics ───────────────────────────────────────────────────────────────────


def compute_metrics(
    hypotheses: list[str],
    references: list[str],
    sources: list[str],
    use_comet: bool = True,
) -> dict[str, float]:
    """Compute chrF++, BLEU, and COMET."""
    chrf = CHRF(word_order=2)  # word_order=2 → chrF++
    bleu = BLEU(tokenize="13a")  # standard BLEU for whitespace languages

    scores: dict[str, float] = {
        "chrF++": round(chrf.corpus_score(hypotheses, [references]).score, 2),
        "BLEU": round(bleu.corpus_score(hypotheses, [references]).score, 2),
    }

    if use_comet and COMET_AVAILABLE:
        model_path = download_model("Unbabel/wmt22-comet-da")
        comet_model = load_from_checkpoint(model_path)
        data = [
            {"src": s, "mt": h, "ref": r}
            for s, h, r in zip(sources, hypotheses, references)
        ]
        gpus = 1 if torch.cuda.is_available() else 0
        result = comet_model.predict(data, batch_size=16, gpus=gpus)
        scores["COMET"] = round(result.system_score, 4)
    elif use_comet and not COMET_AVAILABLE:
        print("  [COMET skipped — unbabel-comet not installed]")

    return scores


# ── translators ───────────────────────────────────────────────────────────────


def translate_nllb(
    sentences: list[str],
    model_name: str,
    adapter_path: str | None = None, # Present for uniform signature
    src_lang: str = "eng_Latn",
    tgt_lang: str = "bbc_Latn",
    batch_size: int | None = None,
) -> list[str]:
    """Translate with NLLB-200 (Base or Fine-tuned)."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    print(f"  Loading NLLB model from {model_name} … (src: {src_lang}, tgt: {tgt_lang})")
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang)
    tokenizer.src_lang = src_lang
    
    device = get_device()
    effective_batch = batch_size if batch_size is not None else (8 if device == "mps" else 16)
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=safe_dtype(device))
    model = model.to(device).eval()

    forced_bos_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    if forced_bos_id is None or forced_bos_id == tokenizer.unk_token_id:
        print(f"  [WARNING] Target language token {tgt_lang} is not recognized by the tokenizer.")

    translations: list[str] = []

    for i in range(0, len(sentences), effective_batch):
        batch = sentences[i : i + effective_batch]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        with torch.no_grad():
            ids = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_id,
                max_length=256,
                num_beams=4,
            )
        translations.extend(tokenizer.batch_decode(ids, skip_special_tokens=True))
        free_cache(device)
        print(
            f"  {min(i + effective_batch, len(sentences))}/{len(sentences)}",
            end="\r",
            flush=True,
        )

    print()
    return translations


def translate_qwen(
    sentences: list[str],
    model_name: str,
    adapter_path: str | None = None,
    src_lang: str = "eng_Latn",
    tgt_lang: str = "bbc_Latn",
    batch_size: int | None = None,
) -> list[str]:
    """Translate with a Qwen instruction-tuned model via few-shot prompt."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading Qwen model from {model_name} …")
    if adapter_path:
        print(f"  Applying LoRA adapter from {adapter_path} …")

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    device = get_device()
    effective_batch = batch_size if batch_size is not None else (2 if device == "mps" else 4)
    
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=safe_dtype(device))
    
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = base_model
        
    model = model.to(device).eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def lang_name(code: str) -> str:
        names = {
            "eng_Latn": "English",
            "ind_Latn": "Indonesian",
            "bbc_Latn": "Batak",
        }
        return names.get(code, code)

    def make_prompt_chat(src: str) -> str:
        src_name = lang_name(src_lang)
        tgt_name = lang_name(tgt_lang)

        prompt = (
            f"Translate the following {src_name} text into {tgt_name}.\n"
            f"{src_name}: {src}\n"
            f"{tgt_name}: "
        )
        messages = [
            {"role": "system", "content": "You are Qwen, an expert translator."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return text

    translations: list[str] = []
    for i in range(0, len(sentences), effective_batch):
        batch = sentences[i : i + effective_batch]
        prompts = [make_prompt_chat(s) for s in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        # Decode only the newly generated tokens (strip the prompt)
        new_ids = out_ids[:, prompt_len:]
        decoded = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
        translations.extend(t.strip() for t in decoded)
        free_cache(device)
        print(
            f"  {min(i + effective_batch, len(sentences))}/{len(sentences)}",
            end="\r",
            flush=True,
        )

    print()
    return translations


# ── registry ──────────────────────────────────────────────────────────────────


MODEL_REGISTRY: dict[str, tuple[str, str, Callable]] = {
    "nllb": ("facebook/nllb-200-distilled-600M", "", translate_nllb),
    "nllb_1.3b": ("facebook/nllb-200-distilled-1.3B", "", translate_nllb),
    "nllb_batak": ("/dss/dsshome1/0C/go93pec2/batak-nllb/outputs/nllb-600m-batak-10epoch_v2", "", translate_nllb),
    "nllb_batak_curriculum": ("/dss/dsshome1/0C/go93pec2/batak-nllb/outputs/nllb-600m-batak-curriculum/final_curriculum_model", "", translate_nllb),
    "nllb_batak_ind_bbc_curriculum": ("/dss/dsshome1/0C/go93pec2/batak-nllb/outputs/nllb-600m-batak-ind-curriculum/final_curriculum_model", "", translate_nllb),
    "nllb_batak_synt": ("/dss/dsshome1/0C/go93pec2/batak-nllb/outputs/nllb-600m-batak-5epoch-real-synt", "", translate_nllb),

    "qwen": ("Qwen/Qwen2.5-1.5B-Instruct", "", translate_qwen),
    "qwen_batak": ("Qwen/Qwen2.5-1.5B-Instruct", "/dss/dsshome1/0C/go93pec2/batak-qwen/outputs/qwen2.5-1.5b-batak-qlora", translate_qwen),
    "qwen_batak2": ("Qwen/Qwen2.5-1.5B-Instruct", "/dss/dsshome1/0C/go93pec2/batak-qwen-sft/outputs/qwen2.5-1.5b-batak-qlora-sft", translate_qwen),
    "qwen_batak_curriculum": ("Qwen/Qwen2.5-1.5B-Instruct", "/dss/dsshome1/0C/go93pec2/batak-qwen-sft/outputs/qwen2.5-1.5b-batak-qlora-sft-curriculum/final_curriculum_adapter", translate_qwen),
    "qwen_batak_ind_bbc_curriculum": ("Qwen/Qwen2.5-1.5B-Instruct", "/dss/dsshome1/0C/go93pec2/batak-qwen-sft/outputs/qwen2.5-1.5b-batak-qlora-sft-ind-bbc-curriculum/final_curriculum_adapter", translate_qwen),
}


# ── main ──────────────────────────────────────────────────────────────────────


def run_evaluation(
    split: str,
    model_keys: list[str],
    use_comet: bool = True,
    src_lang: str = None,
    tgt_lang: str = None,
) -> dict:
    """Run the full evaluation loop and save results."""
    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"\nLoading FLORES-200 [{split}] …")
    sources, references = load_flores(split, nusax=True, src_lang=src_lang, tgt_lang=tgt_lang)
    print(f"  {len(sources)} sentences loaded.")

    all_results: dict = {
        "split": split,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "n_sentences": len(sources),
        "models": {},
    }

    for key in model_keys:
        model_name, adapter_path, translate_fn = MODEL_REGISTRY[key]
        hyp_path = RESULTS_DIR / f"{key}_{src_lang}_to_{tgt_lang}_{split}_hypotheses.txt"

        print(f"\n{'=' * 60}")
        print(f"Model: {key} ({model_name})")
        print(f"{'=' * 60}")

        # Cache hypotheses so you can rerun metric computation without retranslating.
        if hyp_path.exists():
            print(f"  [cache hit] Loading hypotheses from {hyp_path}")
            hypotheses = hyp_path.read_text(encoding="utf-8").splitlines()
        else:
            t0 = time.time()
            hypotheses = translate_fn(
                sentences=sources, 
                model_name=model_name, 
                adapter_path=adapter_path if adapter_path else None, 
                src_lang=src_lang, 
                tgt_lang=tgt_lang
            )
            elapsed = time.time() - t0
            
            # Free model weights before the next model loads
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            hyp_path.write_text("\n".join(hypotheses), encoding="utf-8")
            print(
                f"  Translated {len(hypotheses)} sentences in {elapsed:.1f}s "
                f"({elapsed / len(hypotheses):.2f}s/sent)"
            )

        print("  Computing metrics …")
        scores = compute_metrics(hypotheses, references, sources, use_comet=use_comet)
        print(f"  chrF++  : {scores['chrF++']}")
        print(f"  BLEU    : {scores['BLEU']}")
        if "COMET" in scores:
            print(f"  COMET   : {scores['COMET']}")

        all_results["models"][key] = {
            "model_name": model_name,
            "scores": scores,
            "sample_hypotheses": hypotheses[:10],
        }

    out_path = RESULTS_DIR / f"results_{src_lang}_to_{tgt_lang}_{split}.json"
    out_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nResults saved to {out_path}")

    # Summary table
    print(f"\n{'Model':<30} {'chrF++':>8} {'BLEU':>8} {'COMET':>8}")
    print("-" * 60)
    for key, data in all_results["models"].items():
        sc = data["scores"]
        comet = f"{sc['COMET']:.4f}" if "COMET" in sc else "   N/A"
        print(f"{key:<30} {sc['chrF++']:>8.2f} {sc['BLEU']:>8.2f} {comet:>8}")

    return all_results


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="FLORES-200 Evaluation")
    parser.add_argument(
        "--split",
        choices=["dev", "devtest"],
        default="dev",
        help="Use 'dev' for tuning/checkpoint selection, 'devtest' for final numbers.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_REGISTRY),
        default=["nllb"],
        help="Which models to evaluate.",
    )
    parser.add_argument(
        "--no-comet",
        action="store_true",
        help="Skip COMET (faster; use when iterating on checkpoint selection).",
    )
    args = parser.parse_args()

    print(f"Device: {get_device()}")
    
    print("\n=== Running Forward Translation (ID → BBC) ===")
    run_evaluation(
        split=args.split,
        model_keys=args.models,
        use_comet=not args.no_comet,
        src_lang="ind_Latn",
        tgt_lang="bbc_Latn",
    )
    
    print("\n=== Running Back Translation (BBC → ID) ===")
    run_evaluation(
        split=args.split,
        model_keys=args.models,
        use_comet=not args.no_comet,
        src_lang="bbc_Latn",
        tgt_lang="ind_Latn",
    )


if __name__ == "__main__":
    main()
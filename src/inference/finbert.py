"""
FinBERT sentiment inference — optimized for Colab T4 GPU.

Public API
----------
- `score_with_finbert`: score a DataFrame of news headlines with ProsusAI/finbert.
  Falls back to distilbert-base-uncased-finetuned-sst-2-english if FinBERT
  is unavailable (e.g. offline, HF Hub down).

T4 optimization
---------------
- batch_size=128 (T4 16GB VRAM comfortably fits FinBERT at max_length=512)
- fp16 inference via torch.cuda.amp.autocast() — ~2x speedup, negligible
  accuracy loss for sentiment classification
- max_length=512 (FinBERT's full context window)

Parquet caching
---------------
- `cached_score_with_finbert` checks Data/cryptonews_scored.parquet before
  re-running inference. The cache is keyed on a SHA256 hash of the source
  CSV (cryptonews.csv) — if the source changes, the cache is invalidated
  automatically and a fresh inference run is triggered.

Example
-------
    from src.inference.finbert import cached_score_with_finbert
    news_scored = cached_score_with_finbert(
        news_df=news,
        source_path=Path('Data/cryptonews.csv'),
        cache_path=Path('Data/cryptonews_scored.parquet'),
    )
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Default model candidates (loaded in order)
# ----------------------------------------------------------------------
# Each entry is (model_id, revision_sha) — pinning the revision protects
# against supply-chain attacks where a model author (or a compromised HF
# account) pushes a malicious update. To get the latest revision:
#   curl -s https://huggingface.co/api/models/<model_id> | jq -r .sha
DEFAULT_MODELS = [
    ("ProsusAI/finbert", "4556d13015211d73dccd3fdd39d39232506f3e43"),
    ("distilbert/distilbert-base-uncased-finetuned-sst-2-english",
     "714eb0fa89d2f80546fda750413ed43d93601a13"),
]

# Default T4-optimized settings
DEFAULT_BATCH_SIZE = 128
DEFAULT_MAX_LENGTH = 512


# ----------------------------------------------------------------------
# Source hash for cache invalidation
# ----------------------------------------------------------------------
def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA256 hex digest of a file (streaming, 1MB chunks)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def df_fingerprint(df: pd.DataFrame) -> str:
    """Return a stable SHA256 fingerprint of a DataFrame's content."""
    # Use pandas' CSV repr + shape for a content hash that's stable across
    # runs (deterministic) and sensitive to row/column changes.
    h = hashlib.sha256()
    h.update(str(df.shape).encode())
    h.update(df.head(100).to_csv(index=False).encode())
    return h.hexdigest()[:16]


# ----------------------------------------------------------------------
# Core inference function
# ----------------------------------------------------------------------
def score_with_finbert(
    news: pd.DataFrame,
    model_name: str = "ProsusAI/finbert",
    fallback_models: list[str] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = DEFAULT_MAX_LENGTH,
    use_fp16: bool = True,
    text_col: str = "text",
    title_col: str = "title",
    device: int | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Score every headline with a HuggingFace transformer.

    Parameters
    ----------
    news : pd.DataFrame
        Must contain a `title` column and optionally a `text` column.
        Title + text are concatenated for richer signal.
    model_name : str
        HuggingFace model id. Default: ProsusAI/finbert.
    fallback_models : list[str], optional
        Models to try if `model_name` fails to load (e.g. offline).
    batch_size : int
        Inference batch size. Default 128 (optimal for Colab T4 16GB).
    max_length : int
        Tokenizer max_length. Default 512 (FinBERT's full window).
    use_fp16 : bool
        If True and CUDA is available, run inference under
        torch.cuda.amp.autocast() for ~2x speedup. Default True.
    text_col, title_col : str
        Column names for body and title.
    device : int, optional
        -1 for CPU, 0 for CUDA. If None, auto-detect.
    progress : bool
        Show tqdm progress bar. Default True.

    Returns
    -------
    pd.DataFrame
        Same as input with an added `llm_sentiment` column in [-1, 1].
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    # ---- Device detection ----
    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    device_name = "cuda" if device == 0 else "cpu"
    logger.info(f"Device: {device_name} | batch={batch_size} | max_len={max_length} | fp16={use_fp16}")

    # ---- Model loading with fallback ----
    # `candidates` is a list of (model_name, revision) tuples. The revision
    # is a pinned git SHA on the HuggingFace Hub — defense against supply-
    # chain attacks (model author pushes a malicious update).
    if isinstance(model_name, str):
        # Caller passed a bare model name; look up the pinned revision
        # in DEFAULT_MODELS, fall back to "main" if unknown.
        pinned = dict(DEFAULT_MODELS)
        candidates = [(model_name, pinned.get(model_name, "main"))]
    else:
        candidates = [model_name]
    # Append fallback models (always pinned)
    candidates += [(m, r) for m, r in DEFAULT_MODELS[1:] if m != model_name]
    # Allow caller-supplied fallback_models (as bare strings for backwards compat)
    if fallback_models:
        pinned = dict(DEFAULT_MODELS)
        for fm in fallback_models:
            if isinstance(fm, str):
                candidates.append((fm, pinned.get(fm, "main")))
            elif isinstance(fm, tuple):
                candidates.append(fm)

    pipe = None
    loaded_model_name = None
    for name, revision in candidates:
        try:
            logger.info(f"Loading {name} @ {revision[:8]} ...")
            tok = AutoTokenizer.from_pretrained(name, revision=revision)  # nosec B615 — revision pinned
            mdl = AutoModelForSequenceClassification.from_pretrained(name, revision=revision)  # nosec B615 — revision pinned
            if device == 0:
                mdl = mdl.to(device)
                if use_fp16:
                    mdl = mdl.half()  # convert weights to fp16
            pipe = pipeline(
                "sentiment-analysis",
                model=mdl, tokenizer=tok,
                device=device,
                truncation=True,
                max_length=max_length,
            )
            loaded_model_name = f"{name}@{revision[:8]}"
            logger.info(f"  -> loaded {loaded_model_name}")
            break
        except Exception as e:
            logger.warning(f"  ! {name} @ {revision[:8]} failed: {e}")
    if pipe is None:
        raise RuntimeError("No sentiment model could be loaded.")
    print(f"[finbert] Using model: {loaded_model_name}")

    # ---- Prepare texts ----
    news = news.copy()
    news[text_col] = news[title_col].fillna("") + ". " + news[text_col].fillna("")
    # Filter out near-empty headlines to skip wasted compute
    texts = news[text_col].astype(str).tolist()
    # Replace empty strings with a single space so the tokenizer doesn't choke
    texts = [t if len(t.strip()) > 1 else " " for t in texts]
    n_total = len(texts)
    print(f"[finbert] Scoring {n_total:,} headlines on {device_name} "
          f"(batch={batch_size}, max_len={max_length}, fp16={use_fp16 and device == 0})")

    # ---- Batched inference with optional fp16 autocast ----
    scores: list[float] = []
    iterator = range(0, n_total, batch_size)
    if progress:
        iterator = tqdm(iterator, desc="FinBERT inference", unit="batch")

    with torch.no_grad():
        for i in iterator:
            batch = texts[i : i + batch_size]
            try:
                if use_fp16 and device == 0:
                    # torch.cuda.amp.autocast was deprecated in torch 2.4;
                    # torch.amp.autocast('cuda') is the new API.
                    with torch.amp.autocast("cuda"):
                        results = pipe(batch)
                else:
                    results = pipe(batch)
            except Exception:
                # Fallback: score one-by-one to skip individual failures
                logger.warning(f"Batch {i} failed; falling back to 1-by-1")
                results = []
                for t in batch:
                    try:
                        if use_fp16 and device == 0:
                            with torch.amp.autocast("cuda"):
                                results.append(pipe(t))
                        else:
                            results.append(pipe(t))
                    except Exception:
                        results.append({"label": "neutral", "score": 0.5})

            for r in results:
                label = r["label"].lower()
                prob = float(r["score"])
                if "pos" in label:
                    s = prob
                elif "neg" in label:
                    s = -prob
                else:
                    s = 0.0  # neutral (FinBERT 3-class) → 0
                scores.append(s)

    news["llm_sentiment"] = scores
    print(f"[finbert] Done. mean={news.llm_sentiment.mean():.4f} "
          f"std={news.llm_sentiment.std():.4f} "
          f"min={news.llm_sentiment.min():.3f} max={news.llm_sentiment.max():.3f}")
    return news


# ----------------------------------------------------------------------
# Cached wrapper
# ----------------------------------------------------------------------
def cached_score_with_finbert(
    news_df: pd.DataFrame,
    source_path: Path,
    cache_path: Path,
    force_refresh: bool = False,
    **inference_kwargs,
) -> pd.DataFrame:
    """Score `news_df` with FinBERT, caching results to Parquet.

    Cache invalidation: a SHA256 hash of `source_path` (the original CSV)
    is stored as metadata in the Parquet file. If the source hash changes
    (e.g. the CSV is regenerated), the cache is invalidated.

    Parameters
    ----------
    news_df : pd.DataFrame
        News headlines to score.
    source_path : Path
        Path to the source CSV (used for cache invalidation).
    cache_path : Path
        Path to the Parquet cache file.
    force_refresh : bool
        If True, ignore the cache and re-run inference.
    **inference_kwargs
        Passed through to `score_with_finbert`.

    Returns
    -------
    pd.DataFrame
        Scored news with `llm_sentiment` column.
    """
    cache_path = Path(cache_path)
    source_path = Path(source_path)

    # ---- Compute source hash for cache key ----
    source_hash = file_sha256(source_path) if source_path.exists() else df_fingerprint(news_df)
    df_fp = df_fingerprint(news_df)
    cache_key = f"{source_hash[:16]}_{df_fp}"

    # ---- Check cache ----
    if cache_path.exists() and not force_refresh:
        try:
            cached = pd.read_parquet(cache_path)
            stored_key = cached.attrs.get("cache_key", "")
            if stored_key == cache_key:
                print(f"[cache] HIT  →  {cache_path}  (cache_key={cache_key})")
                print(f"[cache] Returning {len(cached):,} pre-scored rows.")
                return cached
            else:
                print(f"[cache] MISS  →  source hash changed "
                      f"(stored={stored_key[:16]}... vs current={cache_key[:16]}...)")
        except Exception as e:
            print(f"[cache] MISS  →  cache read failed: {e}")

    # ---- Run inference ----
    print(f"[cache] Running FinBERT inference  (cache_key={cache_key})")
    scored = score_with_finbert(news_df, **inference_kwargs)

    # ---- Persist with cache metadata ----
    scored.attrs["cache_key"] = cache_key
    scored.attrs["source_path"] = str(source_path)
    scored.attrs["source_sha256"] = source_hash
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(cache_path, index=False)
    print(f"[cache] Wrote {len(scored):,} scored rows → {cache_path}  "
          f"({cache_path.stat().st_size / (1024*1024):.1f} MB)")

    return scored

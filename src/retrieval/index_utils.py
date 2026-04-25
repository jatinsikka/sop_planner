"""FAISS-backed cosine index over sentence embeddings."""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from rich.console import Console

console = Console()

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def _normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-10
    return v / norms


@dataclass
class IndexPaths:
    root: Path

    @property
    def faiss_index(self) -> Path:
        return self.root / "index.faiss"

    @property
    def embeddings(self) -> Path:
        return self.root / "embeddings.npy"

    @property
    def ids(self) -> Path:
        return self.root / "ids.json"

    @property
    def vectorizer(self) -> Path:
        return self.root / "vectorizer.pkl"

    @property
    def meta(self) -> Path:
        return self.root / "meta.json"


def _load_encoder(model_path: str | Path):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(str(model_path))


def _embed_st(encoder, texts: Sequence[str]) -> np.ndarray:
    vecs = encoder.encode(
        list(texts),
        batch_size=16,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 16,
    )
    return vecs.astype(np.float32)


def _tfidf_fit(texts: Sequence[str]):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(max_features=4096, ngram_range=(1, 2))
    X = vec.fit_transform(texts).toarray().astype(np.float32)
    return _normalize(X), vec


def _tfidf_transform(vectorizer, texts: Sequence[str]) -> np.ndarray:
    X = vectorizer.transform(list(texts)).toarray().astype(np.float32)
    return _normalize(X)


def build_and_save_index(
    texts: Sequence[str],
    ids: Sequence[str],
    out_dir: str | Path,
    encoder_path: Optional[str | Path] = None,
) -> None:
    """Embed `texts`, save normalized vectors and FAISS IP index to `out_dir`.

    Tries the trained sentence-transformers encoder first, falls back to the
    default BGE model, and finally to TF-IDF if nothing else works.
    """
    out = IndexPaths(Path(out_dir))
    out.root.mkdir(parents=True, exist_ok=True)

    embed_type = "sentence-transformers"
    vectorizer = None
    embeddings: np.ndarray

    candidate_models: List[str] = []
    if encoder_path is not None:
        candidate_models.append(str(encoder_path))
    candidate_models.append(DEFAULT_MODEL)

    encoder = None
    used_model = None
    for cand in candidate_models:
        try:
            console.log(f"[dim]Loading encoder: {cand}[/dim]")
            encoder = _load_encoder(cand)
            used_model = cand
            break
        except Exception as exc:
            console.log(f"[yellow]Encoder load failed for {cand}: {exc}[/yellow]")

    if encoder is not None:
        console.log(f"[bold]Embedding {len(texts)} SOPs with {used_model}[/bold]")
        embeddings = _embed_st(encoder, texts)
    else:
        console.log("[yellow]Falling back to TF-IDF embeddings[/yellow]")
        embeddings, vectorizer = _tfidf_fit(texts)
        embed_type = "tfidf"
        with out.vectorizer.open("wb") as f:
            pickle.dump(vectorizer, f)

    np.save(out.embeddings, embeddings)
    out.ids.write_text(json.dumps(list(ids), indent=2), encoding="utf-8")
    out.meta.write_text(
        json.dumps({"embed_type": embed_type, "model": used_model, "dim": int(embeddings.shape[1])}, indent=2),
        encoding="utf-8",
    )

    try:
        import faiss

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, str(out.faiss_index))
        console.log(f"[green]✓ FAISS index saved to {out.faiss_index}[/green]")
    except Exception as exc:
        console.log(f"[yellow]FAISS unavailable, will use numpy search: {exc}[/yellow]")


def load_index(out_dir: str | Path):
    out = IndexPaths(Path(out_dir))
    embeddings = np.load(out.embeddings)
    ids = json.loads(out.ids.read_text(encoding="utf-8"))
    meta = json.loads(out.meta.read_text(encoding="utf-8")) if out.meta.exists() else {"embed_type": "sentence-transformers"}

    faiss_index = None
    try:
        import faiss

        if out.faiss_index.exists():
            faiss_index = faiss.read_index(str(out.faiss_index))
    except Exception:
        faiss_index = None

    vectorizer = None
    if meta.get("embed_type") == "tfidf" and out.vectorizer.exists():
        with out.vectorizer.open("rb") as f:
            vectorizer = pickle.load(f)

    return embeddings, ids, faiss_index, meta, vectorizer


def search(
    query_texts: Sequence[str],
    corpus_embeddings: np.ndarray,
    corpus_ids: Sequence[str],
    faiss_index,
    top_k: int = 5,
    meta: Optional[dict] = None,
    vectorizer=None,
    encoder_path: Optional[str | Path] = None,
) -> List[List[Tuple[str, float]]]:
    """Encode queries and return top-k (id, score) per query."""
    embed_type = (meta or {}).get("embed_type", "sentence-transformers")

    if embed_type == "tfidf":
        if vectorizer is None:
            raise RuntimeError("TF-IDF index loaded without saved vectorizer")
        query_vecs = _tfidf_transform(vectorizer, query_texts)
    else:
        candidate_models: List[str] = []
        if encoder_path is not None:
            candidate_models.append(str(encoder_path))
        if meta and meta.get("model"):
            candidate_models.append(str(meta["model"]))
        candidate_models.append(DEFAULT_MODEL)

        encoder = None
        for cand in candidate_models:
            try:
                encoder = _load_encoder(cand)
                break
            except Exception:
                continue
        if encoder is None:
            raise RuntimeError("No sentence-transformers encoder available for query encoding")
        query_vecs = _embed_st(encoder, query_texts)

    if faiss_index is not None:
        scores, idx = faiss_index.search(query_vecs, top_k)
    else:
        sims = query_vecs @ corpus_embeddings.T
        idx = np.argsort(-sims, axis=1)[:, :top_k]
        scores = np.take_along_axis(sims, idx, axis=1)

    scores = np.clip(scores, 0.0, 1.0)
    results: List[List[Tuple[str, float]]] = []
    for r, s in zip(idx, scores):
        results.append([(corpus_ids[int(i)], float(sc)) for i, sc in zip(r, s)])
    return results

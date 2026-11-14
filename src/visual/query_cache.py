"""Formal query vector caching and Zero-GPU offline retrieval module.

Provides long-term, auditable caching of text query embeddings so that
downstream evaluation, ranking, and shopping experiments can execute with
zero GPU memory overhead (pure CPU NumPy vector dot product).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

QUERY_CACHE_SCHEMA = 1


def canonical_text(text: str) -> str:
    """Normalize query text for deterministic hashing and lookup."""
    return " ".join(str(text or "").split()).strip()


def text_sha256(text: str) -> str:
    """Compute SHA256 of normalized text."""
    return hashlib.sha256(canonical_text(text).encode("utf-8")).hexdigest()


def canonical_digest(value: Any) -> str:
    """Deterministic JSON digest for auditable metadata."""
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def unit(vector: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return L2-normalized float32 1D numpy array."""
    arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim != 1 or not np.isfinite(arr).all():
        raise ValueError("Invalid query embedding vector")
    norm = float(np.linalg.norm(arr))
    if norm <= 0:
        raise ValueError("Cannot normalize zero-norm query vector")
    return arr / norm


class QueryVectorCache:
    """Persistent, auditable mapping from query instruction to normalized vector."""

    def __init__(
        self,
        encoder_snapshot: dict[str, Any] | None = None,
        schema_version: int = QUERY_CACHE_SCHEMA,
    ):
        self.schema_version = schema_version
        self.encoder_snapshot = encoder_snapshot or {}
        self._entries: dict[str, dict[str, Any]] = {}
        self._vectors_by_text: dict[str, np.ndarray] = {}
        self._vectors_by_hash: dict[str, np.ndarray] = {}

    def register(
        self,
        instruction: str,
        vector: Sequence[float] | np.ndarray,
        *,
        task_id: str | None = None,
        audit: dict[str, Any] | None = None,
    ) -> str:
        """Register a precomputed query vector into the cache."""
        norm_text = canonical_text(instruction)
        if not norm_text:
            raise ValueError("Query instruction cannot be empty")
        uvec = unit(vector)
        h = text_sha256(norm_text)
        vec_list = uvec.tolist()
        vec_sha = canonical_digest(vec_list)

        entry = {
            "instruction": norm_text,
            "instruction_sha256": h,
            "task_id": task_id,
            "vector": vec_list,
            "vector_sha256": vec_sha,
            "encoding_audit": audit,
        }
        self._entries[norm_text] = entry
        self._vectors_by_text[norm_text] = uvec
        self._vectors_by_hash[h] = uvec
        return h

    def get(self, instruction: str) -> np.ndarray:
        """Retrieve unit vector for query text; raises KeyError if missing."""
        norm_text = canonical_text(instruction)
        if norm_text in self._vectors_by_text:
            return self._vectors_by_text[norm_text]
        h = text_sha256(norm_text)
        if h in self._vectors_by_hash:
            return self._vectors_by_hash[h]
        raise KeyError(f"Query text not found in cache: {norm_text!r}")

    def get_audit(self, instruction: str) -> dict[str, Any] | None:
        """Retrieve encoding audit if available."""
        norm_text = canonical_text(instruction)
        entry = self._entries.get(norm_text)
        if entry is not None:
            return entry.get("encoding_audit")
        return None

    def __contains__(self, instruction: str) -> bool:
        norm_text = canonical_text(instruction)
        return norm_text in self._vectors_by_text or text_sha256(norm_text) in self._vectors_by_hash

    def __len__(self) -> int:
        return len(self._entries)

    def to_dict(self) -> dict[str, Any]:
        """Serialize cache to auditable document."""
        return {
            "schema_version": self.schema_version,
            "scope": "formal_queries_vector_cache",
            "encoder_snapshot": self.encoder_snapshot,
            "encoder_snapshot_sha256": canonical_digest(self.encoder_snapshot),
            "query_count": len(self._entries),
            "queries": self._entries,
        }

    def save(self, path: Path | str) -> Path:
        """Save cache JSON to path."""
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        doc = self.to_dict()
        tmp = p.with_name(f"{p.name}.{hashlib.sha256(p.name.encode()).hexdigest()[:8]}.tmp")
        try:
            tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(p)
        finally:
            if tmp.is_file():
                tmp.unlink()
        return p

    @classmethod
    def load(cls, path: Path | str) -> QueryVectorCache:
        """Load and verify an existing query cache document."""
        p = Path(path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Query cache file not found: {p}")
        doc = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            raise ValueError(f"Invalid query cache structure: {p}")
        schema = doc.get("schema_version")
        if schema != QUERY_CACHE_SCHEMA:
            raise ValueError(f"Unsupported query cache schema version: {schema}")

        instance = cls(
            encoder_snapshot=doc.get("encoder_snapshot", {}),
            schema_version=schema,
        )
        queries = doc.get("queries", {})
        if not isinstance(queries, dict):
            raise ValueError("Cache queries field must be a dictionary")

        for key, record in queries.items():
            text = record.get("instruction") or key
            vector = record.get("vector")
            vec_sha = record.get("vector_sha256")
            if vector is None:
                continue
            if vec_sha and canonical_digest(vector) != vec_sha:
                raise ValueError(f"Vector SHA256 mismatch for query: {text[:30]}")
            instance.register(
                text,
                vector,
                task_id=record.get("task_id"),
                audit=record.get("encoding_audit"),
            )
        return instance


class OfflineQueryEncoder:
    """Zero-GPU encoder adapter satisfying the VisualMemoryRetriever encoder protocol.

    Reads query embeddings directly from a precomputed QueryVectorCache.
    Consumes 0 MB GPU VRAM and incurs zero neural network inference overhead.
    """

    supports_image_weight = True

    def __init__(
        self,
        query_cache: QueryVectorCache | Path | str | Mapping[str, Any],
        encoder_snapshot: dict[str, Any] | None = None,
        *,
        fallback_encoder: Any = None,
    ):
        if isinstance(query_cache, (str, Path)):
            self.query_cache = QueryVectorCache.load(query_cache)
        elif isinstance(query_cache, QueryVectorCache):
            self.query_cache = query_cache
        elif isinstance(query_cache, Mapping):
            self.query_cache = QueryVectorCache(encoder_snapshot=encoder_snapshot)
            for k, v in query_cache.items():
                if isinstance(v, (list, tuple, np.ndarray)):
                    self.query_cache.register(k, v)
                elif isinstance(v, dict) and "vector" in v:
                    self.query_cache.register(k, v["vector"], task_id=v.get("task_id"), audit=v.get("encoding_audit"))
        else:
            raise TypeError(f"Unsupported query_cache type: {type(query_cache)}")

        self.snapshot = encoder_snapshot or self.query_cache.encoder_snapshot
        if not self.snapshot and fallback_encoder is not None:
            self.snapshot = getattr(fallback_encoder, "snapshot", {})
        self.device = "offline_cpu"
        self.fallback_encoder = fallback_encoder
        self.last_encoding_audit = None

    def encode_text(self, text: str) -> np.ndarray:
        """Lookup query vector in cache; falls back to fallback_encoder if provided."""
        norm_text = canonical_text(text)
        if norm_text in self.query_cache:
            self.last_encoding_audit = self.query_cache.get_audit(norm_text)
            return self.query_cache.get(norm_text)
        if self.fallback_encoder is not None and callable(getattr(self.fallback_encoder, "encode_text", None)):
            vec = self.fallback_encoder.encode_text(text)
            self.last_encoding_audit = getattr(self.fallback_encoder, "last_encoding_audit", None)
            return unit(vec)
        raise KeyError(
            f"Query instruction {norm_text!r} not in precomputed cache and no fallback encoder provided."
        )

    def encode(self, path: Path | str) -> np.ndarray:
        """VisualMemoryRetriever bank vectors are loaded from disk cache; image encode only needed on cache miss."""
        if self.fallback_encoder is not None and callable(getattr(self.fallback_encoder, "encode", None)):
            return self.fallback_encoder.encode(path)
        raise NotImplementedError(
            "OfflineQueryEncoder is a Zero-GPU query reader. Bank images must be loaded from retrieval_index_cache."
        )

    def encode_joint(self, text: str, path: Path | str, image_weight: float = 0.5) -> np.ndarray:
        if self.fallback_encoder is not None and callable(getattr(self.fallback_encoder, "encode_joint", None)):
            return self.fallback_encoder.encode_joint(text, path, image_weight=image_weight)
        raise NotImplementedError(
            "OfflineQueryEncoder is a Zero-GPU query reader. Bank entries must be loaded from retrieval_index_cache."
        )

    def close(self):
        """No-op for offline zero-GPU reader; closes fallback encoder if present."""
        if self.fallback_encoder is not None and callable(getattr(self.fallback_encoder, "close", None)):
            self.fallback_encoder.close()


def precompute_query_cache(
    queries: Sequence[str | Mapping[str, Any]],
    encoder: Any,
    output_path: Path | str | None = None,
) -> QueryVectorCache:
    """Precompute embeddings for a sequence of queries using any text encoder.

    Args:
        queries: List of instruction strings or query dictionaries containing 'instruction'.
        encoder: Encoder object implementing encode_text(text) and .snapshot attribute.
        output_path: Optional path to save the serialized QueryVectorCache.

    Returns:
        Populated QueryVectorCache instance.
    """
    if not callable(getattr(encoder, "encode_text", None)):
        raise ValueError("Provided encoder does not implement encode_text")

    snapshot = getattr(encoder, "snapshot", {})
    cache = QueryVectorCache(encoder_snapshot=snapshot)

    for item in queries:
        if isinstance(item, str):
            instruction = item
            task_id = None
        elif isinstance(item, Mapping):
            instruction = item["instruction"]
            task_id = item.get("task_id")
        else:
            raise TypeError(f"Invalid query item type: {type(item)}")

        norm_text = canonical_text(instruction)
        if norm_text in cache:
            continue
        vec = encoder.encode_text(norm_text)
        audit = getattr(encoder, "last_encoding_audit", None)
        cache.register(norm_text, vec, task_id=task_id, audit=audit)

    if output_path is not None:
        cache.save(output_path)

    return cache


def build_groups9_query_cache(
    base_dir: Path | str,
    encoder: Any = None,
    output_path: Path | str | None = None,
) -> QueryVectorCache:
    """Build the global query cache for all 9 groups in experiment_base_groups9.

    Reads evaluation_inputs/<group_key>/queries.json for all groups, encodes
    all 180 instructions, and saves to shared_memory/query_index_cache/.
    """
    base_path = Path(base_dir).resolve()
    manifest_path = base_path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Base manifest not found: {manifest_path}")

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    all_queries: list[dict[str, Any]] = []
    groups = manifest.get("groups", {})
    for g_key in sorted(groups.keys()):
        q_file = base_path / "evaluation_inputs" / g_key / "queries.json"
        if not q_file.is_file():
            continue
        with open(q_file, encoding="utf-8") as f:
            q_data = json.load(f)
        instructions = q_data.get("instructions", [])
        for idx, text in enumerate(instructions, 1):
            all_queries.append({
                "group": g_key,
                "task_id": f"feature_{idx:02d}",
                "instruction": text,
            })

    if encoder is None:
        from src.experiments.shared_evaluation import make_eval_encoder
        encoder_config = base_path / "evaluation_encoder.json"
        encoder = make_eval_encoder(device="cpu", config_path=encoder_config)
        close_needed = True
    else:
        close_needed = False

    if output_path is None:
        out_dir = base_path / "shared_memory" / "query_index_cache"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / "formal_queries_vit_b32.json"

    try:
        cache = precompute_query_cache(all_queries, encoder, output_path=output_path)
    finally:
        if close_needed and callable(getattr(encoder, "close", None)):
            encoder.close()

    return cache

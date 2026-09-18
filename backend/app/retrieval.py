"""
Retrieval layer for Darukaa.Earth Biodiversity Intelligence Chatbot.
Production-grade hybrid RAG pipeline:
  1. Contextual chunk embedding (Anthropic's "Contextual Retrieval" technique)
  2. Dense search (MiniLM embeddings via ChromaDB)
  3. BM25 sparse keyword search (rank_bm25)
  4. Reciprocal Rank Fusion (RRF) to merge dense + sparse
  5. Cross-encoder reranking (ms-marco-MiniLM-L-6-v2, local CPU)
  6. Query construction from conversation state
  7. LRU in-memory cache for repeated queries
  8. Full resilience — every stage wrapped in try/except with graceful fallback
"""

import json
import os
import re
import hashlib
import logging
import time
from collections import OrderedDict
from typing import Optional

try:
    import chromadb
except ImportError:
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logger = logging.getLogger("darukaa.retrieval")

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
DOCUMENTS_DIR = os.path.join(BASE_DIR, "knowledge", "documents")
GRAPH_PATH = os.path.join(BASE_DIR, "knowledge", "graph.json")
COLLECTION_NAME = "biodiversity_knowledge_v2"

import math

# RRF constant (standard value from Cormack et al.)
RRF_K = 60

# Cache settings
MAX_CACHE_SIZE = 64

# Configurable Relevance / Confidence Thresholds
# For MS-MARCO MiniLM Cross-Encoder: logits >= 0.0 indicate semantic relevance; negative logits are irrelevant.
RERANK_RELEVANCE_THRESHOLD = float(os.getenv("RERANK_RELEVANCE_THRESHOLD", "0.0"))
# For ChromaDB cosine space: cosine similarity (1.0 - distance) >= 0.58
DENSE_RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", os.getenv("DENSE_RELEVANCE_THRESHOLD", "0.58")))
# For BM25 sparse keyword score:
BM25_RELEVANCE_THRESHOLD = float(os.getenv("BM25_RELEVANCE_THRESHOLD", "3.0"))

# Common English stopwords to prevent BM25 false positives on question words / FAQ headers
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself",
    "me", "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "she", "should", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what",
    "when", "where", "which", "while", "who", "whom", "why", "with", "would", "you",
    "your", "yours", "yourself", "yourselves", "question"
}

# ---------------------------------------------------------------------------
# Lazy-loaded globals
# ---------------------------------------------------------------------------
_client = None
_collection = None
_model = None
_cross_encoder = None
_cross_encoder_failed = False  # sticky flag — don't retry after first failure
_bm25_index = None
_bm25_corpus = None       # list of tokenised docs (list[list[str]])
_bm25_chunk_ids = None     # list of chunk IDs parallel to _bm25_corpus
_bm25_raw_texts = None     # list of raw texts parallel to _bm25_corpus
_bm25_metadata = None      # list of metadata dicts parallel to _bm25_corpus
_query_cache: OrderedDict = OrderedDict()

# ---------------------------------------------------------------------------
# Health status (set at startup, inspected by /health)
# ---------------------------------------------------------------------------
_health = {
    "chromadb_ok": False,
    "chromadb_count": 0,
    "bm25_ok": False,
    "bm25_count": 0,
    "reranker_ok": False,
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
}


def get_health() -> dict:
    """Return current retrieval subsystem health."""
    return dict(_health)


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _get_model() -> Optional[SentenceTransformer]:
    """Lazy-load the sentence-transformer bi-encoder embedding model."""
    global _model
    if SentenceTransformer is None:
        return None
    if _model is None:
        try:
            _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception as e:
            logger.warning(f"[BiEncoder] Could not load SentenceTransformer: {e}")
            return None
    return _model


def _get_collection():
    """Lazy-load the ChromaDB client and collection."""
    global _client, _collection
    if chromadb is None:
        return None
    if _collection is None:
        try:
            _client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
            _collection = _client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            logger.warning(f"[ChromaDB] Could not load collection: {e}")
            return None
    return _collection


def _get_cross_encoder():
    """
    Lazy-load the cross-encoder reranker model.
    If it fails to load, set _cross_encoder_failed = True so we never retry.
    """
    global _cross_encoder, _cross_encoder_failed
    if _cross_encoder is not None:
        return _cross_encoder
    if _cross_encoder_failed:
        return None
    try:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("[Reranker] cross-encoder/ms-marco-MiniLM-L-6-v2 loaded successfully")
        _health["reranker_ok"] = True
        return _cross_encoder
    except Exception as e:
        logger.warning(f"[Reranker] Failed to load cross-encoder: {e}. Reranking disabled.")
        _cross_encoder_failed = True
        _health["reranker_ok"] = False
        return None


# ---------------------------------------------------------------------------
# BM25 index management
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words, filtering common stop words."""
    words = re.findall(r'\w+', text.lower())
    filtered = [w for w in words if w not in STOPWORDS and len(w) > 1]
    return filtered if filtered else words


def _build_bm25_index(chunk_ids: list[str], texts: list[str], metadatas: list[dict]):
    """Build or rebuild the BM25 sparse index from chunk corpus."""
    global _bm25_index, _bm25_corpus, _bm25_chunk_ids, _bm25_raw_texts, _bm25_metadata
    try:
        from rank_bm25 import BM25Okapi
        tokenized = [_tokenize(t) for t in texts]
        _bm25_index = BM25Okapi(tokenized)
        _bm25_corpus = tokenized
        _bm25_chunk_ids = list(chunk_ids)
        _bm25_raw_texts = list(texts)
        _bm25_metadata = list(metadatas)
        _health["bm25_ok"] = True
        _health["bm25_count"] = len(chunk_ids)
        logger.info(f"[BM25] Index built: {len(chunk_ids)} documents")
    except Exception as e:
        logger.warning(f"[BM25] Failed to build index: {e}")
        _health["bm25_ok"] = False
        _bm25_index = None


# ---------------------------------------------------------------------------
# Contextual header generation
# ---------------------------------------------------------------------------

def _build_contextual_header(chunk: dict) -> str:
    """
    Build a short contextual header for a chunk (Anthropic's Contextual Retrieval).
    Prepended to the chunk text before embedding so short/ambiguous chunks still
    embed with enough context to be found by unrelated-sounding queries.
    """
    source = chunk.get("source", "unknown source")
    tags = chunk.get("topic_tags", [])
    metrics = chunk.get("linked_metrics", [])

    tags_str = ", ".join(tags) if tags else "general ecology"
    metrics_str = ", ".join(metrics) if metrics else "environmental indicators"

    return (
        f"This chunk is from a {source} evidence record "
        f"about {tags_str}, discussing {metrics_str}. "
    )


# ---------------------------------------------------------------------------
# Chunking utility (for future raw document ingestion)
# ---------------------------------------------------------------------------

def chunk_text(text: str, max_tokens: int = 450, overlap_pct: float = 0.15) -> list[str]:
    """
    Split long text into ~400-500 token chunks with ~15% overlap.
    Splits on paragraph boundaries where possible, never mid-sentence.
    
    Args:
        text: Raw text to chunk
        max_tokens: Target max tokens per chunk (default 450, within 400-500 range)
        overlap_pct: Overlap as a fraction (default 0.15 = 15%)
    
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []

    # Split into paragraphs first
    paragraphs = re.split(r'\n\s*\n', text.strip())
    
    # Estimate tokens (rough: 1 token ≈ 0.75 words for English)
    def est_tokens(s: str) -> int:
        return max(1, int(len(s.split()) / 0.75))

    overlap_tokens = int(max_tokens * overlap_pct)
    chunks = []
    current_chunk_parts = []
    current_tokens = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_tokens = est_tokens(para)

        # If a single paragraph exceeds max, split by sentences
        if para_tokens > max_tokens:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sentence in sentences:
                s_tokens = est_tokens(sentence)
                if current_tokens + s_tokens > max_tokens and current_chunk_parts:
                    chunks.append(" ".join(current_chunk_parts))
                    # Overlap: keep last portion
                    overlap_text = " ".join(current_chunk_parts)
                    overlap_words = overlap_text.split()
                    keep = int(len(overlap_words) * overlap_pct)
                    if keep > 0:
                        current_chunk_parts = [" ".join(overlap_words[-keep:])]
                        current_tokens = est_tokens(current_chunk_parts[0])
                    else:
                        current_chunk_parts = []
                        current_tokens = 0
                current_chunk_parts.append(sentence)
                current_tokens += s_tokens
        elif current_tokens + para_tokens > max_tokens and current_chunk_parts:
            chunks.append("\n\n".join(current_chunk_parts))
            # Overlap: keep last paragraph if short enough
            overlap_text = current_chunk_parts[-1] if est_tokens(current_chunk_parts[-1]) <= overlap_tokens else ""
            current_chunk_parts = [overlap_text] if overlap_text else []
            current_tokens = est_tokens(overlap_text) if overlap_text else 0
            current_chunk_parts.append(para)
            current_tokens += para_tokens
        else:
            current_chunk_parts.append(para)
            current_tokens += para_tokens

    if current_chunk_parts:
        final = "\n\n".join(current_chunk_parts) if len(current_chunk_parts) > 1 else " ".join(current_chunk_parts)
        chunks.append(final.strip())

    return chunks


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_documents():
    """
    Read evidence chunks from /knowledge/documents, build contextual headers,
    embed context-augmented text into ChromaDB, and build the BM25 index.
    
    Stores both original_text (for display) and contextual_text (for embedding)
    in chunk metadata so retrieval can show clean text to the user while
    searching over the context-enriched version.
    
    Returns a summary dict of chunks ingested per topic tag.
    """
    model = _get_model()
    collection = _get_collection()

    # Load all JSON chunk files
    all_chunks = []
    for filename in os.listdir(DOCUMENTS_DIR):
        if filename.endswith(".json") and "DEPRECATED" not in filename:
            filepath = os.path.join(DOCUMENTS_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_chunks.extend(data)
                elif isinstance(data, dict):
                    all_chunks.append(data)

    if not all_chunks:
        print("No chunks found in", DOCUMENTS_DIR)
        return {}

    # Prepare for batch upsert
    ids = []
    embedding_texts = []  # context-augmented text for embedding
    display_texts = []    # original text for display
    metadatas = []

    for chunk in all_chunks:
        chunk_id = chunk["id"]
        original_text = chunk["text"]
        contextual_header = _build_contextual_header(chunk)
        contextual_text = contextual_header + original_text

        ids.append(chunk_id)
        embedding_texts.append(contextual_text)
        display_texts.append(original_text)
        metadatas.append({
            "source": chunk.get("source", ""),
            "url_or_citation": chunk.get("url_or_citation", ""),
            "topic_tags": json.dumps(chunk.get("topic_tags", [])),
            "linked_metrics": json.dumps(chunk.get("linked_metrics", [])),
            "original_text": original_text,
            "contextual_text": contextual_text,
        })

    # Embed contextual texts (not raw text!)
    embeddings = model.encode(embedding_texts, show_progress_bar=True).tolist()

    # Upsert into ChromaDB — document field stores the ORIGINAL text for display
    collection.upsert(
        ids=ids,
        documents=display_texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    # Update health
    _health["chromadb_ok"] = True
    _health["chromadb_count"] = collection.count()

    # Build BM25 index over the contextual texts (richer matching surface)
    _build_bm25_index(ids, embedding_texts, metadatas)

    # Build summary by topic tag
    tag_counts = {}
    for chunk in all_chunks:
        for tag in chunk.get("topic_tags", []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    print(f"\n[OK] Ingested {len(all_chunks)} chunks into ChromaDB collection '{COLLECTION_NAME}'")
    print(f"     Stored at: {CHROMA_DB_DIR}")
    print(f"     Contextual headers prepended for embedding")
    print(f"     BM25 sparse index built: {len(ids)} documents")
    print(f"\n  Sample contextual header:")
    if embedding_texts:
        header_end = embedding_texts[0].find(". ") + 2
        print(f"    \"{embedding_texts[0][:min(header_end + 50, 200)]}...\"")
    print(f"\n[+] Chunks per topic tag:")
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
        print(f"   {tag}: {count}")

    return tag_counts


# ---------------------------------------------------------------------------
# Query construction from conversation state
# ---------------------------------------------------------------------------

def build_retrieval_query(user_message: str, slots: dict | None = None) -> str:
    """
    Build a retrieval query by combining the user's current message with all
    known slot values. This gives retrieval the full environmental context on
    every turn, not just whatever the user typed.
    
    Example output:
        "soil_organic_carbon: 0.3%, land_use: monoculture wheat, rainfall: low
         — question: what should I do?"
    """
    parts = []

    if slots:
        slot_parts = []
        slot_labels = {
            "soil_organic_carbon_pct": "soil organic carbon",
            "soil_ph": "soil pH",
            "soil_moisture": "soil moisture",
            "land_use_type": "land use",
            "rainfall_pattern": "rainfall",
            "region_climate_zone": "climate zone",
            "species_richness_observation": "species richness",
            "pollution_or_deforestation_pressure": "pollution/deforestation pressure",
        }
        for key, label in slot_labels.items():
            val = slots.get(key)
            if val is not None:
                slot_parts.append(f"{label}: {val}")
        if slot_parts:
            parts.append(", ".join(slot_parts))

    if user_message and user_message.strip():
        parts.append(f"question: {user_message.strip()}")

    return " — ".join(parts) if parts else user_message


# ---------------------------------------------------------------------------
# LRU Cache
# ---------------------------------------------------------------------------

def _cache_key(query: str, topic_filter: list[str] | None, k: int) -> str:
    """Generate a deterministic cache key for a retrieval query."""
    raw = f"{query}|{sorted(topic_filter) if topic_filter else ''}|{k}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _cache_get(key: str):
    """Get from LRU cache, moving to end (most recently used)."""
    if key in _query_cache:
        _query_cache.move_to_end(key)
        return _query_cache[key]
    return None


def _cache_put(key: str, value):
    """Put into LRU cache, evicting oldest if over capacity."""
    _query_cache[key] = value
    _query_cache.move_to_end(key)
    while len(_query_cache) > MAX_CACHE_SIZE:
        _query_cache.popitem(last=False)


# ---------------------------------------------------------------------------
# Core search functions
# ---------------------------------------------------------------------------

def _dense_search(query: str, topic_filter: list[str] | None = None, n: int = 20) -> list[dict]:
    """
    Dense vector search via ChromaDB.
    Returns list of {id, text, source, score, topic_tags, linked_metrics}.
    """
    model = _get_model()
    collection = _get_collection()

    if model is None or collection is None or collection.count() == 0:
        return []

    query_embedding = model.encode([query]).tolist()

    # Build where filter for topic tags if provided
    where_filter = None
    if topic_filter and len(topic_filter) > 0:
        if len(topic_filter) == 1:
            where_filter = {"topic_tags": {"$contains": topic_filter[0]}}
        else:
            where_filter = {
                "$or": [{"topic_tags": {"$contains": tag}} for tag in topic_filter]
            }

    try:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(n, collection.count()),
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
    except Exception:
        # If filter fails, retry without filter
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(n, collection.count()),
            include=["documents", "metadatas", "distances"]
        )

    formatted = []
    if results and results["ids"] and len(results["ids"]) > 0:
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else 0.0
            similarity = 1.0 - (distance / 2.0)

            formatted.append({
                "id": doc_id,
                "text": meta.get("original_text", results["documents"][0][i]),
                "source": meta.get("source", ""),
                "document_title": meta.get("document_title", ""),
                "page_number": meta.get("page_number"),
                "source_file": meta.get("source_file", ""),
                "score": round(similarity, 4),
                "topic_tags": json.loads(meta.get("topic_tags", "[]")),
                "linked_metrics": json.loads(meta.get("linked_metrics", "[]"))
            })

    return formatted


def _bm25_search(query: str, n: int = 20) -> list[dict]:
    """
    BM25 sparse keyword search.
    Returns list of {id, text, source, score, topic_tags, linked_metrics}.
    """
    if _bm25_index is None or _bm25_chunk_ids is None:
        return []

    tokenized_query = _tokenize(query)
    if not tokenized_query:
        return []

    scores = _bm25_index.get_scores(tokenized_query)

    # Get top-n indices by score
    indexed_scores = list(enumerate(scores))
    indexed_scores.sort(key=lambda x: x[1], reverse=True)
    top_n = indexed_scores[:n]

    formatted = []
    for idx, bm25_score in top_n:
        if bm25_score <= 0:
            continue
        meta = _bm25_metadata[idx] if _bm25_metadata and idx < len(_bm25_metadata) else {}
        formatted.append({
            "id": _bm25_chunk_ids[idx],
            "text": meta.get("original_text", _bm25_raw_texts[idx] if _bm25_raw_texts else ""),
            "source": meta.get("source", ""),
            "document_title": meta.get("document_title", ""),
            "page_number": meta.get("page_number"),
            "source_file": meta.get("source_file", ""),
            "score": round(float(bm25_score), 4),
            "topic_tags": json.loads(meta.get("topic_tags", "[]")),
            "linked_metrics": json.loads(meta.get("linked_metrics", "[]"))
        })

    return formatted


def _reciprocal_rank_fusion(ranked_lists: list[list[dict]], k: int = RRF_K) -> list[dict]:
    """
    Reciprocal Rank Fusion: merge multiple ranked lists into one.
    score(doc) = sum(1 / (k + rank_in_list)) across all lists.
    
    Args:
        ranked_lists: List of ranked result lists (each item must have 'id')
        k: RRF constant (default 60)
    
    Returns:
        Fused list sorted by RRF score descending
    """
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            doc_id = doc["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

    # Sort by RRF score
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    fused = []
    for doc_id in sorted_ids:
        doc = dict(doc_map[doc_id])
        doc["rrf_score"] = round(rrf_scores[doc_id], 6)
        fused.append(doc)

    return fused


def _rerank(query: str, candidates: list[dict], top_k: int = 6) -> tuple[list[dict], bool]:
    """
    Cross-encoder reranking of candidates.
    
    Returns:
        (reranked_list, did_rerank) — if reranker fails, returns candidates
        in original order and did_rerank=False.
    """
    encoder = _get_cross_encoder()
    if encoder is None or not candidates:
        return candidates[:top_k], False

    try:
        pairs = [(query, c["text"]) for c in candidates]
        scores = encoder.predict(pairs)
        
        # Attach cross-encoder score and sort
        for i, c in enumerate(candidates):
            c["rerank_score"] = float(scores[i])
        
        candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        return candidates[:top_k], True
    except Exception as e:
        logger.warning(f"[Reranker] Runtime error: {e}. Falling back to RRF order.")
        return candidates[:top_k], False


# ---------------------------------------------------------------------------
# Graph-based fallback
# ---------------------------------------------------------------------------

def _graph_fallback(slots: dict | None) -> list[dict]:
    """
    Last-resort fallback: return graph.json interventions that structurally
    match the user's known slots (rule-based, no embeddings needed).
    """
    try:
        from backend.app.reasoning import rank_interventions
        if slots:
            ranked = rank_interventions(slots, min_overlap=1)
            if ranked:
                # Convert graph interventions to chunk-like format
                return [{
                    "id": f"graph_{r['name'].replace(' ', '_').lower()}",
                    "text": f"{r['name']}: {r.get('mechanism', '')}",
                    "source": r.get("source", "knowledge_graph"),
                    "score": r.get("score", 0) / 10.0,
                    "topic_tags": r.get("preconditions", []),
                    "linked_metrics": r.get("affects_metrics", []),
                    "fallback": True
                } for r in ranked[:6]]
    except Exception as e:
        logger.warning(f"[Fallback] Graph fallback failed: {e}")
    return []


# ---------------------------------------------------------------------------
# Relevance Calibration & Confidence Evaluation
# ---------------------------------------------------------------------------

def calibrate_chunk_score(chunk: dict, did_rerank: bool) -> tuple[float, str]:
    """
    Produce a normalized [0.0, 1.0] confidence score and qualitative relevance label.
    Prevents displaying misleading raw values or uncalibrated percentages in the UI.
    """
    if did_rerank and "rerank_score" in chunk:
        r_score = chunk["rerank_score"]
        # Sigmoid normalization: 0.0 -> 0.50, 2.0 -> 0.69, 4.0 -> 0.83, -2.0 -> 0.31
        norm = round(1.0 / (1.0 + math.exp(-r_score / 2.5)), 4)
        if r_score >= 2.0:
            label = "Highly Relevant"
        elif r_score >= RERANK_RELEVANCE_THRESHOLD:
            label = "Relevant"
        elif r_score >= -1.0:
            label = "Moderately Relevant"
        else:
            label = "Low Relevance"
        return norm, label

    # Dense vector similarity (cosine space: 1.0 - distance)
    if "cosine_similarity" in chunk:
        sim = chunk["cosine_similarity"]
        norm = round(sim, 4)
        if sim >= 0.72:
            label = "Highly Relevant"
        elif sim >= DENSE_RELEVANCE_THRESHOLD:
            label = "Relevant"
        elif sim >= 0.45:
            label = "Moderately Relevant"
        else:
            label = "Low Relevance"
        return norm, label

    # BM25 score
    if "bm25_score" in chunk:
        bm25 = chunk["bm25_score"]
        norm = round(min(1.0, max(0.0, bm25 / 15.0)), 4)
        label = "Relevant" if bm25 >= BM25_RELEVANCE_THRESHOLD else "Low Relevance"
        return norm, label

    raw = chunk.get("score", 0.0)
    try:
        val = float(raw)
    except (ValueError, TypeError):
        val = 0.0
    norm = round(max(0.0, min(1.0, val)), 4)
    return norm, "Relevant" if norm >= 0.5 else "Low Relevance"


def evaluate_internal_evidence(
    candidates: list[dict],
    did_rerank: bool = False,
    min_required: int = 1
) -> tuple[list[dict], bool, dict]:
    """
    Strictly filters retrieved internal chunks against relevance thresholds.
    Ensures unrelated documents (e.g. general IPCC climate report chunks for
    unrelated soil pH or farming queries) are NOT passed along.

    Returns:
        (filtered_candidates, is_sufficient, debug_info)
    """
    accepted = []
    rejected = []

    for c in candidates:
        norm_score, label = calibrate_chunk_score(c, did_rerank)
        c["calibrated_score"] = norm_score
        c["score"] = norm_score  # update score to calibrated [0.0, 1.0] for frontend/API
        c["relevance_label"] = label

        # Threshold check
        is_relevant = False
        if did_rerank and "rerank_score" in c:
            is_relevant = c["rerank_score"] >= RERANK_RELEVANCE_THRESHOLD
        elif "cosine_similarity" in c:
            is_relevant = c["cosine_similarity"] >= DENSE_RELEVANCE_THRESHOLD
        elif "bm25_score" in c:
            is_relevant = c["bm25_score"] >= BM25_RELEVANCE_THRESHOLD
        else:
            is_relevant = c.get("score", 0.0) >= 0.55

        if is_relevant:
            accepted.append(c)
        else:
            rejected.append({
                "id": c.get("id"),
                "rerank_score": c.get("rerank_score"),
                "score": norm_score,
                "label": label
            })

    is_sufficient = len(accepted) >= min_required

    debug_info = {
        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "is_sufficient": is_sufficient,
        "rerank_threshold": RERANK_RELEVANCE_THRESHOLD,
        "dense_threshold": DENSE_RELEVANCE_THRESHOLD,
        "bm25_threshold": BM25_RELEVANCE_THRESHOLD,
        "did_rerank": did_rerank,
        "top_accepted_score": accepted[0].get("score") if accepted else None,
        "top_accepted_label": accepted[0].get("relevance_label") if accepted else None,
    }

    logger.info(
        f"[InternalEvidenceEvaluation] Candidates: {len(candidates)}, "
        f"Accepted: {len(accepted)}, Sufficient: {is_sufficient}, TopScore: {debug_info['top_accepted_score']}"
    )

    return accepted, is_sufficient, debug_info


# ---------------------------------------------------------------------------
# Main hybrid retrieval function
# ---------------------------------------------------------------------------

def hybrid_retrieve(
    query: str,
    topic_filter: list[str] | None = None,
    k: int = 6,
    slots: dict | None = None,
    use_cache: bool = True,
) -> tuple[list[dict], dict]:
    """
    Production-grade hybrid retrieval pipeline with relevance thresholding.
    
    Pipeline stages:
        1. Dense search (ChromaDB MiniLM embeddings) -> top-20
        2. BM25 sparse keyword search -> top-20
        3. Reciprocal Rank Fusion to merge -> top-20
        4. Cross-encoder reranking -> top-k
        5. Relevance threshold evaluation -> filtered accepted results
    
    Returns:
        (results, retrieval_trace) tuple
    """
    trace = {
        "query_used": query,
        "dense_candidates": 0,
        "bm25_candidates": 0,
        "rrf_candidates": 0,
        "reranker_ran": False,
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "final_chunk_ids": [],
        "fallback_used": None,
        "cache_hit": False,
        "latency_ms": 0,
        "is_sufficient": False,
        "relevance_eval": None
    }

    start_time = time.time()

    # Check cache
    if use_cache:
        ckey = _cache_key(query, topic_filter, k)
        cached = _cache_get(ckey)
        if cached is not None:
            trace["cache_hit"] = True
            trace["final_chunk_ids"] = [c["id"] for c in cached]
            trace["is_sufficient"] = len(cached) > 0
            trace["latency_ms"] = round((time.time() - start_time) * 1000, 1)
            return cached, trace

    # Stage 1: Dense search
    dense_results = []
    try:
        dense_results = _dense_search(query, topic_filter=topic_filter, n=20)
        trace["dense_candidates"] = len(dense_results)
    except Exception as e:
        logger.warning(f"[Pipeline] Dense search failed: {e}")
        trace["fallback_used"] = "dense_failed"

    # Stage 2: BM25 search
    bm25_results = []
    try:
        bm25_results = _bm25_search(query, n=20)
        trace["bm25_candidates"] = len(bm25_results)
    except Exception as e:
        logger.warning(f"[Pipeline] BM25 search failed: {e}")
        if trace["fallback_used"]:
            trace["fallback_used"] += "+bm25_failed"
        else:
            trace["fallback_used"] = "bm25_failed"

    # Stage 3: Reciprocal Rank Fusion
    fused = []
    try:
        lists_to_fuse = [l for l in [dense_results, bm25_results] if l]
        if lists_to_fuse:
            fused = _reciprocal_rank_fusion(lists_to_fuse)
            trace["rrf_candidates"] = len(fused)
        elif dense_results:
            fused = dense_results
            trace["fallback_used"] = "bm25_unavailable_dense_only"
        elif bm25_results:
            fused = bm25_results
            trace["fallback_used"] = "dense_unavailable_bm25_only"
    except Exception as e:
        logger.warning(f"[Pipeline] RRF fusion failed: {e}")
        fused = dense_results or bm25_results
        trace["fallback_used"] = "rrf_failed"

    # Stage 4: Cross-encoder reranking
    raw_results = []
    did_rerank = False
    if fused:
        try:
            raw_results, did_rerank = _rerank(query, fused, top_k=k)
            trace["reranker_ran"] = did_rerank
            if not did_rerank:
                trace["fallback_used"] = (trace.get("fallback_used") or "") + ("" if not trace.get("fallback_used") else "+") + "reranker_skipped"
        except Exception as e:
            logger.warning(f"[Pipeline] Reranking failed: {e}")
            raw_results = fused[:k]
            trace["fallback_used"] = (trace.get("fallback_used") or "") + "+reranker_failed"
    else:
        raw_results = []

    # Stage 5: Strict Relevance Thresholding
    # Do NOT blindly return top-K results if they are not relevant!
    final_results, is_sufficient, eval_debug = evaluate_internal_evidence(
        raw_results, did_rerank=did_rerank, min_required=1
    )
    trace["is_sufficient"] = is_sufficient
    trace["relevance_eval"] = eval_debug

    # Fallback to relationship graph ONLY if user has provided environmental slots (for agricultural interventions)
    # Never inject graph fallbacks for general Q&A questions!
    has_known_slots = bool(slots and any(v is not None for k, v in slots.items() if k not in ["geo_lat", "geo_lon", "country"]))
    if not final_results and has_known_slots:
        logger.info("[Pipeline] No internal documents passed threshold. Attempting graph intervention fallback for known slots.")
        graph_results = _graph_fallback(slots)
        if graph_results:
            final_results = graph_results
            trace["fallback_used"] = "graph_based_zero_result_fallback"
            trace["is_sufficient"] = True

    trace["final_chunk_ids"] = [c["id"] for c in final_results]
    trace["latency_ms"] = round((time.time() - start_time) * 1000, 1)

    # Cache the results
    if use_cache and final_results:
        _cache_put(_cache_key(query, topic_filter, k), final_results)

    return final_results, trace


# ---------------------------------------------------------------------------
# Legacy-compatible wrapper (so existing code doesn't break)
# ---------------------------------------------------------------------------

def retrieve(query: str, topic_filter: list[str] | None = None, k: int = 6) -> list[dict]:
    """
    Legacy-compatible retrieval function.
    Calls hybrid_retrieve internally but returns only the results list
    (no trace) to maintain backward compatibility.
    """
    results, _trace = hybrid_retrieve(query, topic_filter=topic_filter, k=k)
    return results


# ---------------------------------------------------------------------------
# Collection stats
# ---------------------------------------------------------------------------

def get_collection_stats() -> dict:
    """Return basic stats about the knowledge base."""
    try:
        collection = _get_collection()
        count = collection.count()
        _health["chromadb_ok"] = count > 0
        _health["chromadb_count"] = count
        return {
            "total_chunks": count,
            "collection_name": COLLECTION_NAME,
            "db_path": CHROMA_DB_DIR,
            "bm25_indexed": _health["bm25_ok"],
            "bm25_count": _health["bm25_count"],
            "reranker_loaded": _health["reranker_ok"],
        }
    except Exception as e:
        logger.warning(f"[Stats] Failed to get collection stats: {e}")
        return {"total_chunks": 0, "error": str(e)}


# ---------------------------------------------------------------------------
# Startup health check (called from main.py)
# ---------------------------------------------------------------------------

def run_startup_health_check():
    """
    Verify retrieval subsystem components at startup.
    Logs clear errors for any that fail, but lets the app start in degraded mode.
    """
    import sys
    # Force UTF-8 on Windows to avoid cp1252 encoding errors
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("\n" + "=" * 60)
    print("  Retrieval Pipeline -- Startup Health Check")
    print("=" * 60)

    # 1. ChromaDB
    try:
        collection = _get_collection()
        count = collection.count()
        _health["chromadb_ok"] = count > 0
        _health["chromadb_count"] = count
        if count > 0:
            print(f"  [OK] ChromaDB: {count} chunks in '{COLLECTION_NAME}'")
        else:
            print(f"  [!!] ChromaDB: Collection '{COLLECTION_NAME}' is EMPTY. Run ingest.py first.")
    except Exception as e:
        print(f"  [FAIL] ChromaDB: FAILED to load -- {e}")
        _health["chromadb_ok"] = False

    # 2. BM25 index -- rebuild from ChromaDB or direct JSON chunks
    if _bm25_index is None:
        if _health["chromadb_ok"]:
            try:
                collection = _get_collection()
                count = collection.count() if collection else 0
                if count > 0:
                    all_docs = collection.get(include=["documents", "metadatas"])
                    chunk_ids = all_docs["ids"]
                    texts = []
                    metas = []
                    for i, doc_id in enumerate(chunk_ids):
                        meta = all_docs["metadatas"][i] if all_docs["metadatas"] else {}
                        text = meta.get("contextual_text", all_docs["documents"][i])
                        texts.append(text)
                        metas.append(meta)
                    _build_bm25_index(chunk_ids, texts, metas)
                    if _health["bm25_ok"]:
                        print(f"  [OK] BM25 Index: {len(chunk_ids)} documents indexed from ChromaDB")
            except Exception as e:
                print(f"  [!] BM25 Index from ChromaDB failed: {e}")

        # Fallback to direct JSON chunks in /knowledge/documents if ChromaDB is unavailable
        if _bm25_index is None:
            try:
                real_json = os.path.join(DOCUMENTS_DIR, "evidence_chunks_real.json")
                if os.path.exists(real_json):
                    with open(real_json, "r", encoding="utf-8") as f:
                        chunks_data = json.load(f)
                    chunk_ids = [c["id"] for c in chunks_data]
                    texts = [c.get("contextual_text", c.get("text", "")) for c in chunks_data]
                    metas = [{
                        "source": c.get("source", ""),
                        "document_title": c.get("document_title", ""),
                        "page_number": c.get("page_number"),
                        "source_file": c.get("source_file", ""),
                        "topic_tags": json.dumps(c.get("topic_tags", [])),
                        "linked_metrics": json.dumps(c.get("linked_metrics", [])),
                        "original_text": c.get("text", "")
                    } for c in chunks_data]
                    _build_bm25_index(chunk_ids, texts, metas)
                    if _health["bm25_ok"]:
                        print(f"  [OK] BM25 Index: {len(chunk_ids)} documents indexed from evidence_chunks_real.json")
                else:
                    print(f"  [!!] BM25 Index: evidence_chunks_real.json not found in {DOCUMENTS_DIR}")
            except Exception as e:
                print(f"  [FAIL] BM25 Index from JSON failed: {e}")
                _health["bm25_ok"] = False
    elif _bm25_index is not None:
        print(f"  [OK] BM25 Index: already loaded ({_health['bm25_count']} docs)")

    # 3. Cross-encoder reranker
    try:
        encoder = _get_cross_encoder()
        if encoder is not None:
            print(f"  [OK] Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2 loaded")
        else:
            print(f"  [!!] Reranker: UNAVAILABLE -- retrieval will work without reranking")
    except Exception as e:
        print(f"  [FAIL] Reranker: FAILED -- {e}")

    # 4. Bi-encoder
    try:
        _get_model()
        print(f"  [OK] Bi-encoder: all-MiniLM-L6-v2 loaded")
    except Exception as e:
        print(f"  [FAIL] Bi-encoder: FAILED -- {e}")

    mode = "FULL (dense+BM25+reranker)"
    if not _health["reranker_ok"]:
        mode = "DEGRADED (dense+BM25, no reranking)"
    if not _health["bm25_ok"]:
        mode = "DEGRADED (dense-only)"
    if not _health["chromadb_ok"]:
        mode = "MINIMAL (graph-only fallback)"

    print(f"\n  Pipeline mode: {mode}")
    print("=" * 60 + "\n")

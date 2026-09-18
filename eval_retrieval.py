"""
Darukaa.Earth -- Retrieval Quality Evaluation (eval_retrieval.py)

10 hand-written test queries (2 per topic category: soil, climate, land-use,
biodiversity, human-impact) paired with expected chunk IDs. Runs hybrid
retrieval for each and reports recall@5.

Usage:
    python eval_retrieval.py
"""

import sys
import os

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.app.retrieval import hybrid_retrieve, run_startup_health_check

# ─────────────────────────────────────────────────────────────────────────────
# Test queries: 10 queries, 2 per topic category
# Each has: query text, expected chunk ID(s), category label
# ─────────────────────────────────────────────────────────────────────────────
TEST_QUERIES = [
    # ── SOIL (2 queries) ──
    {
        "category": "soil",
        "query": "How does soil organic carbon affect climate change and what is the 4 per 1000 initiative?",
        "expected_ids": ["doc_001_chunk_1"],
        "description": "SOC stocks and FAO 4 per 1000"
    },
    {
        "category": "soil",
        "query": "What happens to soil microbial communities under no-till farming?",
        "expected_ids": ["doc_007_chunk_2"],
        "description": "No-till microbial shifts"
    },

    # ── CLIMATE (2 queries) ──
    {
        "category": "climate",
        "query": "How does climate change affect species extinction risk according to IPCC AR6?",
        "expected_ids": ["doc_016_chunk_1"],
        "description": "IPCC AR6 extinction projections"
    },
    {
        "category": "climate",
        "query": "What are water harvesting techniques for semi-arid regions like the Sahel?",
        "expected_ids": ["doc_011_chunk_1"],
        "description": "Zai pits and contour bunds in Sahel"
    },

    # ── LAND-USE (2 queries) ──
    {
        "category": "land-use",
        "query": "What is the biodiversity impact of monoculture cropping vs polyculture?",
        "expected_ids": ["doc_014_chunk_1"],
        "description": "Monoculture biodiversity loss"
    },
    {
        "category": "land-use",
        "query": "How does intercropping with cereal-legume combinations improve yields and pest control?",
        "expected_ids": ["doc_015_chunk_1"],
        "description": "Intercropping meta-analysis"
    },

    # ── BIODIVERSITY (2 queries) ──
    {
        "category": "biodiversity",
        "query": "How do riparian buffer strips improve water quality and create ecological corridors?",
        "expected_ids": ["doc_003_chunk_1"],
        "description": "Riparian buffer meta-analysis"
    },
    {
        "category": "biodiversity",
        "query": "How do hedgerows and field margins support pollinators and pest control?",
        "expected_ids": ["doc_004_chunk_1", "doc_004_chunk_2"],
        "description": "Hedgerow pollinators and pest control"
    },

    # ── HUMAN-IMPACT (2 queries) ──
    {
        "category": "human-impact",
        "query": "What are the five direct drivers of biodiversity loss identified by IPBES?",
        "expected_ids": ["doc_013_chunk_1"],
        "description": "IPBES five drivers of biodiversity loss"
    },
    {
        "category": "human-impact",
        "query": "How does biochar improve soil fertility and sequester carbon for centuries?",
        "expected_ids": ["doc_012_chunk_1"],
        "description": "Biochar meta-analysis"
    },
]


def compute_recall_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int = 5) -> float:
    """Compute recall@k: fraction of expected IDs that appear in the top-k retrieved."""
    top_k = set(retrieved_ids[:k])
    hits = sum(1 for eid in expected_ids if eid in top_k)
    return hits / len(expected_ids) if expected_ids else 0.0


def run_evaluation():
    """Run the full retrieval evaluation suite."""
    print("\n" + "=" * 70)
    print("  Darukaa.Earth — Retrieval Quality Evaluation")
    print("  Hybrid RAG Pipeline: Dense + BM25 + RRF + Cross-Encoder Reranking")
    print("=" * 70)

    # Run startup health check to initialise all components
    run_startup_health_check()

    print("\n" + "-" * 70)
    print(f"  Running {len(TEST_QUERIES)} test queries across 5 categories")
    print("-" * 70 + "\n")

    results = []
    category_scores = {}

    for i, test in enumerate(TEST_QUERIES, 1):
        query = test["query"]
        expected = test["expected_ids"]
        category = test["category"]
        desc = test["description"]

        # Run hybrid retrieval
        retrieved, trace = hybrid_retrieve(query, k=5)
        retrieved_ids = [r["id"] for r in retrieved]

        recall = compute_recall_at_k(retrieved_ids, expected, k=5)
        results.append(recall)

        # Track per-category
        if category not in category_scores:
            category_scores[category] = []
        category_scores[category].append(recall)

        # Print result
        status = "✓ PASS" if recall >= 1.0 else ("~ PARTIAL" if recall > 0 else "✗ MISS")
        print(f"  [{i:2d}] {status} | recall@5={recall:.0%} | {category:12s} | {desc}")
        print(f"        Query: \"{query[:80]}...\"" if len(query) > 80 else f"        Query: \"{query}\"")
        print(f"        Expected: {expected}")
        print(f"        Retrieved: {retrieved_ids[:5]}")
        print(f"        Pipeline: dense={trace['dense_candidates']}, bm25={trace['bm25_candidates']}, "
              f"reranked={trace['reranker_ran']}, latency={trace['latency_ms']}ms")
        if trace.get("fallback_used"):
            print(f"        Fallback: {trace['fallback_used']}")
        print()

    # Summary
    overall_recall = sum(results) / len(results) if results else 0.0
    perfect_count = sum(1 for r in results if r >= 1.0)

    print("=" * 70)
    print(f"  OVERALL RECALL@5: {overall_recall:.0%} ({perfect_count}/{len(results)} perfect retrievals)")
    print()
    print("  Per-category breakdown:")
    for cat, scores in sorted(category_scores.items()):
        avg = sum(scores) / len(scores)
        print(f"    {cat:15s}: {avg:.0%} ({sum(1 for s in scores if s >= 1.0)}/{len(scores)} perfect)")
    print("=" * 70)

    return overall_recall


if __name__ == "__main__":
    score = run_evaluation()
    sys.exit(0 if score >= 0.5 else 1)

"""
Darukaa.Earth — Production Real PDF Ingestion Pipeline (ingest.py)

Ingests real scientific PDF documents from /data (IPCC AR6 WGII SPM, Technical Summary,
CCP1, CCP5, Chapters 10 and 18) with:
1. Page-by-page extraction with exact page number attribution.
2. Running header/footer removal based on frequency across pages.
3. Figure/table caption separation (stored in metadata).
4. Paragraph-aware chunking (300–500 tokens).
5. Human-decodable chunk IDs (e.g. ipcc_ar6_wgii_ch10_p47_c1).
6. Anthropic-style Contextual Retrieval prefix for embedding.
7. Dual storage: original text for display, contextual text for embedding & BM25.
8. Persists ChromaDB collection 'biodiversity_knowledge_v2' and knowledge/documents/evidence_chunks_real.json.

Usage:
    python ingest.py
"""

import sys
import os
import re
import json
import glob
from typing import List, Dict, Tuple

# Force UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

try:
    from pypdf import PdfReader
except ImportError:
    import pdfplumber
    PdfReader = None

from backend.app.retrieval import (
    _get_model, _get_collection, _build_bm25_index,
    COLLECTION_NAME, CHROMA_DB_DIR
)


DATA_DIR = os.path.join(BASE_DIR, "data")
DOCUMENTS_DIR = os.path.join(BASE_DIR, "knowledge", "documents")
OUTPUT_JSON_PATH = os.path.join(DOCUMENTS_DIR, "evidence_chunks_real.json")


# Document titles & human-readable prefix mapping
DOC_META_MAP = {
    "IPCC_AR6_WGII_SummaryForPolicymakers.pdf": {
        "prefix": "ipcc_ar6_wgii_spm",
        "title": "IPCC AR6 WGII: Summary for Policymakers (2022)"
    },
    "IPCC_AR6_WGII_TechnicalSummary.pdf": {
        "prefix": "ipcc_ar6_wgii_ts",
        "title": "IPCC AR6 WGII: Technical Summary (2022)"
    },
    "IPCC_AR6_WGII_Chapter10.pdf": {
        "prefix": "ipcc_ar6_wgii_ch10",
        "title": "IPCC AR6 WGII Chapter 10: Asia (2022)"
    },
    "IPCC_AR6_WGII_Chapter10_SM.pdf": {
        "prefix": "ipcc_ar6_wgii_ch10_sm",
        "title": "IPCC AR6 WGII Chapter 10: Asia — Supplementary Material (2022)"
    },
    "IPCC_AR6_WGII_Chapter18.pdf": {
        "prefix": "ipcc_ar6_wgii_ch18",
        "title": "IPCC AR6 WGII Chapter 18: Climate Resilient Development Pathways (2022)"
    },
    "IPCC_AR6_WGII_CCP1.pdf": {
        "prefix": "ipcc_ar6_wgii_ccp1",
        "title": "IPCC AR6 WGII Cross-Chapter Paper 1: Biodiversity Hotspots (2022)"
    },
    "IPCC_AR6_WGII_CCP1_SM.pdf": {
        "prefix": "ipcc_ar6_wgii_ccp1_sm",
        "title": "IPCC AR6 WGII Cross-Chapter Paper 1: Biodiversity Hotspots — Supplementary Material (2022)"
    },
    "IPCC_AR6_WGII_CCP5.pdf": {
        "prefix": "ipcc_ar6_wgii_ccp5",
        "title": "IPCC AR6 WGII Cross-Chapter Paper 5: Mountains (2022)"
    },
}


def get_doc_metadata(filename: str) -> Tuple[str, str]:
    """Return prefix and clean title for any PDF filename."""
    base = os.path.basename(filename)
    if base in DOC_META_MAP:
        return DOC_META_MAP[base]["prefix"], DOC_META_MAP[base]["title"]
    
    clean_name = os.path.splitext(base)[0].replace("-", "_").lower()
    prefix = re.sub(r'[^a-z0-9_]', '', clean_name)
    title = base.replace(".pdf", "").replace("_", " ")
    return prefix, title


def extract_raw_pages(pdf_path: str) -> List[Tuple[int, str]]:
    """Extract page text using pypdf or pdfplumber."""
    pages = []
    if PdfReader is not None:
        try:
            reader = PdfReader(pdf_path)
            for idx, page in enumerate(reader.pages):
                txt = page.extract_text() or ""
                pages.append((idx + 1, txt))
            return pages
        except Exception as e:
            print(f"    [!] pypdf error on {pdf_path}: {e}, trying pdfplumber fallback...")

    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        for idx, page in enumerate(pdf.pages):
            txt = page.extract_text() or ""
            pages.append((idx + 1, txt))
    return pages


def remove_headers_footers(raw_pages: List[Tuple[int, str]]) -> List[Tuple[int, str, List[str]]]:
    """
    Detect repeated header/footer lines occurring across > 25% of pages and strip them.
    Also separates figure and table captions into metadata.
    """
    if not raw_pages:
        return []

    # Frequency analysis of top 3 lines and bottom 3 lines
    header_counts = {}
    footer_counts = {}
    page_count = len(raw_pages)

    for _, text in raw_pages:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            continue
        for line in lines[:3]:
            if len(line) < 100:
                header_counts[line] = header_counts.get(line, 0) + 1
        for line in lines[-3:]:
            if len(line) < 100:
                footer_counts[line] = footer_counts.get(line, 0) + 1

    threshold = max(2, int(page_count * 0.25))
    repeated_headers = {l for l, c in header_counts.items() if c >= threshold}
    repeated_footers = {l for l, c in footer_counts.items() if c >= threshold}

    cleaned_pages = []
    for page_num, text in raw_pages:
        lines = text.splitlines()
        clean_lines = []
        fig_tables = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                clean_lines.append("")
                continue

            # Strip recurring headers/footers or simple page number lines
            if stripped in repeated_headers or stripped in repeated_footers:
                continue
            if re.match(r'^\d+$', stripped) or re.match(r'^[A-Z0-9_\-]+-\d+$', stripped):
                continue

            # Separate figure/table/box captions
            if re.match(r'^(?:Figure|Table|Box|Fig\.)\s+\d+[\.:]', stripped, re.IGNORECASE):
                fig_tables.append(stripped)
                continue

            clean_lines.append(line)

        cleaned_text = "\n".join(clean_lines).strip()
        cleaned_pages.append((page_num, cleaned_text, fig_tables))

    return cleaned_pages


def tag_chunk(text: str) -> Tuple[List[str], List[str]]:
    """Assign topic tags and linked metrics based on scientific keywords."""
    lowered = text.lower()
    topics = set()
    metrics = set()

    # Soil
    if any(k in lowered for k in ["soil", "carbon", "humus", "microb", "organic carbon", "tillage", "nutrient"]):
        topics.add("soil")
        metrics.add("soil_organic_carbon")
    if "ph" in lowered or "acid" in lowered or "alkal" in lowered:
        topics.add("soil")

    # Biodiversity
    if any(k in lowered for k in ["biodiversity", "species", "extinction", "corridor", "pollinat", "fauna", "flora", "habitat", "ecosystem"]):
        topics.add("biodiversity")
        metrics.add("habitat_diversity")
        metrics.add("species_richness")

    # Climate
    if any(k in lowered for k in ["rainfall", "precipitation", "drought", "climate", "temperature", "warming", "flood", "evapotranspiration", "water"]):
        topics.add("climate")
        metrics.add("water_retention")

    # Land-use / Agriculture
    if any(k in lowered for k in ["monoculture", "crop", "yield", "agroforestry", "pasture", "grazing", "farming", "intercropping", "reforest"]):
        topics.add("land-use")
        metrics.add("crop_yield")
        metrics.add("vegetation_cover")

    # Human impact
    if any(k in lowered for k in ["degradation", "deforestation", "human", "driver", "emission", "overgrazing", "policy", "restoration"]):
        topics.add("human-impact")
        metrics.add("vegetation_cover")

    if not topics:
        topics.add("biodiversity")
    if not metrics:
        metrics.add("habitat_diversity")

    return list(topics), list(metrics)


def chunk_page_text(
    page_text: str,
    page_num: int,
    doc_prefix: str,
    doc_title: str,
    source_file: str,
    fig_tables: List[str],
    min_words: int = 150,
    max_words: int = 380,
) -> List[Dict]:
    """
    Split clean page text into 300-500 token chunks respecting paragraph boundaries.
    """
    if not page_text or len(page_text.split()) < 30:
        return []

    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', page_text) if p.strip()]
    if not paragraphs:
        paragraphs = [page_text]

    chunks_data = []
    current_words = []
    chunk_idx = 1

    for para in paragraphs:
        words = para.split()
        if len(current_words) + len(words) > max_words and len(current_words) >= min_words:
            chunk_text = " ".join(current_words)
            chunk_id = f"{doc_prefix}_p{page_num}_c{chunk_idx}"
            topic_tags, linked_metrics = tag_chunk(chunk_text)

            contextual_header = f"From {doc_title}, page {page_num}: "
            contextual_text = contextual_header + chunk_text

            chunks_data.append({
                "id": chunk_id,
                "text": chunk_text,
                "document_title": doc_title,
                "page_number": page_num,
                "source_file": source_file,
                "source": f"{doc_title}, Page {page_num}",
                "topic_tags": topic_tags,
                "linked_metrics": linked_metrics,
                "figures_tables": fig_tables,
                "contextual_text": contextual_text
            })
            chunk_idx += 1
            current_words = list(words)
        else:
            current_words.extend(words)

    # Remaining tail words
    if current_words and len(current_words) >= 40:
        chunk_text = " ".join(current_words)
        chunk_id = f"{doc_prefix}_p{page_num}_c{chunk_idx}"
        topic_tags, linked_metrics = tag_chunk(chunk_text)
        contextual_header = f"From {doc_title}, page {page_num}: "
        contextual_text = contextual_header + chunk_text

        chunks_data.append({
            "id": chunk_id,
            "text": chunk_text,
            "document_title": doc_title,
            "page_number": page_num,
            "source_file": source_file,
            "source": f"{doc_title}, Page {page_num}",
            "topic_tags": topic_tags,
            "linked_metrics": linked_metrics,
            "figures_tables": fig_tables,
            "contextual_text": contextual_text
        })

    return chunks_data


def ingest_real_pdfs():
    """Main ingestion orchestrator for real PDFs in data/."""
    print("=" * 70)
    print("  Darukaa.Earth — Production Real PDF Ingestion Pipeline")
    print("  Corpus: IPCC AR6 WGII Reports & Cross-Chapter Papers in /data")
    print("=" * 70)
    print()

    pdf_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))
    if not pdf_files:
        print(f"[!] No PDF files found in {DATA_DIR}")
        return

    total_pages = 0
    all_chunks = []
    doc_stats = {}

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        prefix, title = get_doc_metadata(filename)
        print(f"[*] Processing: {filename}...")

        raw_pages = extract_raw_pages(pdf_path)
        total_pages += len(raw_pages)
        cleaned_pages = remove_headers_footers(raw_pages)

        doc_chunks = []
        for page_num, clean_text, fig_tables in cleaned_pages:
            chunks = chunk_page_text(
                clean_text, page_num, prefix, title, filename, fig_tables
            )
            doc_chunks.extend(chunks)

        all_chunks.extend(doc_chunks)
        doc_stats[filename] = {
            "pages": len(raw_pages),
            "chunks": len(doc_chunks),
            "title": title
        }
        print(f"    -> Extracted {len(raw_pages)} pages, created {len(doc_chunks)} chunks.")

    print()
    print("-" * 70)
    print(f"Total Pages Processed: {total_pages}")
    print(f"Total Chunks Created:  {len(all_chunks)}")
    print("-" * 70)

    # Save to JSON for inspectability and backup
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    print(f"[+] Saved structured chunks to: {OUTPUT_JSON_PATH}")

    # Embed and index into ChromaDB + BM25
    print("\n[*] Embedding context-augmented chunks into ChromaDB & building BM25...")
    model = _get_model()
    collection = _get_collection()

    ids = [c["id"] for c in all_chunks]
    display_texts = [c["text"] for c in all_chunks]
    contextual_texts = [c["contextual_text"] for c in all_chunks]
    metadatas = [
        {
            "source": c["source"],
            "document_title": c["document_title"],
            "page_number": c["page_number"],
            "source_file": c["source_file"],
            "topic_tags": json.dumps(c["topic_tags"]),
            "linked_metrics": json.dumps(c["linked_metrics"]),
            "original_text": c["text"],
            "contextual_text": c["contextual_text"],
        }
        for c in all_chunks
    ]

    # Batch embedding (size 64)
    batch_size = 64
    all_embeddings = []
    print(f"    Encoding {len(contextual_texts)} chunks with MiniLM-L6-v2 in batches of {batch_size}...")
    for i in range(0, len(contextual_texts), batch_size):
        batch = contextual_texts[i:i + batch_size]
        emb = model.encode(batch, show_progress_bar=False).tolist()
        all_embeddings.extend(emb)
        print(f"    Embedded {min(i + batch_size, len(contextual_texts))}/{len(contextual_texts)} chunks...", end="\r")
    print(f"    Completed embedding {len(contextual_texts)} chunks successfully.    ")

    # Clean existing collection and upsert new
    print(f"[*] Upserting into ChromaDB collection '{COLLECTION_NAME}'...")
    try:
        # Delete old chunks if any
        existing = collection.get(include=[])
        if existing and existing.get("ids"):
            collection.delete(ids=existing["ids"])
    except Exception as e:
        print(f"    Note on collection clear: {e}")

    # Upsert in batches of 200
    for i in range(0, len(ids), 200):
        b_ids = ids[i:i + 200]
        b_docs = display_texts[i:i + 200]
        b_emb = all_embeddings[i:i + 200]
        b_meta = metadatas[i:i + 200]
        collection.upsert(
            ids=b_ids,
            documents=b_docs,
            embeddings=b_emb,
            metadatas=b_meta
        )

    # Build BM25 index over contextual texts
    print("[*] Building BM25 sparse keyword index...")
    _build_bm25_index(ids, contextual_texts, metadatas)

    # Final Summary Report
    print("\n" + "=" * 70)
    print("  INGESTION SUMMARY REPORT")
    print("=" * 70)
    print(f"  Total Documents:        {len(pdf_files)}")
    print(f"  Total Pages Processed:  {total_pages}")
    print(f"  Total Chunks Created:   {len(all_chunks)}")
    print(f"  ChromaDB Chunks Stored: {collection.count()}")
    print("\n  Chunks per Source Document:")
    for fname, info in doc_stats.items():
        print(f"    • {fname[:38]:<38} | {info['pages']:>3} pages | {info['chunks']:>4} chunks | {info['title'][:32]}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    ingest_real_pdfs()

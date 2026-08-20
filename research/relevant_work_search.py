#!/usr/bin/env python3
"""
Novelty claim search: nnU-Net for liver mass segmentation on ultrasound.

Searches PubMed, Arxiv, and Semantic Scholar for prior work matching the
novelty claim in Price-Gauger (2025): "no prior work has applied nnU-Net
to liver mass segmentation on B-mode ultrasound."

Outputs a timestamped CSV of all results, deduplicated by title similarity.

Usage:
    python novelty_search.py
    python novelty_search.py --output results/  # custom output directory

APIs used (no keys required):
    - PubMed E-utilities (NCBI)
    - Arxiv API
    - Semantic Scholar Academic Graph API

Rate limits are respected with conservative delays between requests.
"""

import argparse
import csv
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

QUERIES = [
    {
        "label": "nnunet_liver_ultrasound",
        "description": "nnU-Net AND liver AND ultrasound",
        "pubmed": '("nnU-Net"[All Fields] OR "nnUNet"[All Fields]) AND ("liver"[All Fields] OR "hepatic"[All Fields]) AND ("ultrasound"[All Fields] OR "ultrasonography"[All Fields] OR "B-mode"[All Fields])',
        "arxiv": '(all:"nnU-Net" OR all:"nnUNet") AND (all:liver OR all:hepatic) AND (all:ultrasound OR all:ultrasonography OR all:"B-mode")',
        "semantic_scholar": "nnU-Net liver ultrasound segmentation",
    },
    {
        "label": "nnunet_liver_ultrasound_broad",
        "description": "nnU-Net AND liver ultrasound (broader)",
        "pubmed": '("nnU-Net"[All Fields] OR "nnUNet"[All Fields]) AND "liver ultrasound"[All Fields]',
        "arxiv": '(all:"nnU-Net" OR all:"nnUNet") AND all:"liver ultrasound"',
        "semantic_scholar": "nnUNet liver ultrasound",
    },
    {
        "label": "data_efficiency_liver_ultrasound",
        "description": "data efficiency AND liver AND ultrasound segmentation",
        "pubmed": '("data efficiency"[All Fields] OR "learning curve"[All Fields] OR "sample efficiency"[All Fields]) AND ("liver"[All Fields]) AND ("ultrasound"[All Fields]) AND ("segmentation"[All Fields])',
        "arxiv": '(all:"data efficiency" OR all:"learning curve" OR all:"sample efficiency") AND all:liver AND all:ultrasound AND all:segmentation',
        "semantic_scholar": "data efficiency liver ultrasound segmentation",
    },
    {
        "label": "liver_ultrasound_segmentation_dl",
        "description": "liver ultrasound mass/tumor/lesion segmentation deep learning (broader landscape)",
        "pubmed": '("liver"[All Fields] OR "hepatic"[All Fields]) AND ("ultrasound"[All Fields] OR "B-mode"[All Fields]) AND ("segmentation"[All Fields]) AND ("mass"[All Fields] OR "tumor"[All Fields] OR "tumour"[All Fields] OR "lesion"[All Fields]) AND ("deep learning"[All Fields] OR "neural network"[All Fields])',
        "arxiv": '(all:liver OR all:hepatic) AND (all:ultrasound OR all:"B-mode") AND all:segmentation AND (all:mass OR all:tumor OR all:lesion) AND (all:"deep learning" OR all:"neural network")',
        "semantic_scholar": "liver ultrasound tumor lesion segmentation deep learning",
    },
]

TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
USER_AGENT = "NoveltySearch/1.0 (academic literature search; contact: github.com/roprice/liver-us-nnunet-baselines)"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(url: str, headers: dict | None = None) -> bytes:
    """GET with a user-agent and basic error handling."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()
    except Exception as e:
        print(f"  WARNING: request failed for {url[:120]}... -> {e}")
        return b""


def normalize_title(title: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    t = title.lower()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def title_hash(title: str) -> str:
    return hashlib.md5(normalize_title(title).encode()).hexdigest()


# ---------------------------------------------------------------------------
# PubMed
# ---------------------------------------------------------------------------

def search_pubmed(query: str, label: str) -> list[dict]:
    """Search PubMed via E-utilities and fetch article metadata."""
    print(f"  PubMed: searching '{label}'...")

    # Step 1: esearch to get PMIDs
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    search_url = (
        f"{base}esearch.fcgi?db=pubmed&retmode=json&retmax=200"
        f"&term={urllib.parse.quote(query)}"
    )
    data = _get(search_url)
    if not data:
        return []
    result = json.loads(data)
    id_list = result.get("esearchresult", {}).get("idlist", [])
    count = result.get("esearchresult", {}).get("count", "0")
    print(f"    Found {count} results, fetching {len(id_list)} records")

    if not id_list:
        return []

    time.sleep(0.5)  # rate limit

    # Step 2: efetch to get metadata
    ids = ",".join(id_list)
    fetch_url = f"{base}efetch.fcgi?db=pubmed&retmode=xml&id={ids}"
    xml_data = _get(fetch_url)
    if not xml_data:
        return []

    records = []
    root = ET.fromstring(xml_data)
    for article in root.findall(".//PubmedArticle"):
        title_el = article.find(".//ArticleTitle")
        abstract_els = article.findall(".//AbstractText")
        pmid_el = article.find(".//PMID")
        year_el = article.find(".//PubDate/Year")
        journal_el = article.find(".//Journal/Title")

        # Authors
        authors = []
        for author in article.findall(".//Author"):
            last = author.findtext("LastName", "")
            fore = author.findtext("ForeName", "")
            if last:
                authors.append(f"{last}, {fore}".strip(", "))

        # DOI
        doi = ""
        for id_el in article.findall(".//ArticleId"):
            if id_el.get("IdType") == "doi":
                doi = id_el.text or ""

        title = title_el.text if title_el is not None else ""
        records.append({
            "source": "PubMed",
            "query_label": label,
            "title": title,
            "authors": "; ".join(authors[:5]) + (" et al." if len(authors) > 5 else ""),
            "year": year_el.text if year_el is not None else "",
            "abstract": " ".join(
                (el.get("Label", "") + ": " if el.get("Label") else "") + (el.text or "")
                for el in abstract_els
            ) if abstract_els else "",
            "doi": doi,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid_el.text}/" if pmid_el is not None else "",
            "journal": journal_el.text if journal_el is not None else "",
        })

    time.sleep(0.5)
    return records


# ---------------------------------------------------------------------------
# Arxiv
# ---------------------------------------------------------------------------

def search_arxiv(query: str, label: str) -> list[dict]:
    """Search Arxiv API."""
    print(f"  Arxiv: searching '{label}'...")
    url = (
        f"http://export.arxiv.org/api/query?"
        f"search_query={urllib.parse.quote(query)}&start=0&max_results=200"
    )
    data = _get(url)
    if not data:
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(data)
    entries = root.findall("atom:entry", ns)
    print(f"    Found {len(entries)} results")

    records = []
    for entry in entries:
        title = entry.findtext("atom:title", "", ns).replace("\n", " ").strip()
        summary = entry.findtext("atom:summary", "", ns).replace("\n", " ").strip()
        published = entry.findtext("atom:published", "", ns)[:4]
        link = entry.findtext("atom:id", "", ns)

        authors = []
        for author in entry.findall("atom:author", ns):
            name = author.findtext("atom:name", "", ns)
            if name:
                authors.append(name)

        # Try to find DOI in links
        doi = ""
        for link_el in entry.findall("atom:link", ns):
            href = link_el.get("href", "")
            if "doi.org" in href:
                doi = href.replace("https://doi.org/", "").replace("http://doi.org/", "")

        records.append({
            "source": "Arxiv",
            "query_label": label,
            "title": title,
            "authors": "; ".join(authors[:5]) + (" et al." if len(authors) > 5 else ""),
            "year": published,
            "abstract": summary[:1000],
            "doi": doi,
            "url": link,
            "journal": "arXiv preprint",
        })

    time.sleep(3)  # arxiv asks for 3s between requests
    return records


# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------

def search_semantic_scholar(query: str, label: str, max_retries: int = 3) -> list[dict]:
    """Search Semantic Scholar Academic Graph API with retry on 429."""
    print(f"  Semantic Scholar: searching '{label}'...")
    fields = "title,authors,year,abstract,externalIds,venue"
    url = (
        f"https://api.semanticscholar.org/graph/v1/paper/search"
        f"?query={urllib.parse.quote(query)}&limit=100&fields={fields}"
    )
    data = b""
    for attempt in range(max_retries):
        data = _get(url)
        if data:
            break
        wait = 5 * (attempt + 1)
        print(f"    Retry {attempt + 1}/{max_retries} after {wait}s...")
        time.sleep(wait)
    if not data:
        print(f"    WARNING: Semantic Scholar failed after {max_retries} retries")
        return []

    result = json.loads(data)
    papers = result.get("data", [])
    total = result.get("total", 0)
    print(f"    Found {total} total, fetched {len(papers)} records")

    records = []
    for paper in papers:
        authors = paper.get("authors", [])
        author_names = [a.get("name", "") for a in authors[:5]]
        if len(authors) > 5:
            author_names.append("et al.")

        ext_ids = paper.get("externalIds", {}) or {}
        doi = ext_ids.get("DOI", "")
        arxiv_id = ext_ids.get("ArXiv", "")

        url_str = ""
        if doi:
            url_str = f"https://doi.org/{doi}"
        elif arxiv_id:
            url_str = f"https://arxiv.org/abs/{arxiv_id}"

        records.append({
            "source": "Semantic Scholar",
            "query_label": label,
            "title": paper.get("title", ""),
            "authors": "; ".join(author_names),
            "year": str(paper.get("year", "")),
            "abstract": (paper.get("abstract", "") or "")[:1000],
            "doi": doi,
            "url": url_str,
            "journal": paper.get("venue", "") or "",
        })

    time.sleep(1)
    return records


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate(records: list[dict]) -> list[dict]:
    """Deduplicate by normalized title, keeping the first occurrence."""
    seen = set()
    unique = []
    dupes = 0
    for r in records:
        h = title_hash(r["title"])
        if h not in seen:
            seen.add(h)
            unique.append(r)
        else:
            dupes += 1
    print(f"\n  Deduplication: {len(records)} total -> {len(unique)} unique ({dupes} duplicates removed)")
    return unique


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Search for prior work on nnU-Net liver ultrasound segmentation")
    parser.add_argument("--output", default=".", help="Output directory for CSV")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Novelty claim search - {TIMESTAMP}")
    print("=" * 60)

    all_records = []

    for q in QUERIES:
        print(f"\nQuery: {q['description']}")
        print("-" * 40)

        all_records.extend(search_pubmed(q["pubmed"], q["label"]))
        all_records.extend(search_arxiv(q["arxiv"], q["label"]))
        all_records.extend(search_semantic_scholar(q["semantic_scholar"], q["label"]))

    # Deduplicate
    unique_records = deduplicate(all_records)

    # Sort by year descending, then title
    unique_records.sort(key=lambda r: (-(int(r["year"]) if r["year"].isdigit() else 0), r["title"]))

    # Write CSV
    filename = f"novelty_search_{TIMESTAMP}.csv"
    filepath = output_dir / filename
    fieldnames = ["source", "query_label", "title", "authors", "year", "journal", "doi", "url", "abstract"]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unique_records)

    print(f"\n{'=' * 60}")
    print(f"Results written to: {filepath}")
    print(f"Total unique records: {len(unique_records)}")
    print(f"Search timestamp: {TIMESTAMP}")

    # Summary by source
    source_counts = {}
    for r in unique_records:
        source_counts[r["source"]] = source_counts.get(r["source"], 0) + 1
    print("\nBy source (after dedup):")
    for src, count in sorted(source_counts.items()):
        print(f"  {src}: {count}")

    # Summary by query
    query_counts = {}
    for r in unique_records:
        query_counts[r["query_label"]] = query_counts.get(r["query_label"], 0) + 1
    print("\nBy query (after dedup):")
    for ql, count in sorted(query_counts.items()):
        print(f"  {ql}: {count}")


if __name__ == "__main__":
    main()

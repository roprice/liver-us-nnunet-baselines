# Novelty claim validation

This folder contains the script and results used to validate the novelty claim that no prior work has applied nnU-Net to liver mass segmentation on diagnostic B-mode ultrasound, nor conducted a resource efficiency analysis for this task.

## Methodology

`relevant_work_search.py` queries three academic databases (PubMed, Arxiv, Semantic Scholar) with multiple query formulations, deduplicates results by title similarity, and exports a timestamped CSV for manual review.

No API keys are required. Rate limits are respected with conservative delays between requests.

## Search queries

Four query strategies are used, from narrow to broad:

**1. nnU-Net + liver + ultrasound (narrow)**
Targets direct prior work using the same framework and modality.
- PubMed: `("nnU-Net" OR "nnUNet") AND ("liver" OR "hepatic") AND ("ultrasound" OR "ultrasonography" OR "B-mode")`
- Arxiv: `(all:"nnU-Net" OR all:"nnUNet") AND (all:liver OR all:hepatic) AND (all:ultrasound OR all:ultrasonography OR all:"B-mode")`
- Semantic Scholar: `nnU-Net liver ultrasound segmentation`

**2. nnU-Net + liver ultrasound (broader phrasing)**
Catches results where "liver ultrasound" appears as a phrase.
- PubMed: `("nnU-Net" OR "nnUNet") AND "liver ultrasound"`
- Arxiv: `(all:"nnU-Net" OR all:"nnUNet") AND all:"liver ultrasound"`
- Semantic Scholar: `nnUNet liver ultrasound`

**3. Data/resource efficiency + liver + ultrasound segmentation**
Targets prior efficiency analyses on the same task, regardless of framework.
- PubMed: `("data efficiency" OR "learning curve" OR "sample efficiency") AND "liver" AND "ultrasound" AND "segmentation"`
- Arxiv: `(all:"data efficiency" OR all:"learning curve" OR all:"sample efficiency") AND all:liver AND all:ultrasound AND all:segmentation`
- Semantic Scholar: `data efficiency liver ultrasound segmentation`

**4. Liver ultrasound mass/tumor segmentation + deep learning (landscape)**
Broad survey of the surrounding literature to identify any work we may have missed.
- PubMed: `("liver" OR "hepatic") AND ("ultrasound" OR "B-mode") AND "segmentation" AND ("mass" OR "tumor" OR "tumour" OR "lesion") AND ("deep learning" OR "neural network")`
- Arxiv: `(all:liver OR all:hepatic) AND (all:ultrasound OR all:"B-mode") AND all:segmentation AND (all:mass OR all:tumor OR all:lesion) AND (all:"deep learning" OR all:"neural network")`
- Semantic Scholar: `liver ultrasound tumor lesion segmentation deep learning`

## Usage

```bash
python relevant_work_search.py
python relevant_work_search.py --output results/  # custom output directory
```

## Output

A timestamped CSV file (e.g., `relevant_work_search_20260820_162851_UTC.csv`) containing deduplicated results with: source database, query label, title, authors, year, journal/venue, DOI, URL, and abstract excerpt. Results are sorted by year descending.

Manual review of this CSV is required to assess relevance. The script casts a wide net; most results will be tangentially related rather than directly competing.

## Reproducing

Run the script at any time to check for new publications that may affect the novelty claim. Compare the new CSV against previous results to identify new entries.

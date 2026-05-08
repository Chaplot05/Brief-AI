"""
=============================================================
PDF_LOADER.PY — Download and Extract Text from Real PDFs
=============================================================

WHY REAL PDFs?
    Our seed data was written by us — that's fine for testing
    but NOT impressive for a portfolio project. Real PDFs from
    NASSCOM, IBEF, RBI, and government reports show that our
    system handles REAL, messy, unstructured data.

WHAT THIS DOES:
    1. Downloads PDF reports from the internet
    2. Extracts text from each page using PyMuPDF (fitz)
    3. Saves as JSON files in data/raw/ (same format as seed data)

WHY PyMuPDF (fitz) OVER PyPDF2?
    - PyMuPDF handles complex layouts better (tables, columns)
    - Faster text extraction
    - Better Unicode support (important for Indian language text)
    - PyPDF2 often returns garbled text from complex PDFs
=============================================================
"""

import sys
import json
import hashlib
import requests
import fitz  # PyMuPDF
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import RAW_DATA_DIR


# Real, freely available PDFs about Indian startups/economy
PDF_SOURCES = [
    {
        "url": "https://www.ibef.org/download/1729074075_Startups-October-2024.pdf",
        "title": "IBEF India Startups Report October 2024",
        "source": "IBEF"
    },
    {
        "url": "https://www.ibef.org/download/1721631498_E-Commerce-July-2024.pdf",
        "title": "IBEF E-Commerce Industry Report July 2024",
        "source": "IBEF"
    },
    {
        "url": "https://www.ibef.org/download/1726546536_IT-and-BPM-September-2024.pdf",
        "title": "IBEF IT and BPM Industry Report September 2024",
        "source": "IBEF"
    },
    {
        "url": "https://www.ibef.org/download/1726546583_Banking-September-2024.pdf",
        "title": "IBEF Banking Industry Report September 2024",
        "source": "IBEF"
    },
    {
        "url": "https://www.ibef.org/download/1722227403_Financial-Services-July-2024.pdf",
        "title": "IBEF Financial Services Report July 2024",
        "source": "IBEF"
    },
    {
        "url": "https://www.startupindia.gov.in/content/dam/invest-india/Templates/public/198702702.pdf",
        "title": "Startup India Scheme Overview",
        "source": "DPIIT"
    },
]


def download_pdf(url: str, save_dir: Path) -> Path | None:
    """Download a PDF from a URL and save locally."""
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = hashlib.md5(url.encode()).hexdigest()[:12] + ".pdf"
    filepath = save_dir / filename

    if filepath.exists():
        print(f"  Already downloaded: {filepath.name}")
        return filepath

    try:
        print(f"  Downloading: {url[:70]}...")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        filepath.write_bytes(response.content)
        print(f"  Saved: {filepath.name} ({len(response.content) // 1024} KB)")
        return filepath
    except Exception as e:
        print(f"  Failed: {e}")
        return None


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text from a PDF using PyMuPDF.

    WHY PAGE-BY-PAGE?
        PDFs don't have a single "text" field. They're made of pages,
        and each page has text positioned at specific coordinates.
        PyMuPDF reads each page and concatenates the text.
    """
    try:
        doc = fitz.open(str(pdf_path))
        text_parts = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                text_parts.append(text.strip())
        doc.close()
        full_text = "\n\n".join(text_parts)
        return full_text
    except Exception as e:
        print(f"  Error extracting text: {e}")
        return ""


def ingest_pdfs():
    """Download PDFs and convert to JSON articles."""
    pdf_dir = RAW_DATA_DIR.parent / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Downloading and processing real PDF reports...")
    print("=" * 60)

    articles_created = 0

    for source in PDF_SOURCES:
        print(f"\n--- {source['title']} ---")

        # Step 1: Download
        pdf_path = download_pdf(source["url"], pdf_dir)
        if not pdf_path:
            continue

        # Step 2: Extract text
        text = extract_text_from_pdf(pdf_path)
        if len(text) < 500:
            print(f"  Skipped: too little text extracted ({len(text)} chars)")
            continue

        # Step 3: Save as JSON (same format as seed data)
        article_id = "pdf_" + hashlib.md5(source["url"].encode()).hexdigest()[:10]
        article = {
            "id": article_id,
            "title": source["title"],
            "url": source["url"],
            "text": text,
            "authors": [source["source"]],
            "publish_date": "2024-01-01",
            "source_type": "pdf"
        }

        output_path = RAW_DATA_DIR / f"{article_id}.json"
        output_path.write_text(
            json.dumps(article, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        articles_created += 1
        print(f"  Created: {output_path.name} ({len(text)} chars)")

    print(f"\nDone! Created {articles_created} articles from PDFs")
    print(f"Total articles in data/raw/: {len(list(RAW_DATA_DIR.glob('*.json')))}")


if __name__ == "__main__":
    ingest_pdfs()

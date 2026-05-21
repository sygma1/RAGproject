"""
build_index.py
--------------
Standalone script to (re)build the ChromaDB index from PDFs in data/.
Run this once after adding new documents so the next app launch is instant.

Usage:
    python build_index.py
    python build_index.py --force   # force full rebuild
"""

import argparse
from pathlib import Path

from rag_pipeline import get_vectorstore, DATA_DIR, CHROMA_INDEX_DIR

def main():
    parser = argparse.ArgumentParser(description="Build ChromaDB index for TelecomGPT")
    parser.add_argument("--force", action="store_true", help="Force full rebuild")
    args = parser.parse_args()

    print("=" * 55)
    print("  TelecomGPT — Index Builder")
    print("=" * 55)

    pdf_count = len(list(DATA_DIR.glob("*.pdf")))
    print(f"\nPDFs found in '{DATA_DIR}/': {pdf_count}")
    if pdf_count == 0:
        print("\n❌  No PDFs found. Add spec files to data/ first.")
        return

    vs = get_vectorstore(force_rebuild=args.force)
    n  = len(vs._collection.get()["documents"])
    print(f"\n✅  Index ready  —  {n} vectors stored in '{CHROMA_INDEX_DIR}/'")
    print("\nYou can now run the app with:  streamlit run app.py")

if __name__ == "__main__":
    main()

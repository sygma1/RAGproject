import os
import re
from pathlib import Path
from typing import Optional, List

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer, CrossEncoder
import shutil

# ──────────────────────────────────────────────
# Paths & constants
# ──────────────────────────────────────────────
CHROMA_INDEX_DIR = str(Path("chroma_db").resolve())
DATA_DIR         = Path("data")

EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL       = "llama-3.3-70b-versatile"

CHUNK_SIZE       = 800
CHUNK_OVERLAP    = 150
TOP_K_RETRIEVE   = 8
TOP_K_RERANK     = 3


# ──────────────────────────────────────────────
# Unicode sanitizer
# ──────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Remove surrogate characters and other non-encodable Unicode."""
    # Encode to UTF-8 ignoring surrogates, decode back to str
    return text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")


# ──────────────────────────────────────────────
# Safe embedding wrapper
# ──────────────────────────────────────────────
class SafeHFEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self._dim  = len(self.model.encode(["test"])[0].tolist())

    def _safe_encode_one(self, text: str) -> List[float]:
        try:
            text = clean_text(str(text).strip())
            if not text:
                return [0.0] * self._dim
            return self.model.encode([text])[0].tolist()
        except Exception as e:
            print(f"  ⚠  Encoding failed: {e}")
            return [0.0] * self._dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._safe_encode_one(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._safe_encode_one(text)


# Single shared instance
embeddings = SafeHFEmbeddings(EMBEDDING_MODEL)


# ──────────────────────────────────────────────
# Document loading & indexing
# ──────────────────────────────────────────────
def load_pdf_documents(data_dir: Path = DATA_DIR):
    pdf_paths = sorted(data_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDFs found in '{data_dir}/'. "
            "Add 3GPP / ITU spec PDFs and restart."
        )
    documents = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        pages  = loader.load()
        for page in pages:
            page.metadata["source"] = pdf_path.name
        documents.extend(pages)
        print(f"  ✓ Loaded {len(pages):>4} pages  ←  {pdf_path.name}")
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"  ✓ {len(chunks)} chunks created (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def build_vectorstore(chunks, save=True):
    clean_chunks, skipped = [], 0
    for c in chunks:
        content = c.page_content
        # Fix bytes
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="ignore")
        # Must be a string with real content
        if not isinstance(content, str) or len(content.strip()) < 10:
            skipped += 1
            continue
        # Strip surrogates and other bad Unicode
        content = clean_text(content.strip())
        if len(content) < 10:
            skipped += 1
            continue
        c.page_content = content
        clean_chunks.append(c)

    print(f"  ✓ {len(clean_chunks)} valid chunks (dropped {skipped} empty/bad)")

    texts     = [doc.page_content for doc in clean_chunks]
    metadatas = [doc.metadata for doc in clean_chunks]

    print(f"  ⚙  Computing embeddings — this may take a minute …")
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name="document_collection",
        persist_directory=CHROMA_INDEX_DIR,
    )
    return vectorstore


def load_vectorstore():
    vectorstore = Chroma(
        persist_directory=CHROMA_INDEX_DIR,
        collection_name="document_collection",
        embedding_function=embeddings,
    )
    print(f"  ✓ ChromaDB collection loaded from '{CHROMA_INDEX_DIR}/'")
    return vectorstore


def get_vectorstore(force_rebuild: bool = False):
    index_path = Path(CHROMA_INDEX_DIR)
    if force_rebuild and index_path.exists():
        print("\n[Index] Clearing existing ChromaDB index …")
        shutil.rmtree(index_path)
    if not force_rebuild and index_path.exists():
        print("\n[Index] Existing ChromaDB index found — loading …")
        return load_vectorstore()
    print("\n[Index] Building ChromaDB index from PDFs …")
    documents = load_pdf_documents()
    chunks    = split_documents(documents)
    return build_vectorstore(chunks, save=True)


# ──────────────────────────────────────────────
# Re-ranking
# ──────────────────────────────────────────────
_reranker: Optional[CrossEncoder] = None

def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        print("  ⚙  Loading cross-encoder re-ranker …")
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank(query: str, docs: list, top_k: int = TOP_K_RERANK) -> list:
    reranker = get_reranker()
    pairs    = [(query, doc.page_content) for doc in docs]
    scores   = reranker.predict(pairs)
    ranked   = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]


# ──────────────────────────────────────────────
# Prompt template
# ──────────────────────────────────────────────
SYSTEM_PROMPT = PromptTemplate.from_template("""
You are TelecomGPT, an expert technical assistant specialised in mobile \
telecommunications standards (3GPP, ITU-T, ETSI) and 5G/LTE architecture.

RULES:
- Answer ONLY from the provided context excerpts.
- If the answer is not present, say clearly:
  "I could not find this information in the loaded documents."
- Be precise, structured, and technical.
- Use bullet points or numbered lists where appropriate.
- Always end with a "Sources:" line listing the document(s) used.

---
CONTEXT:
{context}

---
CONVERSATION HISTORY:
{history}

---
QUESTION:
{question}
""")


def format_context(docs: list) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "?")
        page   = (page + 1) if isinstance(page, int) else page
        parts.append(
            f"[Excerpt {i} | {source} | p.{page}]\n{doc.page_content}"
        )
    return "\n\n".join(parts)


def format_sources(docs: list) -> str:
    seen, items = set(), []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "?")
        page   = (page + 1) if isinstance(page, int) else page
        key    = f"{source} (p.{page})"
        if key not in seen:
            seen.add(key)
            items.append(key)
    return " | ".join(items)


# ──────────────────────────────────────────────
# Answer generation
# ──────────────────────────────────────────────
def answer_question(
    question:  str,
    retriever,
    llm:       ChatGroq,
    history:   str = "",
) -> tuple[str, list, list]:
    all_docs     = retriever.invoke(question)
    best_docs    = rerank(question, all_docs, top_k=TOP_K_RERANK)
    context      = format_context(best_docs)
    sources      = format_sources(best_docs)
    final_prompt = SYSTEM_PROMPT.format(
        context=context,
        history=history or "None",
        question=question,
    )
    response = llm.invoke(final_prompt).content
    if "Sources:" not in response:
        response = response.strip() + f"\n\nSources: {sources}"
    return response, best_docs, all_docs


# ──────────────────────────────────────────────
# LLM factory
# ──────────────────────────────────────────────
def build_llm(api_key: str) -> ChatGroq:
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        api_key=api_key,
    )


# ──────────────────────────────────────────────
# Entrypoint (CLI fallback)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY missing from .env")

    vs        = get_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": TOP_K_RETRIEVE})
    llm       = build_llm(api_key)
    history   = ""

    print("\n🛰  TelecomGPT ready. Type 'quit' to exit.\n")
    while True:
        q = input("Question > ").strip()
        if not q:
            continue
        if q.lower() in {"quit", "exit", "q"}:
            break
        answer, docs, _ = answer_question(q, retriever, llm, history)
        print(f"\n{answer}\n")
        history += f"\nUser: {q}\nAssistant: {answer}\n"
        history  = "\n".join(history.split("\n")[-24:])


import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
DATA_DIR        = Path("data")
FAISS_INDEX_DIR = Path("faiss_index")

EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL       = "llama-3.3-70b-versatile"

CHUNK_SIZE       = 800
CHUNK_OVERLAP    = 150
TOP_K_RETRIEVE   = 8   # initial retrieval pool
TOP_K_RERANK     = 3   # kept after re-ranking


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


def build_vectorstore(chunks, save: bool = True):
    print("  ⚙  Computing embeddings — this may take a minute …")
    embeddings   = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore  = FAISS.from_documents(chunks, embeddings)
    if save:
        FAISS_INDEX_DIR.mkdir(exist_ok=True)
        vectorstore.save_local(str(FAISS_INDEX_DIR))
        print(f"  ✓ FAISS index saved to '{FAISS_INDEX_DIR}/'")
    return vectorstore


def load_vectorstore():
    embeddings  = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        str(FAISS_INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"  ✓ FAISS index loaded from '{FAISS_INDEX_DIR}/'")
    return vectorstore


def get_vectorstore(force_rebuild: bool = False):
    if not force_rebuild and FAISS_INDEX_DIR.exists():
        print("\n[Index] Existing FAISS index found — loading …")
        return load_vectorstore()
    print("\n[Index] Building FAISS index from PDFs …")
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
    """
    Full RAG pipeline step:
      1. Retrieve top-K candidates
      2. Re-rank with CrossEncoder
      3. Build prompt and call Groq
    Returns: (answer, reranked_docs, all_retrieved_docs)
    """
    # 1 – retrieve
    all_docs     = retriever.invoke(question)

    # 2 – re-rank
    best_docs    = rerank(question, all_docs, top_k=TOP_K_RERANK)

    # 3 – build prompt
    context      = format_context(best_docs)
    sources      = format_sources(best_docs)
    final_prompt = SYSTEM_PROMPT.format(
        context=context,
        history=history or "None",
        question=question,
    )

    # 4 – generate
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
        # append turn to history (last 3 turns kept)
        history += f"\nUser: {q}\nAssistant: {answer}\n"
        history  = "\n".join(history.split("\n")[-24:])   # ~3 turns

# 📡 TelecomGPT — 5G & LTE Technical Assistant

> **Module IA Générative — Projet RAG Amélioré (Option 2)**
> Système de Question-Réponse sur les standards télécoms 3GPP / ITU

---

## Table of Contents

1. [Project Description](#1-project-description)
2. [Architecture](#2-architecture)
3. [Technical Choices](#3-technical-choices)
4. [Project Structure](#4-project-structure)
5. [Installation](#5-installation)
6. [Usage](#6-usage)
7. [RAG Pipeline — Step by Step](#7-rag-pipeline--step-by-step)
8. [Advanced Features (Option 2)](#8-advanced-features-option-2)
9. [Suggested Documents](#9-suggested-documents)
10. [Results & Limits](#10-results--limits)

---

## 1. Project Description

**TelecomGPT** is an intelligent Q&A assistant built on the RAG
(Retrieval-Augmented Generation) architecture. It allows users to ask technical
questions about 5G NR, LTE, network architecture, and 3GPP specifications, and
receive answers grounded exclusively in the loaded document corpus.

**Use cases:**
- Understanding 5G architecture concepts (AMF, SMF, UPF, gNB…)
- Comparing LTE and 5G NR radio parameters
- Querying 3GPP release notes
- Learning about OFDM numerology, beamforming, network slicing
- Quick lookup across multiple spec PDFs without reading them manually

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INDEXING PHASE (once)                     │
│                                                             │
│  PDF files  →  PyPDFLoader  →  Chunker  →  Embeddings       │
│               (3GPP specs)     800 chars    MiniLM-L6        │
│                                              ↓               │
│                                         FAISS index          │
│                                         (saved to disk)      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    QUERY PHASE (per question)                │
│                                                             │
│  User question                                              │
│       ↓                                                     │
│  Embed query  →  FAISS similarity search  →  Top-8 chunks   │
│                                                ↓            │
│                                      CrossEncoder re-rank    │
│                                        → Top-3 chunks        │
│                                                ↓            │
│                                  PromptTemplate (context +  │
│                                  history + question)         │
│                                                ↓            │
│                                     Groq LLaMA-3.3-70B       │
│                                                ↓            │
│                                       Final answer           │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Technical Choices

| Component | Choice | Reason |
|---|---|---|
| **LLM** | Groq `llama-3.3-70b-versatile` | Free API, very fast inference, strong technical reasoning |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Good balance of speed and quality, runs locally |
| **Vector Store** | FAISS (CPU) | Fast similarity search, persistent index, no server needed |
| **Re-ranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Improves precision by scoring (query, passage) pairs jointly |
| **PDF Loading** | `PyPDFLoader` (LangChain) | Handles multi-page specs, preserves page metadata |
| **Chunking** | `RecursiveCharacterTextSplitter` | Respects paragraph boundaries, configurable overlap |
| **UI** | Streamlit | Rapid development, chat interface, sidebar controls |
| **Memory** | Sliding window (last 3 turns) | Keeps conversation context without exceeding context limits |

**Why re-ranking?** FAISS retrieves by cosine similarity of dense embeddings, which
is fast but can miss nuanced relevance. The CrossEncoder scores each (query, passage)
pair jointly, significantly improving precision at the cost of slightly more compute.

---

## 4. Project Structure

```
telecom_rag/
│
├── app.py              # Streamlit UI (main entry point)
├── rag_pipeline.py     # Core RAG logic (indexing, retrieval, re-rank, generation)
├── build_index.py      # Standalone index builder script
│
├── data/               # ← Place your PDF files here
│   └── (your 3GPP / ITU spec PDFs)
│
├── faiss_index/        # Auto-created after first build
│   ├── index.faiss
│   └── index.pkl
│
├── requirements.txt
├── .env.example        # Copy to .env and add your Groq key
├── .gitignore
└── README.md
```

---

## 5. Installation

### Prerequisites
- Python 3.10 or 3.11
- A free Groq API key ([console.groq.com](https://console.groq.com/keys))
- PDF files of telecom specs in `data/`

### Steps

```bash
# 1. Clone / download the project
git clone https://github.com/your-username/telecom-rag.git
cd telecom-rag

# 2. Create a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt


# 4. Add PDFs (see section 9 for links)
mkdir data
# Copy your 3GPP / ITU PDFs into data/
```

---

## 6. Usage

### Option A — Streamlit app (recommended)

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

1. Enter your Groq API key in the sidebar
2. Click **Build / Load Index**
3. Start asking questions!

### Option B — Pre-build the index (faster startup)

```bash
python build_index.py         # build once
python build_index.py --force # force full rebuild after adding new PDFs
streamlit run app.py          # index loads instantly
```

### Option C — CLI mode (no UI)

```bash
python rag_pipeline.py
```

---

## 7. RAG Pipeline — Step by Step

### Phase 1: Indexing (`build_index.py` or first run of `app.py`)

1. **Load** — `PyPDFLoader` reads every PDF in `data/`, extracting text page by page
   with source filename and page number in metadata.

2. **Chunk** — `RecursiveCharacterTextSplitter` splits pages into overlapping chunks
   (800 chars, 150 overlap). Overlap prevents answers from being cut mid-sentence.

3. **Embed** — `HuggingFaceEmbeddings` (MiniLM-L6) converts each chunk into a
   384-dimensional dense vector.

4. **Index** — All vectors are stored in a FAISS index saved to `faiss_index/`
   for reuse across sessions.

### Phase 2: Question Answering

1. **Embed query** — The user's question is embedded with the same model.

2. **ANN search** — FAISS finds the 8 most similar chunks using cosine similarity.

3. **Re-rank** — A CrossEncoder scores each (question, chunk) pair jointly and
   keeps the top 3. This is the key improvement over basic RAG.

4. **Prompt** — A structured `PromptTemplate` injects the 3 chunks as context,
   plus the conversation history (last 3 turns) and the question.

5. **Generate** — Groq's LLaMA-3.3-70B produces a grounded answer with source
   citations.

---

## 8. Advanced Features (Option 2)

| Feature | Implementation |
|---|---|
| **Re-ranking** | `CrossEncoder` (ms-marco-MiniLM-L-6-v2) applied after initial retrieval |
| **Conversation memory** | Sliding window of last 3 turns injected into every prompt |
| **Source transparency** | Every answer ends with `Sources: filename (p.N)` |
| **Chunk inspection** | Expandable panel in UI shows all retrieved chunks with source and page |
| **Index persistence** | FAISS index saved to disk — no re-embedding on restart |
| **Force rebuild** | Checkbox in sidebar to trigger full re-indexing after adding new PDFs |
| **Configurable top-K** | Separate `TOP_K_RETRIEVE` (8) and `TOP_K_RERANK` (3) parameters |

---

## 9. Suggested Documents

All 3GPP specs are free. Download PDFs from [3gpp.org/specifications](https://www.3gpp.org/specifications)
by searching the spec number.

| Spec | Title | Why useful |
|---|---|---|
| TS 38.300 | NR; NR and NG-RAN Overall Description | Core 5G NR radio overview |
| TS 23.501 | System Architecture for 5G System | 5G core network (AMF, SMF, UPF…) |
| TS 36.300 | LTE; Overall description | LTE/E-UTRAN overview |
| TS 38.211 | NR; Physical channels and modulation | OFDM, waveforms, numerology |
| TR 21.905 | Vocabulary for 3GPP Specifications | Definitions and abbreviations |
| TR 38.801 | Study on New Radio Access Technology | 5G design study |
| ITU-T Y.3101 | Requirements for IMT-2020 networks | ITU 5G requirements |

> **Tip:** Start with TR 21.905 (vocabulary) + TS 23.501 (architecture) + TS 38.300 (radio).
> That gives broad coverage for most common questions.

---

## 10. Results & Limits

### What works well
- Questions directly answered in the loaded specs (definitions, procedures, parameters)
- Multi-document queries that need synthesis across specs
- Follow-up questions using conversation history
- Source attribution — answers clearly cite the relevant document and page

### Known limits
- **Coverage** — The system only knows what is in the loaded PDFs. Missing specs = missing answers.
- **Table parsing** — Complex tables in 3GPP PDFs are sometimes extracted poorly by PyPDFLoader.
- **Very long answers** — Groq's `max_tokens` may truncate very detailed responses.
- **Hallucination risk** — Although the prompt instructs the model to stay grounded, LLMs can occasionally generate plausible-sounding but incorrect technical details. Always verify critical figures against the source spec.

### Possible improvements
- Add `Unstructured` loader for better table extraction from PDFs
- Implement hybrid search (BM25 + dense retrieval) for better keyword precision
- Add a feedback button to flag incorrect answers
- Extend to Agentic RAG (Option 3) with a tool that can search 3GPP online

---

*Project developed as part of the IA Générative module — Option 2: RAG Amélioré*

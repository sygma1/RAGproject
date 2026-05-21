"""
app.py
------
Streamlit UI for the Telecom & 5G Technical Assistant (RAG Amélioré).

Run:
    streamlit run app.py
"""

import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import (
    TOP_K_RETRIEVE,
    answer_question,
    build_llm,
    get_vectorstore,
)

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="TelecomGPT — 5G & LTE Assistant",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
.stApp { background-color: #0d1117; color: #e6edf3; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}

/* Chat messages */
.user-bubble {
    background: #1f6feb;
    color: white;
    padding: 10px 15px;
    border-radius: 12px 12px 2px 12px;
    margin: 8px 0 8px 20%;
    text-align: right;
    line-height: 1.5;
}
.bot-bubble {
    background: #161b22;
    color: #e6edf3;
    padding: 12px 16px;
    border-radius: 2px 12px 12px 12px;
    margin: 8px 20% 8px 0;
    border-left: 3px solid #238636;
    line-height: 1.6;
}
.source-tag {
    font-size: 0.75rem;
    color: #8b949e;
    margin-top: 6px;
    font-style: italic;
}
.chunk-card {
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.82rem;
    color: #c9d1d9;
}
.metric-box {
    background: #1c2128;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
    border: 1px solid #30363d;
}
h1, h2, h3 { color: #58a6ff !important; }
.stButton button {
    background-color: #238636;
    color: white;
    border: none;
    border-radius: 6px;
    font-weight: 600;
}
.stButton button:hover { background-color: #2ea043; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Session state initialisation
# ──────────────────────────────────────────────
def init_state():
    defaults = {
        "messages":    [],        # [{role, content, sources, chunks}]
        "history_str": "",        # plain-text history fed to the LLM
        "vectorstore": None,
        "retriever":   None,
        "llm":         None,
        "index_ready": False,
        "index_stats": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 TelecomGPT")
    st.markdown("*5G & LTE Technical Assistant*")
    st.divider()

    # API key
    load_dotenv()
    default_key = os.getenv("GROQ_API_KEY", "")
    api_key = st.text_input(
        "🔑 Groq API Key",
        value=default_key,
        type="password",
        help="Get a free key at console.groq.com",
    )

    st.divider()

    # Index management
    st.markdown("### 📂 Document Index")
    data_dir = Path("data")
    pdf_list = sorted(data_dir.glob("*.pdf")) if data_dir.exists() else []

    if pdf_list:
        st.success(f"{len(pdf_list)} PDF(s) found in `data/`")
        with st.expander("View files"):
            for p in pdf_list:
                st.markdown(f"- `{p.name}`")
    else:
        st.warning("No PDFs in `data/` folder yet.")
        st.markdown(
            "Download 3GPP / ITU specs and place them in the `data/` folder, "
            "then click **Build Index**."
        )

    force_rebuild = st.checkbox("Force rebuild index", value=False)

    if st.button("⚡ Build / Load Index", disabled=not api_key):
        if not pdf_list:
            st.error("Add PDF files to `data/` first.")
        else:
            with st.spinner("Building ChromaDB index …"):
                try:
                    vs = get_vectorstore(force_rebuild=force_rebuild)
                    st.session_state.vectorstore = vs
                    st.session_state.retriever   = vs.as_retriever(
                        search_kwargs={"k": TOP_K_RETRIEVE}
                    )
                    st.session_state.llm         = build_llm(api_key)
                    st.session_state.index_ready = True

                    # crude stats
                    n_docs = len(vs._collection.get()["documents"])
                    st.session_state.index_stats = {
                        "vectors": n_docs,
                        "pdfs":    len(pdf_list),
                    }
                    st.success("Index ready!")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()

    # Clear chat
    if st.button("🗑 Clear conversation"):
        st.session_state.messages    = []
        st.session_state.history_str = ""
        st.rerun()

    st.divider()
    st.markdown("### 🤖 Agent")
    agent_query = st.text_input("Agent command / question", key="agent_query_input")
    if st.button("Run Agent", key="run_agent", disabled=not api_key):
        # ensure index & llm are loaded
        if not st.session_state.vectorstore:
            with st.spinner("Loading index and LLM…"):
                try:
                    vs = get_vectorstore(force_rebuild=force_rebuild)
                    st.session_state.vectorstore = vs
                    st.session_state.retriever   = vs.as_retriever(search_kwargs={"k": TOP_K_RETRIEVE})
                    st.session_state.llm         = build_llm(api_key)
                    st.session_state.index_ready = True
                except Exception as e:
                    st.error(f"Error loading index: {e}")
        # run agent query
        if st.session_state.vectorstore:
            q = (agent_query or "").strip()
            if not q:
                st.warning("Enter a question for the agent first.")
            else:
                with st.spinner("Agent running…"):
                    try:
                        answer, best_docs, all_docs = answer_question(
                            question  = q,
                            retriever = st.session_state.retriever,
                            llm       = st.session_state.llm,
                            history   = st.session_state.history_str,
                        )
                        # update history and messages
                        st.session_state.history_str += f"\nUser: {q}\nAssistant: {answer}\n"
                        st.session_state.messages.append({"role": "user", "content": q})
                        st.session_state.messages.append({"role": "assistant", "content": answer, "chunks": best_docs})
                        st.success("Agent completed — answer appended to chat.")
                    except Exception as e:
                        st.error(f"Agent error: {e}")

    st.divider()
    st.markdown(
        "<small>Stack: LangChain · ChromaDB · CrossEncoder · Groq LLaMA-3</small>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# Main area — header
# ──────────────────────────────────────────────
st.markdown("# 📡 TelecomGPT — 5G & LTE Technical Assistant")
st.markdown(
    "Ask technical questions about **5G NR**, **LTE**, **network slicing**, "
    "**OFDM**, **3GPP releases**, **core network architecture**, and more."
)

if not st.session_state.index_ready:
    st.info(
        "👈 **Getting started:** Add your telecom PDF documents to the `data/` folder, "
        "enter your Groq API key in the sidebar, then click **Build / Load Index**."
    )
    st.markdown("""
### Suggested documents to download
| Document | Source |
|---|---|
| 3GPP TS 38.300 — NR Overall description | [3gpp.org](https://www.3gpp.org/specifications) |
| 3GPP TS 23.501 — 5GS architecture | [3gpp.org](https://www.3gpp.org/specifications) |
| 3GPP TS 36.300 — LTE Overall description | [3gpp.org](https://www.3gpp.org/specifications) |
| ITU-T Y.3101 — IMT-2020 requirements | [itu.int](https://www.itu.int/rec/T-REC-Y/en) |
| 3GPP TR 21.905 — Vocabulary | [3gpp.org](https://www.3gpp.org/specifications) |

> All 3GPP specs are free to download. Search the spec number at **3gpp.org → Specifications**.
""")
    st.stop()


# ──────────────────────────────────────────────
# Suggested questions
# ──────────────────────────────────────────────
SUGGESTIONS = [
    # "What is the difference between 5G NR and LTE?",
    # "Explain network slicing in 5G.",
    # "What is the role of the AMF in 5G core network?",
    # "How does OFDM work in 5G NR?",
    # "What are the frequency bands used in 5G?",
    # "What is the difference between NSA and SA deployment?",
]

# st.markdown("**💡 Try asking:**")
cols = st.columns(3)
for i, suggestion in enumerate(SUGGESTIONS):
    if cols[i % 3].button(suggestion, key=f"sug_{i}"):
        st.session_state["pending_question"] = suggestion


# ──────────────────────────────────────────────
# Chat history display
# ──────────────────────────────────────────────
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f"<div class='user-bubble'>🧑 {msg['content']}</div>",
                unsafe_allow_html=True,
            )
        else:
            # Format the answer — replace newlines with <br> for HTML
            content_html = msg["content"].replace("\n", "<br>")
            st.markdown(
                f"<div class='bot-bubble'>🤖 {content_html}</div>",
                unsafe_allow_html=True,
            )
            # Retrieved chunks expander
            if msg.get("chunks"):
                with st.expander(f"🔍 View {len(msg['chunks'])} retrieved chunks"):
                    for i, chunk in enumerate(msg["chunks"], 1):
                        src  = chunk.metadata.get("source", "?")
                        page = chunk.metadata.get("page", "?")
                        page = (page + 1) if isinstance(page, int) else page
                        preview = chunk.page_content[:300].replace("\n", " ")
                        st.markdown(
                            f"<div class='chunk-card'>"
                            f"<b>Chunk {i}</b> — {src} | p.{page}<br/>{preview}…"
                            f"</div>",
                            unsafe_allow_html=True,
                        )


# ──────────────────────────────────────────────
# Input
# ──────────────────────────────────────────────
st.divider()
col_input, col_btn = st.columns([5, 1])

with col_input:
    question = st.text_input(
        "Ask a technical question …",
        value=st.session_state.pop("pending_question", ""),
        placeholder="e.g. What is the 5G NR numerology?",
        label_visibility="collapsed",
    )

with col_btn:
    send = st.button("Send ➤", use_container_width=True)


# ──────────────────────────────────────────────
# Query handling
# ──────────────────────────────────────────────
if send and question.strip():
    q = question.strip()

    # Add user message
    st.session_state.messages.append({"role": "user", "content": q})

    with st.spinner("🔍 Retrieving & re-ranking … ⚙ Generating answer …"):
        t0 = time.time()
        try:
            answer, best_docs, all_docs = answer_question(
                question  = q,
                retriever = st.session_state.retriever,
                llm       = st.session_state.llm,
                history   = st.session_state.history_str,
            )
            elapsed = time.time() - t0

            # Update conversation history (keep last ~3 turns)
            st.session_state.history_str += f"\nUser: {q}\nAssistant: {answer}\n"
            lines = st.session_state.history_str.split("\n")
            st.session_state.history_str = "\n".join(lines[-24:])

            # Store message
            st.session_state.messages.append({
                "role":    "assistant",
                "content": answer,
                "chunks":  best_docs,
                "elapsed": elapsed,
            })

        except Exception as e:
            st.session_state.messages.append({
                "role":    "assistant",
                "content": f"⚠️ Error: {e}",
                "chunks":  [],
            })

    st.rerun()

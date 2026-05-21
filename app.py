# """
# app.py
# ------
# Streamlit UI for the Telecom & 5G Technical Assistant (RAG Amélioré).

# Run:
#     streamlit run app.py
# """

# import os
# import time
# from pathlib import Path

# import streamlit as st
# from dotenv import load_dotenv

# from rag_pipeline import (
#     TOP_K_RETRIEVE,
#     answer_question,
#     build_llm,
#     get_vectorstore,
# )

# # ──────────────────────────────────────────────
# # Page config
# # ──────────────────────────────────────────────
# st.set_page_config(
#     page_title="TelecomGPT — 5G & LTE Assistant",
#     page_icon="📡",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # ──────────────────────────────────────────────
# # CSS
# # ──────────────────────────────────────────────
# st.markdown("""
# <style>
# /* Main background */
# .stApp { background-color: #0d1117; color: #e6edf3; }

# /* Sidebar */
# section[data-testid="stSidebar"] {
#     background-color: #161b22;
#     border-right: 1px solid #30363d;
# }

# /* Chat messages */
# .user-bubble {
#     background: #1f6feb;
#     color: white;
#     padding: 10px 15px;
#     border-radius: 12px 12px 2px 12px;
#     margin: 8px 0 8px 20%;
#     text-align: right;
#     line-height: 1.5;
# }
# .bot-bubble {
#     background: #161b22;
#     color: #e6edf3;
#     padding: 12px 16px;
#     border-radius: 2px 12px 12px 12px;
#     margin: 8px 20% 8px 0;
#     border-left: 3px solid #238636;
#     line-height: 1.6;
# }
# .source-tag {
#     font-size: 0.75rem;
#     color: #8b949e;
#     margin-top: 6px;
#     font-style: italic;
# }
# .chunk-card {
#     background: #1c2128;
#     border: 1px solid #30363d;
#     border-radius: 8px;
#     padding: 10px 14px;
#     margin: 6px 0;
#     font-size: 0.82rem;
#     color: #c9d1d9;
# }
# .metric-box {
#     background: #1c2128;
#     border-radius: 8px;
#     padding: 12px;
#     text-align: center;
#     border: 1px solid #30363d;
# }
# h1, h2, h3 { color: #58a6ff !important; }
# .stButton button {
#     background-color: #238636;
#     color: white;
#     border: none;
#     border-radius: 6px;
#     font-weight: 600;
# }
# .stButton button:hover { background-color: #2ea043; }
# </style>
# """, unsafe_allow_html=True)


# # ──────────────────────────────────────────────
# # Session state initialisation
# # ──────────────────────────────────────────────
# def init_state():
#     defaults = {
#         "messages":    [],        # [{role, content, sources, chunks}]
#         "history_str": "",        # plain-text history fed to the LLM
#         "vectorstore": None,
#         "retriever":   None,
#         "llm":         None,
#         "index_ready": False,
#         "index_stats": {},
#     }
#     for k, v in defaults.items():
#         if k not in st.session_state:
#             st.session_state[k] = v

# init_state()


# # ──────────────────────────────────────────────
# # Sidebar
# # ──────────────────────────────────────────────
# with st.sidebar:
#     st.markdown("## 📡 TelecomGPT")
#     st.markdown("*5G & LTE Technical Assistant*")
#     st.divider()

#     # API key
#     load_dotenv()
#     default_key = os.getenv("GROQ_API_KEY", "")
#     api_key = st.text_input(
#         "🔑 Groq API Key",
#         value=default_key,
#         type="password",
#         help="Get a free key at console.groq.com",
#     )

#     st.divider()

#     # Index management
#     st.markdown("### 📂 Document Index")
#     data_dir = Path("data")
#     pdf_list = sorted(data_dir.glob("*.pdf")) if data_dir.exists() else []

#     if pdf_list:
#         st.success(f"{len(pdf_list)} PDF(s) found in `data/`")
#         with st.expander("View files"):
#             for p in pdf_list:
#                 st.markdown(f"- `{p.name}`")
#     else:
#         st.warning("No PDFs in `data/` folder yet.")
#         st.markdown(
#             "Download 3GPP / ITU specs and place them in the `data/` folder, "
#             "then click **Build Index**."
#         )

#     force_rebuild = st.checkbox("Force rebuild index", value=False)

#     if st.button("⚡ Build / Load Index", disabled=not api_key):
#         if not pdf_list:
#             st.error("Add PDF files to `data/` first.")
#         else:
#             with st.spinner("Building ChromaDB index …"):
#                 try:
#                     vs = get_vectorstore(force_rebuild=force_rebuild)
#                     st.session_state.vectorstore = vs
#                     st.session_state.retriever   = vs.as_retriever(
#                         search_kwargs={"k": TOP_K_RETRIEVE}
#                     )
#                     st.session_state.llm         = build_llm(api_key)
#                     st.session_state.index_ready = True

#                     # crude stats
#                     n_docs = len(vs._collection.get()["documents"])
#                     st.session_state.index_stats = {
#                         "vectors": n_docs,
#                         "pdfs":    len(pdf_list),
#                     }
#                     st.success("Index ready!")
#                 except Exception as e:
#                     st.error(f"Error: {e}")

#     st.divider()

#     # Clear chat
#     if st.button("🗑 Clear conversation"):
#         st.session_state.messages    = []
#         st.session_state.history_str = ""
#         st.rerun()

#     st.divider()
#     st.markdown("### 🤖 Agent")
#     agent_query = st.text_input("Agent command / question", key="agent_query_input")
#     if st.button("Run Agent", key="run_agent", disabled=not api_key):
#         # ensure index & llm are loaded
#         if not st.session_state.vectorstore:
#             with st.spinner("Loading index and LLM…"):
#                 try:
#                     vs = get_vectorstore(force_rebuild=force_rebuild)
#                     st.session_state.vectorstore = vs
#                     st.session_state.retriever   = vs.as_retriever(search_kwargs={"k": TOP_K_RETRIEVE})
#                     st.session_state.llm         = build_llm(api_key)
#                     st.session_state.index_ready = True
#                 except Exception as e:
#                     st.error(f"Error loading index: {e}")
#         # run agent query
#         if st.session_state.vectorstore:
#             q = (agent_query or "").strip()
#             if not q:
#                 st.warning("Enter a question for the agent first.")
#             else:
#                 with st.spinner("Agent running…"):
#                     try:
#                         answer, best_docs, all_docs = answer_question(
#                             question  = q,
#                             retriever = st.session_state.retriever,
#                             llm       = st.session_state.llm,
#                             history   = st.session_state.history_str,
#                         )
#                         # update history and messages
#                         st.session_state.history_str += f"\nUser: {q}\nAssistant: {answer}\n"
#                         st.session_state.messages.append({"role": "user", "content": q})
#                         st.session_state.messages.append({"role": "assistant", "content": answer, "chunks": best_docs})
#                         st.success("Agent completed — answer appended to chat.")
#                     except Exception as e:
#                         st.error(f"Agent error: {e}")

#     st.divider()
#     st.markdown(
#         "<small>Stack: LangChain · ChromaDB · CrossEncoder · Groq LLaMA-3</small>",
#         unsafe_allow_html=True,
#     )


# # ──────────────────────────────────────────────
# # Main area — header
# # ──────────────────────────────────────────────
# st.markdown("# 📡 TelecomGPT — 5G & LTE Technical Assistant")
# st.markdown(
#     "Ask technical questions about **5G NR**, **LTE**, **network slicing**, "
#     "**OFDM**, **3GPP releases**, **core network architecture**, and more."
# )

# if not st.session_state.index_ready:
#     st.info(
#         "👈 **Getting started:** Add your telecom PDF documents to the `data/` folder, "
#         "enter your Groq API key in the sidebar, then click **Build / Load Index**."
#     )
#     st.markdown("""
# ### Suggested documents to download
# | Document | Source |
# |---|---|
# | 3GPP TS 38.300 — NR Overall description | [3gpp.org](https://www.3gpp.org/specifications) |
# | 3GPP TS 23.501 — 5GS architecture | [3gpp.org](https://www.3gpp.org/specifications) |
# | 3GPP TS 36.300 — LTE Overall description | [3gpp.org](https://www.3gpp.org/specifications) |
# | ITU-T Y.3101 — IMT-2020 requirements | [itu.int](https://www.itu.int/rec/T-REC-Y/en) |
# | 3GPP TR 21.905 — Vocabulary | [3gpp.org](https://www.3gpp.org/specifications) |

# > All 3GPP specs are free to download. Search the spec number at **3gpp.org → Specifications**.
# """)
#     st.stop()


# # ──────────────────────────────────────────────
# # Suggested questions
# # ──────────────────────────────────────────────
# SUGGESTIONS = [
#     # "What is the difference between 5G NR and LTE?",
#     # "Explain network slicing in 5G.",
#     # "What is the role of the AMF in 5G core network?",
#     # "How does OFDM work in 5G NR?",
#     # "What are the frequency bands used in 5G?",
#     # "What is the difference between NSA and SA deployment?",
# ]

# # st.markdown("**💡 Try asking:**")
# cols = st.columns(3)
# for i, suggestion in enumerate(SUGGESTIONS):
#     if cols[i % 3].button(suggestion, key=f"sug_{i}"):
#         st.session_state["pending_question"] = suggestion


# # ──────────────────────────────────────────────
# # Chat history display
# # ──────────────────────────────────────────────
# chat_container = st.container()
# with chat_container:
#     for msg in st.session_state.messages:
#         if msg["role"] == "user":
#             st.markdown(
#                 f"<div class='user-bubble'>🧑 {msg['content']}</div>",
#                 unsafe_allow_html=True,
#             )
#         else:
#             # Format the answer — replace newlines with <br> for HTML
#             content_html = msg["content"].replace("\n", "<br>")
#             st.markdown(
#                 f"<div class='bot-bubble'>🤖 {content_html}</div>",
#                 unsafe_allow_html=True,
#             )
#             # Retrieved chunks expander
#             if msg.get("chunks"):
#                 with st.expander(f"🔍 View {len(msg['chunks'])} retrieved chunks"):
#                     for i, chunk in enumerate(msg["chunks"], 1):
#                         src  = chunk.metadata.get("source", "?")
#                         page = chunk.metadata.get("page", "?")
#                         page = (page + 1) if isinstance(page, int) else page
#                         preview = chunk.page_content[:300].replace("\n", " ")
#                         st.markdown(
#                             f"<div class='chunk-card'>"
#                             f"<b>Chunk {i}</b> — {src} | p.{page}<br/>{preview}…"
#                             f"</div>",
#                             unsafe_allow_html=True,
#                         )


# # ──────────────────────────────────────────────
# # Input
# # ──────────────────────────────────────────────
# st.divider()
# col_input, col_btn = st.columns([5, 1])

# with col_input:
#     question = st.text_input(
#         "Ask a technical question …",
#         value=st.session_state.pop("pending_question", ""),
#         placeholder="e.g. What is the 5G NR numerology?",
#         label_visibility="collapsed",
#     )

# with col_btn:
#     send = st.button("Send ➤", use_container_width=True)


# # ──────────────────────────────────────────────
# # Query handling
# # ──────────────────────────────────────────────
# if send and question.strip():
#     q = question.strip()

#     # Add user message
#     st.session_state.messages.append({"role": "user", "content": q})

#     with st.spinner("🔍 Retrieving & re-ranking … ⚙ Generating answer …"):
#         t0 = time.time()
#         try:
#             answer, best_docs, all_docs = answer_question(
#                 question  = q,
#                 retriever = st.session_state.retriever,
#                 llm       = st.session_state.llm,
#                 history   = st.session_state.history_str,
#             )
#             elapsed = time.time() - t0

#             # Update conversation history (keep last ~3 turns)
#             st.session_state.history_str += f"\nUser: {q}\nAssistant: {answer}\n"
#             lines = st.session_state.history_str.split("\n")
#             st.session_state.history_str = "\n".join(lines[-24:])

#             # Store message
#             st.session_state.messages.append({
#                 "role":    "assistant",
#                 "content": answer,
#                 "chunks":  best_docs,
#                 "elapsed": elapsed,
#             })

#         except Exception as e:
#             st.session_state.messages.append({
#                 "role":    "assistant",
#                 "content": f"⚠️ Error: {e}",
#                 "chunks":  [],
#             })

#     st.rerun()
"""
app.py
------
Streamlit UI for the Telecom & 5G Technical Assistant (RAG + MCP Agent integrated).

Run:
    streamlit run app.py
"""

import os
import re
import time
import asyncio
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from mcp import ClientSession
from mcp.client.sse import sse_client

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
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

.stApp { background-color: #0d1117; color: #e6edf3; font-family: 'Rajdhani', sans-serif; }

section[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}

.user-bubble {
    background: #1f6feb;
    color: white;
    padding: 10px 15px;
    border-radius: 12px 12px 2px 12px;
    margin: 8px 0 8px 20%;
    text-align: right;
    line-height: 1.5;
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
}
.bot-bubble {
    background: #161b22;
    color: #e6edf3;
    padding: 12px 16px;
    border-radius: 2px 12px 12px 12px;
    margin: 8px 20% 8px 0;
    border-left: 3px solid #238636;
    line-height: 1.6;
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
}
.agent-bubble {
    background: #1a1a2e;
    color: #e6edf3;
    padding: 12px 16px;
    border-radius: 2px 12px 12px 12px;
    margin: 8px 20% 8px 0;
    border-left: 3px solid #7c3aed;
    line-height: 1.6;
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
}
.agent-label {
    font-size: 0.72rem;
    color: #7c3aed;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.rag-label {
    font-size: 0.72rem;
    color: #238636;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 4px;
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
.status-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.badge-rag   { background: #0f2d16; color: #3fb950; border: 1px solid #238636; }
.badge-agent { background: #1e0a3c; color: #a78bfa; border: 1px solid #7c3aed; }
.badge-saved { background: #0c2233; color: #58a6ff; border: 1px solid #1f6feb; }

h1, h2, h3 { color: #58a6ff !important; font-family: 'Rajdhani', sans-serif !important; font-weight: 700 !important; }

.stButton button {
    background-color: #238636;
    color: white;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    font-family: 'Rajdhani', sans-serif;
}
.stButton button:hover { background-color: #2ea043; }

.mcp-status-ok  { color: #3fb950; font-weight: 700; }
.mcp-status-off { color: #f85149; font-weight: 700; }

.plan-block {
    background: #111827;
    border: 1px solid #374151;
    border-radius: 6px;
    padding: 10px 14px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
    color: #9ca3af;
    margin: 6px 0;
    white-space: pre-wrap;
}
.reflection-block {
    background: #111827;
    border: 1px solid #374151;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 0.85rem;
    color: #d1d5db;
    margin: 6px 0;
}
.tip-box {
    background: #161b22;
    border: 1px dashed #30363d;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 0.82rem;
    color: #8b949e;
    margin: 4px 0 8px 0;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────
def init_state():
    defaults = {
        "messages":    [],
        "history_str": "",
        "vectorstore": None,
        "retriever":   None,
        "llm":         None,
        "index_ready": False,
        "index_stats": {},
        "mcp_url":     "http://localhost:8000/sse",
        "mcp_online":  False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ──────────────────────────────────────────────
# Save intent parser
# ──────────────────────────────────────────────
def parse_save_intent(prompt: str) -> tuple[str, str | None]:
    """
    Detects if the user wants to save the answer to a file.
    Returns (cleaned_question, filename_or_None).

    Examples that trigger saving:
      "What is 5G beamforming? Save to beamforming.txt"
      "Explain NSA vs SA and write the answer to deployment.txt"
      "What is the AMF role? Save to file"   → defaults to answers.txt
    """
    # Pattern 1: explicit filename e.g. "save to notes.txt"
    match = re.search(
        r'(?:save|write|store|export)'
        r'(?:\s+(?:it|the answer|this|result|to a file))?\s+'
        r'(?:to|in|into|as)\s+([\w\-]+\.txt)',
        prompt, re.IGNORECASE
    )
    if match:
        filename = match.group(1)
        cleaned  = re.sub(re.escape(match.group(0)), "", prompt, flags=re.IGNORECASE).strip(" ,.;")
        return cleaned, filename

    # Pattern 2: "save to file" / "write to file" with no filename → default
    match2 = re.search(
        r'(?:save|write|store|export)\s+(?:it\s+)?(?:to\s+)?(?:a\s+)?(?:text\s+)?file',
        prompt, re.IGNORECASE
    )
    if match2:
        cleaned = re.sub(re.escape(match2.group(0)), "", prompt, flags=re.IGNORECASE).strip(" ,.;")
        return cleaned, "answers.txt"

    return prompt, None


# ──────────────────────────────────────────────
# MCP helpers
# ──────────────────────────────────────────────
async def _check_mcp(url: str) -> bool:
    try:
        async with sse_client(url) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                return True
    except Exception:
        return False


async def _mcp_write(url: str, filename: str, content: str, mode: str = "a") -> str:
    async with sse_client(url) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            res = await s.call_tool("write_answer", {
                "filename": filename,
                "content":  content,
                "mode":     mode,
            })
            return res.content[0].text


async def _mcp_full_agent(url: str, question: str, llm) -> dict:
    """
    Full planner → retrieve → generate → reflect pipeline via MCP.
    """
    from langchain_core.prompts import ChatPromptTemplate

    PLANNER_PROMPT = ChatPromptTemplate.from_template(
        "You are TelecomGPT autonomous planner.\n"
        "Question:\n{question}\n"
        "Give a step-by-step plan."
    )
    GENERATION_PROMPT = ChatPromptTemplate.from_template(
        "You are TelecomGPT.\n"
        "Answer ONLY using the provided telecom context.\n\n"
        "CONTEXT:\n{context}\n\n"
        "QUESTION:\n{question}\n\n"
        "RULES:\n- technical\n- structured\n- telecom standards if possible"
    )
    REFLECTION_PROMPT = ChatPromptTemplate.from_template(
        "You are a telecom QA reviewer.\n"
        "Question:\n{question}\n"
        "Answer:\n{answer}\n"
        'Return ONLY "APPROVED" if correct, otherwise explain issues.'
    )

    async with sse_client(url) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()

            # 1 — Plan
            plan = llm.invoke(
                PLANNER_PROMPT.format_messages(question=question)
            ).content

            # 2 — Retrieve via MCP
            res     = await s.call_tool("search_documents", {"query": question})
            context = res.content[0].text

            # 3 — Generate
            answer = llm.invoke(
                GENERATION_PROMPT.format_messages(context=context, question=question)
            ).content

            # 4 — Reflect (retry once if not approved)
            reflection = ""
            for _ in range(2):
                reflection = llm.invoke(
                    REFLECTION_PROMPT.format_messages(question=question, answer=answer)
                ).content
                if "APPROVED" in reflection.upper():
                    break
                answer = llm.invoke(
                    GENERATION_PROMPT.format_messages(context=context, question=question)
                ).content

            return {
                "plan":       plan,
                "context":    context,
                "answer":     answer,
                "reflection": reflection,
            }


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 TelecomGPT")
    st.markdown("*5G & LTE Technical Assistant*")
    st.divider()

    load_dotenv()
    default_key = os.getenv("GROQ_API_KEY", "")
    api_key = st.text_input(
        "🔑 Groq API Key",
        value=default_key,
        type="password",
        help="Get a free key at console.groq.com",
    )

    st.divider()

    # ── MCP Server config ──
    st.markdown("### 🔌 MCP Agent Server")
    mcp_url = st.text_input(
        "Server URL",
        value=st.session_state.mcp_url,
        help="URL of your running mcp_server.py (SSE transport)",
    )
    st.session_state.mcp_url = mcp_url

    col_ping, col_status = st.columns([1, 2])
    with col_ping:
        if st.button("Ping", use_container_width=True):
            with st.spinner("Checking…"):
                ok = asyncio.run(_check_mcp(mcp_url))
                st.session_state.mcp_online = ok
    with col_status:
        if st.session_state.mcp_online:
            st.markdown("<span class='mcp-status-ok'>● ONLINE</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='mcp-status-off'>● OFFLINE</span>", unsafe_allow_html=True)

    st.caption("Start the server first:\n`python mcp_server.py`")

    st.divider()

    # ── Index management ──
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
                    n_docs = len(vs._collection.get()["documents"])
                    st.session_state.index_stats = {
                        "vectors": n_docs,
                        "pdfs":    len(pdf_list),
                    }
                    st.success("Index ready!")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()

    if st.button("🗑 Clear conversation"):
        st.session_state.messages    = []
        st.session_state.history_str = ""
        st.rerun()

    st.divider()
    st.markdown(
        "<small>Stack: LangChain · ChromaDB · CrossEncoder · Groq LLaMA-3 · MCP</small>",
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
### Suggested documents
| Document | Source |
|---|---|
| 3GPP TS 38.300 — NR Overall description | [3gpp.org](https://www.3gpp.org/specifications) |
| 3GPP TS 23.501 — 5GS architecture | [3gpp.org](https://www.3gpp.org/specifications) |
| 3GPP TS 36.300 — LTE Overall description | [3gpp.org](https://www.3gpp.org/specifications) |
| ITU-T Y.3101 — IMT-2020 requirements | [itu.int](https://www.itu.int/rec/T-REC-Y/en) |
""")
    st.stop()


# ──────────────────────────────────────────────
# Chat history display
# ──────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"<div class='user-bubble'>🧑 {msg['content']}</div>",
            unsafe_allow_html=True,
        )
    elif msg["role"] == "assistant":
        mode         = msg.get("mode", "rag")
        bubble_class = "agent-bubble" if mode == "agent" else "bot-bubble"
        label_class  = "agent-label"  if mode == "agent" else "rag-label"
        label_text   = "🤖 MCP Agent"  if mode == "agent" else "🤖 RAG"

        content_html = msg["content"].replace("\n", "<br>")
        st.markdown(
            f"<div class='{bubble_class}'>"
            f"<div class='{label_class}'>{label_text}</div>"
            f"{content_html}"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Agent: plan + reflection expander
        if mode == "agent" and msg.get("plan"):
            with st.expander("📋 View agent plan & reflection"):
                st.markdown("**Plan**")
                st.markdown(
                    f"<div class='plan-block'>{msg['plan']}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("**Reflection**")
                st.markdown(
                    f"<div class='reflection-block'>{msg.get('reflection', '—')}</div>",
                    unsafe_allow_html=True,
                )

        # RAG: retrieved chunks expander
        if msg.get("chunks"):
            with st.expander(f"🔍 View {len(msg['chunks'])} retrieved chunks"):
                for i, chunk in enumerate(msg["chunks"], 1):
                    src     = chunk.metadata.get("source", "?")
                    page    = chunk.metadata.get("page", "?")
                    page    = (page + 1) if isinstance(page, int) else page
                    preview = chunk.page_content[:300].replace("\n", " ")
                    st.markdown(
                        f"<div class='chunk-card'>"
                        f"<b>Chunk {i}</b> — {src} | p.{page}<br/>{preview}…"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        # Saved file badge
        if msg.get("saved_to"):
            st.markdown(
                f"<span class='status-badge badge-saved'>💾 Saved → {msg['saved_to']}</span>",
                unsafe_allow_html=True,
            )


# ──────────────────────────────────────────────
# Input area
# ──────────────────────────────────────────────
st.divider()

mode_col, _ = st.columns([2, 3])
with mode_col:
    query_mode = st.radio(
        "Mode",
        ["💬 RAG (direct)", "🤖 MCP Agent"],
        horizontal=True,
    )

use_agent = query_mode == "🤖 MCP Agent"

if use_agent:
    st.markdown(
        "<div class='tip-box'>"
        "💡 <b>Tip:</b> include a save instruction in your prompt to automatically write the answer to a file.<br>"
        "Examples: &nbsp;"
        "<code>… save to 5g_notes.txt</code> &nbsp;·&nbsp; "
        "<code>… write the answer to lte.txt</code> &nbsp;·&nbsp; "
        "<code>… save to file</code> (uses <code>answers.txt</code>)"
        "</div>",
        unsafe_allow_html=True,
    )

col_input, col_btn = st.columns([5, 1])
with col_input:
    question = st.text_input(
        "Ask a technical question …",
        value=st.session_state.pop("pending_question", ""),
        placeholder=(
            "e.g. What is 5G beamforming? Save to beamforming.txt"
            if use_agent else
            "e.g. What is the 5G NR numerology?"
        ),
        label_visibility="collapsed",
    )
with col_btn:
    send = st.button("Send ➤", use_container_width=True)


# ──────────────────────────────────────────────
# Query handling
# ──────────────────────────────────────────────
if send and question.strip():
    original_q = question.strip()

    # Parse save intent from prompt (agent mode only)
    if use_agent:
        clean_q, save_filename = parse_save_intent(original_q)
    else:
        clean_q, save_filename = original_q, None

    # Always show the original prompt in the chat
    st.session_state.messages.append({"role": "user", "content": original_q})

    # ── MCP Agent mode ──
    if use_agent:
        if not st.session_state.mcp_online:
            ok = asyncio.run(_check_mcp(st.session_state.mcp_url))
            st.session_state.mcp_online = ok

        if not st.session_state.mcp_online:
            st.session_state.messages.append({
                "role":       "assistant",
                "mode":       "agent",
                "content":    "⚠️ MCP server is offline. Start it with `python mcp_server.py` then click **Ping** in the sidebar.",
                "plan":       None,
                "reflection": None,
            })
        else:
            if not st.session_state.llm:
                st.session_state.llm = build_llm(api_key)

            with st.spinner("🤖 Agent: planning → retrieving → generating → reflecting …"):
                try:
                    result = asyncio.run(
                        _mcp_full_agent(
                            st.session_state.mcp_url,
                            clean_q,
                            st.session_state.llm,
                        )
                    )

                    saved_to = None
                    if save_filename:
                        entry = (
                            f"Q: {clean_q}\n\n"
                            f"A: {result['answer']}\n\n"
                            f"{'=' * 60}\n"
                        )
                        asyncio.run(
                            _mcp_write(st.session_state.mcp_url, save_filename, entry)
                        )
                        saved_to = save_filename

                    st.session_state.history_str += f"\nUser: {clean_q}\nAssistant: {result['answer']}\n"

                    st.session_state.messages.append({
                        "role":       "assistant",
                        "mode":       "agent",
                        "content":    result["answer"],
                        "plan":       result["plan"],
                        "reflection": result["reflection"],
                        "saved_to":   saved_to,
                    })

                except Exception as e:
                    st.session_state.messages.append({
                        "role":       "assistant",
                        "mode":       "agent",
                        "content":    f"⚠️ Agent error: {e}",
                        "plan":       None,
                        "reflection": None,
                    })

    # ── RAG direct mode ──
    else:
        with st.spinner("🔍 Retrieving & re-ranking … ⚙ Generating answer …"):
            t0 = time.time()
            try:
                answer, best_docs, all_docs = answer_question(
                    question  = clean_q,
                    retriever = st.session_state.retriever,
                    llm       = st.session_state.llm,
                    history   = st.session_state.history_str,
                )
                elapsed = time.time() - t0

                st.session_state.history_str += f"\nUser: {clean_q}\nAssistant: {answer}\n"
                lines = st.session_state.history_str.split("\n")
                st.session_state.history_str = "\n".join(lines[-24:])

                st.session_state.messages.append({
                    "role":    "assistant",
                    "mode":    "rag",
                    "content": answer,
                    "chunks":  best_docs,
                    "elapsed": elapsed,
                })

            except Exception as e:
                st.session_state.messages.append({
                    "role":    "assistant",
                    "mode":    "rag",
                    "content": f"⚠️ Error: {e}",
                    "chunks":  [],
                })

    st.rerun()
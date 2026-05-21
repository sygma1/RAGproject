# import argparse
# import os
# from typing import TypedDict

# from dotenv import load_dotenv

# from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.memory import MemorySaver

# from langchain.tools import tool
# from langchain_core.prompts import ChatPromptTemplate

# from rag_pipeline import (
#     get_vectorstore,
#     build_llm,
#     rerank,
#     format_context,
#     format_sources,
#     TOP_K_RETRIEVE,
# )




# retriever = None
# llm = None




# @tool
# def search_documents(query: str) -> str:
#     """
#     Search telecom documents using RAG retrieval.
#     """

#     docs = retriever.invoke(query)

#     best_docs = rerank(query, docs)

#     return format_context(best_docs)


# @tool
# def summarize_text(text: str) -> str:
#     """
#     Summarize telecom technical text.
#     """

#     prompt = f"""
#     Summarize this telecom information:

#     {text}
#     """

#     return llm.invoke(prompt).content


# TOOLS = [
#     search_documents,
#     summarize_text,
# ]




# class AgentState(TypedDict):

#     question: str

#     plan: str

#     context: str

#     answer: str

#     reflection: str

#     approved: bool




# PLANNER_PROMPT = ChatPromptTemplate.from_template("""
# You are TelecomGPT autonomous planner.

# You can:
# - search telecom documents
# - summarize telecom information

# Question:
# {question}

# Explain step-by-step what should be done.
# """)


# def planner_node(state: AgentState):

#     prompt = PLANNER_PROMPT.format_messages(
#         question=state["question"]
#     )

#     response = llm.invoke(prompt).content

#     return {
#         "plan": response
#     }




# def retrieve_node(state: AgentState):

#     context = search_documents.invoke(
#         state["question"]
#     )

#     return {
#         "context": context
#     }




# GENERATION_PROMPT = ChatPromptTemplate.from_template("""
# You are TelecomGPT.

# Answer ONLY using the provided telecom context.

# CONTEXT:
# {context}

# QUESTION:
# {question}

# RULES:
# - Be technical
# - Use bullet points
# - Mention standards
# - Mention architecture details
# - Say clearly if information is missing
# """)


# def generate_node(state: AgentState):

#     prompt = GENERATION_PROMPT.format_messages(
#         context=state["context"],
#         question=state["question"]
#     )

#     response = llm.invoke(prompt).content

#     return {
#         "answer": response
#     }




# REFLECTION_PROMPT = ChatPromptTemplate.from_template("""
# You are a telecom QA reviewer.

# Question:
# {question}

# Answer:
# {answer}

# Check:
# - hallucinations
# - telecom accuracy
# - missing context
# - technical correctness

# If answer is good:
# return ONLY:
# APPROVED

# Otherwise explain problems.
# """)


# def reflection_node(state: AgentState):

#     prompt = REFLECTION_PROMPT.format_messages(
#         question=state["question"],
#         answer=state["answer"]
#     )

#     reflection = llm.invoke(prompt).content

#     if "APPROVED" in reflection.upper():

#         return {
#             "reflection": reflection,
#             "approved": True
#         }

#     return {
#         "reflection": reflection,
#         "approved": False
#     }




# def reflection_router(state: AgentState):

#     if state["approved"]:
#         return END

#     return "generate"




# def build_agent():

#     graph = StateGraph(AgentState)

#     graph.add_node(
#         "planner",
#         planner_node
#     )

#     graph.add_node(
#         "retrieve",
#         retrieve_node
#     )

#     graph.add_node(
#         "generate",
#         generate_node
#     )

#     graph.add_node(
#         "reflection",
#         reflection_node
#     )

#     graph.set_entry_point("planner")

#     graph.add_edge(
#         "planner",
#         "retrieve"
#     )

#     graph.add_edge(
#         "retrieve",
#         "generate"
#     )

#     graph.add_edge(
#         "generate",
#         "reflection"
#     )

#     graph.add_conditional_edges(
#         "reflection",
#         reflection_router
#     )

#     memory = MemorySaver()

#     return graph.compile(
#         checkpointer=memory
#     )




# def main():

#     parser = argparse.ArgumentParser()

#     parser.add_argument(
#         "--force",
#         action="store_true"
#     )

#     args = parser.parse_args()

#     load_dotenv()

#     api_key = os.getenv("GROQ_API_KEY")

#     if not api_key:
#         raise EnvironmentError(
#             "GROQ_API_KEY missing from .env"
#         )

#     global retriever
#     global llm

#     vs = get_vectorstore(
#         force_rebuild=args.force
#     )

#     retriever = vs.as_retriever(
#         search_kwargs={
#             "k": TOP_K_RETRIEVE
#         }
#     )

#     llm = build_llm(api_key)

#     agent = build_agent()

#     print("\nTelecomGPT Autonomous Agent Ready\n")

#     while True:

#         q = input("Question > ").strip()

#         if not q:
#             continue

#         if q.lower() in {
#             "quit",
#             "exit",
#             "q"
#         }:
#             break

#         result = agent.invoke(
#             {
#                 "question": q
#             },
#             config={
#                 "configurable": {
#                     "thread_id": "telecom-session"
#                 }
#             }
#         )

#         print("\n====================")
#         print("PLAN")
#         print("====================")
#         print(result["plan"])

#         print("\n====================")
#         print("ANSWER")
#         print("====================")
#         print(result["answer"])

#         print("\n====================")
#         print("REFLECTION")
#         print("====================")
#         print(result["reflection"])


# if __name__ == "__main__":
#     main()

# import argparse
# import os
# import asyncio
# from typing import TypedDict

# from dotenv import load_dotenv

# from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.memory import MemorySaver
# from langchain_core.prompts import ChatPromptTemplate

# from mcp import ClientSession

# from rag_pipeline import build_llm


# # =========================
# # MCP CLIENT WRAPPER
# # =========================

# class MCPTools:
#     def __init__(self, url: str):
#         self.url = url
#         self.session: ClientSession | None = None

#     async def connect(self):
#         self.session = ClientSession(self.url)
#         await self.session.initialize()

#     async def search_documents(self, query: str) -> str:
#         res = await self.session.call_tool(
#             "search_documents",
#             {"query": query}
#         )
#         return res.content[0].text

#     async def summarize_text(self, text: str) -> str:
#         res = await self.session.call_tool(
#             "summarize_text",
#             {"text": text}
#         )
#         return res.content[0].text


# # =========================
# # STATE
# # =========================

# class AgentState(TypedDict):
#     question: str
#     plan: str
#     context: str
#     answer: str
#     reflection: str
#     approved: bool


# # =========================
# # PROMPTS
# # =========================

# PLANNER_PROMPT = ChatPromptTemplate.from_template("""
# You are TelecomGPT autonomous planner.

# Question:
# {question}

# Give a step-by-step plan.
# """)

# GENERATION_PROMPT = ChatPromptTemplate.from_template("""
# You are TelecomGPT.

# Answer ONLY using the provided telecom context.

# CONTEXT:
# {context}

# QUESTION:
# {question}

# RULES:
# - technical
# - structured
# - telecom standards if possible
# """)

# REFLECTION_PROMPT = ChatPromptTemplate.from_template("""
# You are a telecom QA reviewer.

# Question:
# {question}

# Answer:
# {answer}

# Return ONLY "APPROVED" if correct, otherwise explain issues.
# """)


# # =========================
# # GLOBALS
# # =========================

# llm = None
# mcp_tools: MCPTools = None


# # =========================
# # NODES
# # =========================

# def planner_node(state: AgentState):
#     prompt = PLANNER_PROMPT.format_messages(
#         question=state["question"]
#     )
#     return {"plan": llm.invoke(prompt).content}


# async def retrieve_node(state: AgentState):
#     context = await mcp_tools.search_documents(state["question"])
#     return {"context": context}


# def generate_node(state: AgentState):
#     prompt = GENERATION_PROMPT.format_messages(
#         context=state["context"],
#         question=state["question"]
#     )
#     return {"answer": llm.invoke(prompt).content}


# def reflection_node(state: AgentState):
#     prompt = REFLECTION_PROMPT.format_messages(
#         question=state["question"],
#         answer=state["answer"]
#     )

#     result = llm.invoke(prompt).content
#     approved = "APPROVED" in result.upper()

#     return {
#         "reflection": result,
#         "approved": approved
#     }


# def reflection_router(state: AgentState):
#     return END if state["approved"] else "generate"


# # =========================
# # GRAPH
# # =========================

# def build_agent():
#     graph = StateGraph(AgentState)

#     graph.add_node("planner", planner_node)
#     graph.add_node("retrieve", retrieve_node)
#     graph.add_node("generate", generate_node)
#     graph.add_node("reflection", reflection_node)

#     graph.set_entry_point("planner")

#     graph.add_edge("planner", "retrieve")
#     graph.add_edge("retrieve", "generate")
#     graph.add_edge("generate", "reflection")

#     graph.add_conditional_edges("reflection", reflection_router)

#     return graph.compile(checkpointer=MemorySaver())


# # =========================
# # MAIN
# # =========================

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--mcp-url", default="http://localhost:8000/mcp")
#     args = parser.parse_args()

#     load_dotenv()

#     global llm, mcp_tools

#     api_key = os.getenv("GROQ_API_KEY")
#     if not api_key:
#         raise EnvironmentError("GROQ_API_KEY missing")

#     llm = build_llm(api_key)

#     # Connect MCP
#     mcp_tools = MCPTools(args.mcp_url)
#     asyncio.run(mcp_tools.connect())

#     agent = build_agent()

#     print("\n🚀 TelecomGPT MCP Agent Ready\n")

#     while True:
#         q = input("Question > ").strip()

#         if q.lower() in {"exit", "quit", "q"}:
#             break

#         result = asyncio.run(
#             agent.ainvoke(
#                 {"question": q},
#                 config={"configurable": {"thread_id": "telecom-session"}}
#             )
#         )

#         print("\nPLAN:\n", result["plan"])
#         print("\nANSWER:\n", result["answer"])
#         print("\nREFLECTION:\n", result["reflection"])


# if __name__ == "__main__":
#     main()

import argparse
import os
import asyncio
from typing import TypedDict

from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate

from mcp import ClientSession          # ← here
from mcp.client.sse import sse_client  # ← here

from rag_pipeline import build_llm


# =========================
# MCP CLIENT WRAPPER        ← replace your old MCPTools with this
# =========================

class MCPTools:
    def __init__(self, url: str):
        self.url = url
        self.session: ClientSession | None = None

    async def search_documents(self, query: str) -> str:
        res = await self.session.call_tool("search_documents", {"query": query})
        return res.content[0].text

    async def summarize_text(self, text: str) -> str:
        res = await self.session.call_tool("summarize_text", {"text": text})
        return res.content[0].text

    async def write_answer(self, filename: str, content: str, mode: str = "a") -> str:
        res = await self.session.call_tool("write_answer", {
            "filename": filename,
            "content": content,
            "mode": mode
        })
        return res.content[0].text


# =========================
# STATE
# =========================

class AgentState(TypedDict):
    question: str
    plan: str
    context: str
    answer: str
    reflection: str
    approved: bool


# =========================
# PROMPTS
# =========================

PLANNER_PROMPT = ChatPromptTemplate.from_template("""
You are TelecomGPT autonomous planner.

Question:
{question}

Give a step-by-step plan.
""")

GENERATION_PROMPT = ChatPromptTemplate.from_template("""
You are TelecomGPT.

Answer ONLY using the provided telecom context.

CONTEXT:
{context}

QUESTION:
{question}

RULES:
- technical
- structured
- telecom standards if possible
""")

REFLECTION_PROMPT = ChatPromptTemplate.from_template("""
You are a telecom QA reviewer.

Question:
{question}

Answer:
{answer}

Return ONLY "APPROVED" if correct, otherwise explain issues.
""")


# =========================
# AGENT BUILDER
# =========================

def build_agent(llm, mcp_tools):

    def planner_node(state: AgentState):
        prompt = PLANNER_PROMPT.format_messages(question=state["question"])
        return {"plan": llm.invoke(prompt).content}

    async def retrieve_node(state: AgentState):
        context = await mcp_tools.search_documents(state["question"])
        return {"context": context}

    def generate_node(state: AgentState):
        prompt = GENERATION_PROMPT.format_messages(
            context=state["context"],
            question=state["question"]
        )
        return {"answer": llm.invoke(prompt).content}

    def reflection_node(state: AgentState):
        prompt = REFLECTION_PROMPT.format_messages(
            question=state["question"],
            answer=state["answer"]
        )
        result = llm.invoke(prompt).content
        approved = "APPROVED" in result.upper()
        return {"reflection": result, "approved": approved}

    def reflection_router(state: AgentState):
        return END if state["approved"] else "generate"

    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("reflection", reflection_node)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "reflection")

    graph.add_conditional_edges("reflection", reflection_router)

    return graph.compile(checkpointer=MemorySaver())


# =========================
# MAIN ASYNC RUNNER         ← replace your old run_agent with this
# =========================

async def run_agent(mcp_url: str):
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY missing")

    llm = build_llm(api_key)
    mcp_tools = MCPTools(mcp_url)
    agent = build_agent(llm, mcp_tools)

    async with sse_client(mcp_url) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            mcp_tools.session = session
            await session.initialize()
            print("✅ Connected to MCP server")
            print("\n🚀 TelecomGPT MCP Agent Ready\n")

            while True:
                q = input("Question > ").strip()

                if not q:
                    continue

                if q.lower() in {"exit", "quit", "q"}:
                    print("👋 Goodbye!")
                    break

                save = input("💾 Save answer to file? (y/n) > ").strip().lower()
                filename = None
                if save == "y":
                    filename = input("📄 Filename (default: answers.txt) > ").strip()
                    if not filename:
                        filename = "answers.txt"

                result = await agent.ainvoke(
                    {"question": q},
                    config={"configurable": {"thread_id": "telecom-session"}}
                )

                print("\n📋 PLAN:\n", result["plan"])
                print("\n💡 ANSWER:\n", result["answer"])
                print("\n🔍 REFLECTION:\n", result["reflection"])

                if filename:
                    entry = f"Q: {q}\n\nA: {result['answer']}\n\n{'='*60}\n"
                    confirmation = await mcp_tools.write_answer(filename, entry, mode="a")
                    print(f"\n{confirmation}")


# =========================
# ENTRY POINT               ← keep this exactly as is at the bottom
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcp-url", default="http://localhost:8000/mcp")
    args = parser.parse_args()

    asyncio.run(run_agent(args.mcp_url))


if __name__ == "__main__":
    main()
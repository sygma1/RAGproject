from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import os
from langchain_chroma import Chroma
from rag_pipeline import (
    get_vectorstore,
    build_llm,
    rerank,
    format_context,
    TOP_K_RETRIEVE,
)

load_dotenv()

mcp = FastMCP("telecom-gpt")

# -----------------------
# INIT RESOURCES
# -----------------------
api_key = os.getenv("GROQ_API_KEY")

vs = get_vectorstore(force_rebuild=False)
retriever = vs.as_retriever(search_kwargs={"k": TOP_K_RETRIEVE})

llm = build_llm(api_key)

# -----------------------
# MCP TOOLS
# -----------------------
@mcp.tool()
def search_documents(query: str) -> str:
    """Search telecom documents using RAG retrieval."""
    docs = retriever.invoke(query)
    best_docs = rerank(query, docs)
    return format_context(best_docs)


@mcp.tool()
def summarize_text(text: str) -> str:
    """Summarize telecom technical text."""
    prompt = f"Summarize this telecom information:\n\n{text}"
    return llm.invoke(prompt).content


@mcp.tool()
def write_answer(filename: str, content: str, mode: str = "a") -> str:
    """
    Write or append an answer to a text file.

    Args:
        filename: Name of the file to write to (e.g. 'answers.txt').
                  Saved in an 'outputs' folder next to this script.
        content:  The answer or text to save.
        mode:     'a' to append (default) or 'w' to overwrite the file.

    Returns:
        A confirmation message with the absolute path of the file.
    """
    if mode not in ("a", "w"):
        return "Invalid mode. Use 'a' to append or 'w' to overwrite."

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Strip any path separators from filename to prevent directory traversal
    safe_filename = os.path.basename(filename)
    if not safe_filename.endswith(".txt"):
        safe_filename += ".txt"

    filepath = os.path.join(output_dir, safe_filename)

    with open(filepath, mode, encoding="utf-8") as f:
        f.write(content)
        if not content.endswith("\n"):
            f.write("\n")

    action = "Appended to" if mode == "a" else "Written to"
    return f"{action} file: {filepath}"


# -----------------------
# RUN SERVER
# -----------------------
if __name__ == "__main__":
    mcp.run(transport="sse")
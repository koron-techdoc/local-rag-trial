#!env python

from fastmcp import FastMCP

mcp = FastMCP("Vimdoc RAG")

@mcp.tool
def rag(query: str) -> str:
    """Ask Vim's documentation RAG"""
    return f"Your Query: {query}"

if __name__ == "__main__":
    mcp.run()

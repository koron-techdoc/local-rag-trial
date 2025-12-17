#!env python

import sys
import asyncio
from fastmcp import Client

client = Client("http://localhost:8000/mcp") # Assumes my_mcp_server.py exists

async def call_tool_rag(query):
    async with client:
        result = await client.call_tool("rag", {"query": query})
        print(result.data)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"USAGE: {sys.argv[0]} QUERY")
        exit(1)
    query =sys.argv[1]
    asyncio.run(call_tool_rag(query))

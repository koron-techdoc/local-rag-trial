#!env python

import asyncio
from fastmcp import Client

client = Client("http://localhost:8000/mcp")

async def list_tools():
    async with client:
        tools = await client.list_tools()
        for tool in tools:
            print(f"- {tool}")

if __name__ == "__main__":
    asyncio.run(list_tools())

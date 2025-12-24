#!env python
#
# DuckDB永続化したベクトルインデックスを読みこみ、MCPサーバーとして提供する

from fastmcp import FastMCP

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("setup logger")

from llama_index.core import Settings
from llama_index.vector_stores.duckdb import DuckDBVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
#from llama_index.llms.llamafile import Llamafile
from llama_index.llms.lmstudio import LMStudio
import gradio as gr
logger.info("imported packages")

# Embedding には HuggingFace 上の google/embeddinggemma-300m を利用する
embedding_model_name = "google/embeddinggemma-300m"

# Embedding モデルを読みこむ
Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
logger.info("loaded embedding model")

# ベクトルストア に DuckDB を利用する
# データは indexing.py で作ったベクトルスト
vector_store = DuckDBVectorStore.from_local("../repro/var/vectorstore.duckdb")
index = VectorStoreIndex.from_vector_store(vector_store)
logger.info("index ready")

# LLM モデルを読みこむ
#Settings.llm = Llamafile(base_url="http://localhost:8080", temperature=0, seed=0)
#Settings.llm = LMStudio(model_name="google/gemma-3-4b", request_timeout=120, temperature=0, seed=0)
Settings.llm = LMStudio(model_name="qwen/qwen3-vl-4b", request_timeout=120, temperature=0, seed=0)
logger.info("LLM model ready")

# 検索エンジンを作成
query_engine = index.as_query_engine()
logger.info("query engine ready")

# クエリーのための関数
def rag_query(query):
    r = query_engine.query(query)
    return r.response

# MCPサーバーを起動
mcp = FastMCP("Vimdoc RAG")

@mcp.tool
def rag(query: str) -> str:
    """Ask Vim's documentation RAG"""
    return rag_query(query)

if __name__ == "__main__":
    mcp.run()

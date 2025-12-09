#!env python
#
# DuckDB永続化したベクトルインデックスを読みこみ、WebUIでクエリを実行する

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("setup logger")

from llama_index.core import Settings
from llama_index.vector_stores.duckdb import DuckDBVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llamafile import Llamafile
import gradio as gr
logger.info("imported packages")

# Embedding には HuggingFace 上の google/embeddinggemma-300m を利用する
embedding_model_name = "google/embeddinggemma-300m"

# Embedding モデルを読みこむ
Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
logger.info("loaded embedding model")

# ベクトルストア に DuckDB を利用する
# データは indexing.py で作ったベクトルスト
vector_store = DuckDBVectorStore.from_local("./var/vectorstore.duckdb")
index = VectorStoreIndex.from_vector_store(vector_store)
logger.info("index ready")

# LLM モデルを読みこむ
Settings.llm = Llamafile(base_url="http://localhost:8080", temperature=0, seed=0)
logger.info("LLM model ready")

# 検索エンジンを作成
query_engine = index.as_query_engine()
logger.info("query engine ready")

# クエリーのための関数
def rag_query(query):
    r = query_engine.query(query)
    return r

# GradioでWeb UIを作って、クエリー関数と連結
ui = gr.Interface(
    title="Vimdoc RAG",
    fn=rag_query,
    inputs=["text"],
    outputs=["textarea"],
    api_name="rag_query"
)
ui.launch()

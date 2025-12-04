#!env python
#
# DuckDB永続化したベクトルインデックスを読みこみ、WebUIでクエリを実行する

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("setup logger")

from lib.my_debug_handler import MyDebugHandler
from llama_index.core.callbacks import CallbackManager
from llama_index.core import Settings
from llama_index.vector_stores.duckdb import DuckDBVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llamafile import Llamafile
import gradio as gr
logger.info("imported packages")

# オプションを解釈
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--debug', help='enable debug logging', action=argparse.BooleanOptionalAction)
args = parser.parse_args()
DEBUG = args.debug

# デバッグハンドラを組み込む
if DEBUG:
    my_debug = MyDebugHandler()
    callback_manager = CallbackManager([my_debug])
    Settings.callback_manager = callback_manager

# EmbeddingにHFのgoogle/embeddinggemma3-300m を使う
Settings.embed_model = HuggingFaceEmbedding(model_name="google/embeddinggemma-300m")
logger.info("embedding model ready")

# LLMにローカルで動く llama.cpp の llama-server を使う
Settings.llm = Llamafile(base_url="http://localhost:8080")
logger.info("LLM model ready")

# ベクトルストアには DuckDB を使う。データは test05a_indexing+duckdb.py で作ったモノ
vector_store = DuckDBVectorStore.from_local("./var/duckdb/vimdoc.duckdb")
index = VectorStoreIndex.from_vector_store(vector_store)
logger.info("index ready")

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

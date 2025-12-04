#!env python
#
# ファイルを読みこみベクトルインデックスを永続化する

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("setup logger")

from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.duckdb import DuckDBVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
logger.info("import llama_index related")

Settings.embed_model = HuggingFaceEmbedding(model_name="google/embeddinggemma-300m")
logger.info("loaded embedding model")

documents = SimpleDirectoryReader("var/data").load_data()
logger.info("loaded documents")

vector_store = DuckDBVectorStore("vimdoc.duckdb", persist_dir="./var/duckdb")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context)
logger.info("generated embeddings")

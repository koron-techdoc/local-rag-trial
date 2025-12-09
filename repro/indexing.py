#!env python
#
# ./var/data ディレクトリに置いてあるファイルを読みこみ
# ベクトル化してDuckDBベースのストアに保存する

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("setup logger")

from llama_index.core import Settings
from llama_index.vector_stores.duckdb import DuckDBVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext
logger.info("imported packages")

# Embedding には HuggingFace 上の google/embeddinggemma-300m を利用する
embedding_model_name = "google/embeddinggemma-300m"

# Embedding には HuggingFace 上の google/embeddinggemma-300m を利用する
Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
logger.info("loaded embedding model")

# ./var/data ディレクトリ内のファイルをドキュメント≒RAGの対象とする
documents = SimpleDirectoryReader("var/data").load_data()
logger.info("loaded documents")

# ベクトルストア に DuckDB を利用する
vector_store = DuckDBVectorStore("vectorstore.duckdb", persist_dir="./var")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# ベクトルストア≒インデックスにドキュメントを読みこむ
# ベクトルストアにDuckDBを用いているため、自動的に永続化される
index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True)
logger.info("generated embeddings")

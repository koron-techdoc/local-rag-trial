#!env python
#
# ファイルを読みこみベクトルインデックスを永続化する

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("setup logger")

from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
logger.info("import llama_index related")

Settings.embed_model = HuggingFaceEmbedding(model_name="google/embeddinggemma-300m")
logger.info("loaded embedding model")

documents = SimpleDirectoryReader("var/data").load_data()
logger.info("loaded documents")

index = VectorStoreIndex.from_documents(documents)
logger.info("generated embeddings")

index.storage_context(persist_dir="var/storage")

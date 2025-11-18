#!env python
#
# 永続化したベクトルインデックスを読みこみ、クエリを実行する

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("setup logger")

from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core import load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
logger.info("import llama_index related")

# EmbeddingにHFのgoogle/embeddinggemma3-300m を使う
Settings.embed_model = HuggingFaceEmbedding(model_name="google/embeddinggemma-300m")
logger.info("embedding model ready")

# LLMにHFのgoogle/gemma-3-1b-it を使う
Settings.llm = HuggingFaceLLM(
        model_name="google/gemma-3-1b-it",
        tokenizer_name="google/gemma-3-1b-it",
        context_window=32768,
)
logger.info("LLM model ready")

storage_context = StorageContext.from_defaults(persist_dir="var/storage")
logger.info("storage_context ready")

index = load_index_from_storage(storage_context)
logger.info("index ready")

query_engine = index.as_query_engine()
logger.info("query engine ready")

def query(q):
    r = query_engine.query(q)
    print(r)

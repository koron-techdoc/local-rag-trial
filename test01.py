#!env python
#
# とりあえずRAG用のデータベースを構築するサンプル:
#
# 参照元:
# https://developers.llamaindex.ai/python/framework/understanding/putting_it_all_together/q_and_a

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("setup logger")

import sys
query = sys.argv[1]
logger.info(f"query: {query}")

from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

logger.info("import llama_index related")

# EmbeddingにHFのgoogle/embeddinggemma3-300m を使うための準備
Settings.embed_model = HuggingFaceEmbedding(model_name="google/embeddinggemma-300m")
logger.info("loaded embedding model")

# LLMにHFのgoogle/gemma-3-1b-it を使う
Settings.llm = HuggingFaceLLM(
        model_name="google/gemma-3-1b-it",
        tokenizer_name="google/gemma-3-1b-it",
        context_window=32768,
)
logger.info("loaded LLM model")

documents = SimpleDirectoryReader("data").load_data()
logger.info("loaded documents")

index = VectorStoreIndex.from_documents(documents)
logger.info("generated embeddings")

query_engine = index.as_query_engine()
logger.info("generated the query engine")

response = query_engine.query(query)
logger.info(f"queried: {query}")

print(response)

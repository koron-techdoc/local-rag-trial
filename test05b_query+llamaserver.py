#!env python
#
# DuckDBへ永続化したベクトルインデックスを読みこみ、クエリを実行する

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("setup logger")

from llama_index.core import Settings
from llama_index.vector_stores.duckdb import DuckDBVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.callbacks import CallbackManager, CBEventType
from llama_index.core.callbacks.pythonically_printing_base_handler import PythonicallyPrintingBaseHandler
from typing import Any, Dict, List, Optional
from lib.llamafile_embedding import LlamafileEmbedding
from llama_index.llms.llamafile import Llamafile
logger.info("import llama_index related")

# オプションを解釈
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--debug', help='enable debug logging', action=argparse.BooleanOptionalAction)
args = parser.parse_args()
DEBUG = args.debug

# デバッグログ用のハンドラ
class MyDebugHandler(PythonicallyPrintingBaseHandler):
    def __init__(self):
        super().__init__()

    def on_event_start(
            self,
            event_type: CBEventType,
            payload: Optional[Dict[str, Any]] = None,
            event_id: str = "",
            parent_id: str = "",
            **kwargs: Any,
            ) -> str:
        self._print(f"MYDEBUG:event_start: type={event_type} id={event_id} payload={payload}")
        return event_id

    def on_event_end(
            self,
            event_type: CBEventType,
            payload: Optional[Dict[str, Any]] = None,
            event_id: str = "",
            **kwargs: Any,
            ) -> None:
        self._print(f"MYDEBUG:event_end: type={event_type} id={event_id} payload={payload}")

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        self._print(f"MYDEBUG:start_trace: id={trace_id}")

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self._print(f"MYDEBUG:end_trace: id={trace_id}")

# デバッグハンドラを組み込む
if DEBUG:
    my_debug = MyDebugHandler()
    callback_manager = CallbackManager([my_debug])
    Settings.callback_manager = callback_manager

# Embeddingに llama.cpp の llama-server を使う
Settings.embed_model = embedding = LlamafileEmbedding(base_url="http://localhost:8080")
logger.info("embedding model ready")

# LLMに llama.cpp の llama-server を使う
Settings.llm = Llamafile(base_url="http://localhost:8081", temperature=0, seed=0)
logger.info("LLM model ready")

vector_store = DuckDBVectorStore.from_local("./var/duckdb/vimdoc-llamaserver.duckdb")
index = VectorStoreIndex.from_vector_store(vector_store)
logger.info("index ready")

query_engine = index.as_query_engine()
logger.info("query engine ready")

def query(q):
    r = query_engine.query(q)
    print(r)

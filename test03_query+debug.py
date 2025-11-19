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
from llama_index.core.callbacks import CallbackManager, CBEventType
from llama_index.core.callbacks.pythonically_printing_base_handler import PythonicallyPrintingBaseHandler
from typing import Any, Dict, List, Optional
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
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

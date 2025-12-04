from typing import Any, Dict, List, Optional
from llama_index.core.callbacks import CBEventType
from llama_index.core.callbacks.pythonically_printing_base_handler import PythonicallyPrintingBaseHandler

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

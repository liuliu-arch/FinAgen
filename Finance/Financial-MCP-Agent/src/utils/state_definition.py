from typing import TypedDict, Sequence, Dict, Any, Annotated
from langchain_core.messages import BaseMessage
import operator


def merge_dicts(d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two dictionaries, d2 values overwrite d1."""
    return {**d1, **d2}


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Annotated[Dict[str, Any], merge_dicts]
    metadata: Annotated[Dict[str, Any], merge_dicts]
    # Potentially add a field for the initial user query if it needs to be passed around
    # user_query: str

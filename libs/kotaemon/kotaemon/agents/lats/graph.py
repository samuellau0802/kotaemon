from typing_extensions import TypedDict
from .node import Node

class TreeState(TypedDict):
    # The full tree
    root: Node
    # The original input
    input: str
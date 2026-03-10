"""
LangGraph 채팅 워크플로우.

retrieval → persona → factcheck → END

LangSmith 트레이싱 자동 활성화 조건 (환경변수):
  LANGCHAIN_TRACING_V2=true
  LANGCHAIN_API_KEY=ls__...
  LANGCHAIN_PROJECT=persona-cv
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from services.agents.nodes import factcheck_node, persona_node, retrieval_node
from services.agents.state import ChatState


def build_chat_graph():
    workflow = StateGraph(ChatState)

    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("persona", persona_node)
    workflow.add_node("factcheck", factcheck_node)

    workflow.set_entry_point("retrieval")
    workflow.add_edge("retrieval", "persona")
    workflow.add_edge("persona", "factcheck")
    workflow.add_edge("factcheck", END)

    return workflow.compile()


# 모듈 로드 시 한 번만 컴파일
chat_graph = build_chat_graph()

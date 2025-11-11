# -*- coding: utf-8 -*-
"""
루트 오케스트레이터
- Day2를 Function Tool로 변경하여 안정적으로 연동
"""

from __future__ import annotations
from typing import Optional

from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from google.adk.models.lite_llm import LiteLlm

# 서브 에이전트/툴
from student.day1.agent import day1_web_agent
from student.day2.agent import day2_rag_agent
from student.day3.agent import day3_gov_agent

# 프롬프트
from .prompt import ORCHESTRATOR_DESC, ORCHESTRATOR_PROMPT


# ------------------------------------------------------------------------------
# TODO[ROOT-A-01] 모델 선택 ✅
# ------------------------------------------------------------------------------
MODEL = LiteLlm(model="openai/gpt-4o-mini")


# ------------------------------------------------------------------------------
# TODO[ROOT-A-02] 루트 에이전트 구성 ✅
# ------------------------------------------------------------------------------
root_agent = Agent(
    name="KT_AIVLE_Orchestrator",
    model=MODEL,
    description=ORCHESTRATOR_DESC,
    instruction=ORCHESTRATOR_PROMPT,
    tools=[
        # Day1: Web Search (Agent Tool)
        AgentTool(agent=day1_web_agent),
        
        # ✅ Day2: RAG (Function Tool) - AgentTool 없이 직접 등록
        AgentTool(agent=day2_rag_agent),
        
        # Day3: Government Support (Agent Tool)
        AgentTool(agent=day3_gov_agent),
    ],
)
import json
import logging
import re
from functools import partial
from typing import List, Optional

import tiktoken

from kotaemon.agents.base import BaseAgent, BaseLLM
from kotaemon.agents.io import AgentAction, AgentFinish, AgentOutput, AgentType
from kotaemon.agents.tools import BaseTool
from kotaemon.base import Document, Param
from kotaemon.indices.splitters import TokenSplitter
from kotaemon.llms import PromptTemplate
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import chain as as_runnable
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation


from libs.kotaemon.kotaemon.agents.lats.graph import TreeState
from libs.kotaemon.kotaemon.agents.lats.node import Node
from libs.kotaemon.kotaemon.llms.chats.langchain_based import LCChatOpenAI


from .reflection import Reflection
from .prompt import reflection_prompt_template, chat_prompt_template

from collections import defaultdict

from typing import Literal

from langgraph.graph import END, StateGraph, START




class LATSAgent(BaseAgent):
    """
    LATSAgent class inherited from BaseAgent.
    Implementing LATS agent paradigm https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/lats/lats.ipynb
    """

    name: str = "LATSAgent"
    agent_type: AgentType = AgentType.lats
    description: str = "LATSAgent for answering multi-step reasoning questions"
    llm: LCChatOpenAI
    output_lang: str = "English"

    tools: List[BaseTool]
    tool_executor = ToolExecutor(tools=tools)
    n: int = 5

    reflection_llm_chain = (
        reflection_prompt_template
        | llm.bind_tools(tools=[Reflection], tool_choice="Reflection").with_config(
            run_name="Reflection"
        )
        | PydanticToolsParser(tools=[Reflection])
    )
    initial_answer_chain = chat_prompt_template | llm.bind_tools(tools=tools).with_config(
        run_name="GenerateInitialCandidate"
    )
    parser = JsonOutputToolsParser(return_id=True)
    

    @as_runnable
    def reflection_chain(self, inputs) -> Reflection:
        tool_choices = self.reflection_llm_chain.invoke(inputs)
        reflection = tool_choices[0]
        if not isinstance(inputs["candidate"][-1], AIMessage):
            reflection.found_solution = False
        return reflection
    
    # Define the node we will add to the graph
    def generate_initial_response(self, state: TreeState) -> dict:
        """Generate the initial candidate response."""
        res = self.initial_answer_chain.invoke({"input": state["input"]})
        parsed = self.parser.invoke(res)
        tool_responses = self.tool_executor.batch(
            [ToolInvocation(tool=r["type"], tool_input=r["args"]) for r in parsed]
        )
        output_messages = [res] + [
            ToolMessage(content=json.dumps(resp), tool_call_id=tool_call["id"])
            for resp, tool_call in zip(tool_responses, parsed)
        ]
        reflection = self.reflection_chain.invoke(
            {"input": state["input"], "candidate": output_messages}
        )
        root = Node(output_messages, reflection=reflection)
        return {
            **state,
            "root": root,
        }
    
    # This generates N candidate values
    # for a single input to sample actions from the environment
    def generate_candidates(self, messages: ChatPromptValue, config: RunnableConfig):
        bound_kwargs = self.llm.bind_tools(tools=self.tools).kwargs
        chat_result = self.llm.generate(
            [messages.to_messages()],
            n=self.n,
            callbacks=config["callbacks"],
            run_name="GenerateCandidates",
            **bound_kwargs,
        )
        return [gen.message for gen in chat_result.generations[0]]


    expansion_chain = chat_prompt_template | generate_candidates
    

    def expand(self, state: TreeState, config: RunnableConfig) -> dict:
        """Starting from the "best" node in the tree, generate N candidates for the next step."""
        root = state["root"]
        best_candidate: Node = root.best_child if root.children else root
        messages = best_candidate.get_trajectory()
        # Generate N candidates from the single child candidate
        new_candidates = self.expansion_chain.invoke(
            {"input": state["input"], "messages": messages}, config
        )
        parsed = self.parser.batch(new_candidates)
        flattened = [
            (i, tool_call)
            for i, tool_calls in enumerate(parsed)
            for tool_call in tool_calls
        ]
        tool_responses = self.tool_executor.batch(
            [
                ToolInvocation(tool=tool_call["type"], tool_input=tool_call["args"])
                for _, tool_call in flattened
            ]
        )
        collected_responses = defaultdict(list)
        for (i, tool_call), resp in zip(flattened, tool_responses):
            collected_responses[i].append(
                ToolMessage(content=json.dumps(resp), tool_call_id=tool_call["id"])
            )
        output_messages = []
        for i, candidate in enumerate(new_candidates):
            output_messages.append([candidate] + collected_responses[i])

        # Reflect on each candidate
        # For tasks with external validation, you'd add that here.
        reflections = self.reflection_chain.batch(
            [{"input": state["input"], "candidate": msges} for msges in output_messages],
            config,
        )
        # Grow tree
        child_nodes = [
            Node(cand, parent=best_candidate, reflection=reflection)
            for cand, reflection in zip(output_messages, reflections)
        ]
        best_candidate.children.extend(child_nodes)
        # We have already extended the tree directly, so we just return the state
        return state


    def should_loop(state: TreeState) -> Literal["expand", "__end__"]:
        """Determine whether to continue the tree search."""
        root = state["root"]
        if root.is_solved:
            return END
        if root.height > 5:
            return END
        return "expand"
    

    def build_graph(self):
        builder = StateGraph(TreeState)
        builder.add_node("start", self.generate_initial_response)
        builder.add_node("expand", self.expand)
        builder.add_edge(START, "start")


        builder.add_conditional_edges(
            "start",
            # Either expand/rollout or finish
            self.should_loop,
        )
        builder.add_conditional_edges(
            "expand",
            # Either continue to rollout or finish
            self.should_loop,
        )

        graph = builder.compile()
        return graph

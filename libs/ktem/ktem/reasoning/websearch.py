import html
import logging
import threading
from collections import defaultdict
from difflib import SequenceMatcher
from functools import partial
from typing import Generator
import urllib.parse

from libs.ktem.ktem.reasoning.simple import FullQAPipeline
import numpy as np
import requests
import tiktoken
from ktem.llms.manager import llms
from ktem.reasoning.prompt_optimization import (
    DecomposeQuestionPipeline,
    RewriteQuestionPipeline,
)
from ktem.utils.render import Render
from theflow.settings import settings as flowsettings
from decouple import config

from kotaemon.base import (
    AIMessage,
    BaseComponent,
    Document,
    HumanMessage,
    Node,
    RetrievedDocument,
    SystemMessage,
)
from kotaemon.indices.qa.citation import CitationPipeline
from kotaemon.indices.splitters import TokenSplitter
from kotaemon.llms import ChatLLM, PromptTemplate

from ..utils import SUPPORTED_LANGUAGE_MAP
from .base import BaseReasoning

logger = logging.getLogger(__name__)

EVIDENCE_MODE_TEXT = 0
EVIDENCE_MODE_TABLE = 1
EVIDENCE_MODE_CHATBOT = 2
EVIDENCE_MODE_FIGURE = 3
MAX_IMAGES = 10


def find_text(search_span, context):
    sentence_list = search_span.split("\n")
    matches = []
    # don't search for small text
    if len(search_span) > 5:
        for sentence in sentence_list:
            match = SequenceMatcher(
                None, sentence, context, autojunk=False
            ).find_longest_match()
            if match.size > len(sentence) * 0.35:
                matches.append((match.b, match.b + match.size))

    return matches


def web_search(
        message: str
    ) -> tuple[list[RetrievedDocument], list[Document]]:
        
        def create_search_url(message):
            base_url = 'https://s.jina.ai/'
            encoded_message = urllib.parse.quote(message)
            search_url = f"{base_url}{encoded_message}"
            print(search_url)
            return search_url
        
        docs = []
        doc_ids = []

        # Define the URL and headers
        url = create_search_url(message)
        api_key = config("JINA_API_KEY", default="")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "X-With-Links-Summary": "true",
                "Accept": "application/json"
        }

        # Make the GET request
        response = requests.get(url, headers=headers)

        # Check the response
        if response.status_code == 200:
            result = response.json()
        else:
            print(f"Web Search Error: {response.status_code} - {response.text}")
        
        docs, doc_ids = [], []
        plot_docs = []

        for idx, webpage in enumerate(result['data']):

            doc = RetrievedDocument(
                source=f"webpage-{idx}",
                content=webpage['content'],
                metadata={
                    "type": "info",
                },
                channel="info",
                score=0.5,
            )

            docs.append(doc)
            doc_ids.append(f"webpage-{idx}")


        info = [
            Document(
                channel="info",
                content=Render.collapsible_with_header(doc, open_collapsible=True),
            )
            for doc in docs
        ]

        print("###########")
        print("Web Search")
        print("###########")
        print(f"Found {len(docs)} webpages.")
        print(docs)
        return docs, info



class WebSearchPipeline(FullQAPipeline):
    """Web Search pipeline. Handle from question to answer"""

    class Config:
        allow_extra = True

    def retrieve(
        self, message: str, history: list
    ) -> tuple[list[RetrievedDocument], list[Document]]:
        """Retrieve the documents based on the message"""
        # if len(message) < self.trigger_context:
        #     # prefer adding context for short user questions, avoid adding context for
        #     # long questions, as they are likely to contain enough information
        #     # plus, avoid the situation where the original message is already too long
        #     # for the model to handle
        #     query = self.add_query_context(message, history).content
        # else:
        #     query = message
        # print(f"Rewritten query: {query}")
        
        docs, info = web_search(message)

        return docs, info

  
    @classmethod
    def get_pipeline(cls, settings, states, retrievers):
        """Get the reasoning pipeline

        Args:
            settings: the settings for the pipeline
            retrievers: the retrievers to use
        """
        max_context_length_setting = settings.get("reasoning.max_context_length", 32000)

        pipeline = cls(
            retrievers=retrievers,
            rewrite_pipeline=RewriteQuestionPipeline(),
        )

        prefix = f"reasoning.options.{cls.get_info()['id']}"
        llm_name = settings.get(f"{prefix}.llm", None)
        llm = llms.get(llm_name, llms.get_default())

        # prepare evidence pipeline configuration
        evidence_pipeline = pipeline.evidence_pipeline
        evidence_pipeline.max_context_length = max_context_length_setting

        # answering pipeline configuration
        answer_pipeline = pipeline.answering_pipeline
        answer_pipeline.llm = llm
        answer_pipeline.citation_pipeline.llm = llm
        answer_pipeline.n_last_interactions = settings[f"{prefix}.n_last_interactions"]
        answer_pipeline.enable_citation = settings[f"{prefix}.highlight_citation"]
        answer_pipeline.system_prompt = settings[f"{prefix}.system_prompt"]
        answer_pipeline.qa_template = settings[f"{prefix}.qa_prompt"]
        answer_pipeline.lang = SUPPORTED_LANGUAGE_MAP.get(
            settings["reasoning.lang"], "English"
        )

        pipeline.add_query_context.llm = llm
        pipeline.add_query_context.n_last_interactions = settings[
            f"{prefix}.n_last_interactions"
        ]

        pipeline.trigger_context = settings[f"{prefix}.trigger_context"]
        pipeline.use_rewrite = states.get("app", {}).get("regen", False)
        if pipeline.rewrite_pipeline:
            pipeline.rewrite_pipeline.llm = llm
            pipeline.rewrite_pipeline.lang = SUPPORTED_LANGUAGE_MAP.get(
                settings["reasoning.lang"], "English"
            )
        return pipeline


    @classmethod
    def get_info(cls) -> dict:
        return {
            "id": "Web Search",
            "name": "Web Search",
            "description": (
                "Simple RAG-based question answering pipeline. This pipeline can "
                "search online using Jina API to retrieve the "
                "context. After that it includes that context to generate the answer."
            ),
        }

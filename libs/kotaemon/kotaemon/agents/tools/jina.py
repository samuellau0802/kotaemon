from typing import AnyStr, Optional, Type
from urllib.error import HTTPError

from langchain_community.utilities import SerpAPIWrapper
from pydantic import BaseModel, Field
import urllib.parse
import requests
from decouple import config

from .base import BaseTool

class JinaSearchArgs(BaseModel):
    query: str = Field(..., description="a search query")


class JinaSearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "A search engine retrieving top search results as snippets from Google. "
        "Input should be a search query."
    )
    args_schema: Optional[Type[BaseModel]] = JinaSearchArgs

    def _create_search_url(self, message):
        base_url = 'https://s.jina.ai/'
        encoded_message = urllib.parse.quote(message)
        search_url = f"{base_url}{encoded_message}"
        print(search_url)
        return search_url


    def _run_tool(self, query: AnyStr) -> str:
        try:
            output = ""
            url = self._create_search_url(query)
            api_key = config("JINA_API_KEY", default="")
            headers = {
                "Authorization": f"Bearer {api_key}",
                "X-With-Links-Summary": "true",
                    "Accept": "application/json"
            }

            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                result = response.json()

                # Assuming the structure is {'data': [{'title': '...'}, ...]}
                if 'data' in result:
                    output = "\n".join(item['title'] for item in result['data'])
                else:
                    output = "No data found in the response."

        except HTTPError:
            output = "No evidence found."

        return output


# class SerpTool(BaseTool):
#     name = "google_search"
#     description = (
#         "Worker that searches results from Google. Useful when you need to find short "
#         "and succinct answers about a specific topic. Input should be a search query."
#     )
#     args_schema: Optional[Type[BaseModel]] = GoogleSearchArgs

#     def _run_tool(self, query: AnyStr) -> str:
#         tool = SerpAPIWrapper()
#         evidence = tool.run(query)

#         return evidence

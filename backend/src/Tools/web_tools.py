from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from rich import print

from Configs.env import TAVILY_API_KEY
from langchain_tavily import TavilySearch


# tavily_search = TavilySearch(max_results=5, include_answer=True)
tavily_search = TavilySearch(max_results=5)


@tool
def web_search(query: str) -> str:
    """Search the web for the recent and reliable information on a topic . Returns Titles , URLs and some snippets"""
    response = tavily_search.run(query)
    formatted_response = [
        f" Title : {r['title']} \n URL:{r['url']}\n Snippet : {r['content'][:300]}\n\n"
        for r in response["results"]
    ]
    return "\n---------\n.".join(formatted_response)


# print(web_search.invoke("What are the price of iphone 18"))
@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL for deeper reading."""
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        # print(resp.text)
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:5000]

    except Exception as e:
        return f"Could not scrape URL:{str(e)}"


# print(scrape_url.invoke("https://en.wikipedia.org/wiki/Virat_Kohli"))

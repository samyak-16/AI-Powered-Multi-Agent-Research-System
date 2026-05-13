from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from Tools.web_tools import web_search, scrape_url
from Configs.env import OPENAI_API_KEY

# Model Setup
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def build_search_agent():
    return create_agent(model=llm, tools=[web_search])


def build_reader_agent():
    return create_agent(model=llm, tools=[scrape_url])


research_aggereater_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert research writer. Writ clear , structured and insightful report",
        ),
        (
            "human",
            """Write a detailed research report on the topic below.
         
         Topic : {topic}

         Research gathered :
         {research}
         Structure the report as :

         - Introduction
         - Key Findings(minimum 3 well-explained points)
         - Conclusion
         - Sources (List all URLs found in the research)
         
         """,
        ),
    ]
)
research_reasoning_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a sharp and constructive research critic. Be honest and specific.",
        ),
        (
            "human",
            """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
...""",
        ),
    ]
)
parser = StrOutputParser()

writer_chain = research_aggereater_prompt | llm | parser


reasoning_chain = research_reasoning_prompt | llm | parser

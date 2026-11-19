from functools import lru_cache

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


@lru_cache(maxsize=1)
def get_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

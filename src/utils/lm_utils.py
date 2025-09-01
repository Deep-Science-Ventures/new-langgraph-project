import getpass
import os
from langchain_google_genai import ChatGoogleGenerativeAI

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("GOOGLE_API_KEY")


def get_llm(model="gemini-2.0-flash-exp"):
    # Initialize Gemini model
    return ChatGoogleGenerativeAI(
        model=model,
    )
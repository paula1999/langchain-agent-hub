from tools.tools import tools

from langchain_google_genai import ChatGoogleGenerativeAI

# LLM setup
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

# Bind tools to the LLM
llm.bind_tools(tools)
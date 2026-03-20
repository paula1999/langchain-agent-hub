from tools.tools import tools

from langchain_google_genai import ChatGoogleGenerativeAI

# Configuración del LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

llm.bind_tools(tools)
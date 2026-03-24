from tools.tools import tools

from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

# LLM setup
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_retries=2,
    timeout=None
)

# Bind tools to the LLM
llm_tools = llm.bind_tools(tools)

# System prompt
main_system_prompt = f"""
You are an intelligent AI assistant who answers questions.
Use the retriever tool available to answer questions about europe context.
Use the location tool available to get the current location of the user.
Use the get weather tool available to get the weather of a province code. You can use the get province code tool to get the province code.
You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
Today's date is {datetime.today()}
"""
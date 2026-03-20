from tools.api_tool import location_tool, get_province_code, get_weather
from tools.pdf_tool import retriever_tool

# Tools imports
tools = [
    location_tool,
    get_province_code,
    get_weather,
    retriever_tool
]

tools_dict = {t.name: t for t in tools}
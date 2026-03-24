from dotenv import load_dotenv

# Env vars
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from agents.graph import create_graph, running_agent
from utils.ingest import load_files


app = FastAPI(title="FastAPI Agents")

rag_agent = create_graph()


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"


@app.post("/chat")
async def chat_node(request: ChatRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    input_data = {"messages": [HumanMessage(content=request.message)]}
    
    final_text = ""
    
    try:
        for event in rag_agent.stream(input_data, config=config):
            for node_name, output in event.items():
                if "messages" in output:
                    last_msg = output["messages"][-1]
                    if isinstance(last_msg, AIMessage) and last_msg.content:
                        print(f"AI: {last_msg.content}")
                        final_text = last_msg.content
                    elif hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                        print(f"LLM calling Tool: {last_msg.tool_calls[0]['name']}")
                    elif isinstance(last_msg, ToolMessage) and last_msg.content:
                        print(f"Tool: {last_msg.content}")
                        
        return {"response": final_text, "thread_id": request.thread_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# if __name__ == '__main__':
#     """
#     This is the main entry point of the application. It first loads the PDF files into the vector store, then creates the RAG agent and runs it.
#     """
#     # Ingest files to DB
#     #load_files()

#     rag_agent = create_graph()

#     # Agent
#     running_agent(rag_agent)
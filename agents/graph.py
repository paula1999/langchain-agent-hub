from typing import TypedDict, Annotated, Sequence
from operator import add as add_messages

from agents.agents_config import llm, llm_tools, main_system_prompt
from tools.tools import tools, tools_dict, tool_node

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Agent State Definition
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState):
    """
    Check if the last message contains tool calls.
    Args:
        - state: The current state of the agent, which includes a sequence of messages.
    Returns:
        - A boolean indicating whether the agent should continue to the retriever agent (True) or end the conversation (False). The agent should continue if the last message contains tool calls, indicating that there are actions to be taken based on the LLM's response. If there are no tool calls, it means the LLM has provided a final answer and the conversation can end.
    """
    print(f'\n\nState:\n{state}\n\n')
    last_message = state['messages'][-1]
    return last_message.tool_calls



# LLM Agent
def call_llm(state: AgentState) -> AgentState:
    """
    Function to call the LLM with the current state.
    Args:
        - state: The current state of the agent, which includes a sequence of messages.
    Returns:
        - The updated state of the agent after calling the LLM, which includes the new messages from the LLM's response. The function takes the existing messages from the state, adds a system prompt at the beginning, and then invokes the LLM with this updated sequence of messages. The response from the LLM is then appended to the messages in the state, and the updated state is returned.
    """
    messages = list(state['messages'])
    messages = [SystemMessage(content=main_system_prompt)] + messages
    response = llm_tools.invoke(messages)
    
    return {"messages": [response]}


def create_graph():
    """
    This function creates the state graph for the RAG agent. It defines the nodes and edges of the graph, including the LLM node and the tools node. The graph is designed to allow for conditional transitions based on whether the LLM's response includes tool calls. If the LLM calls a tool, the graph transitions to the tools node; otherwise, it continues to process the LLM's response. The function also sets up memory saving for the agent's state and compiles the graph into a runnable RAG agent.
    Returns:
        - The compiled RAG agent that is ready to be run. This agent can process user input, interact with the LLM, and call tools as needed based on the LLM's responses.
    """
    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("llm")

    graph.add_conditional_edges(
        "llm",
        tools_condition
    )
    graph.add_edge("tools", "llm")

    memory = MemorySaver()

    rag_agent = graph.compile(checkpointer=memory)

    return rag_agent


def running_agent(rag_agent):
    """
    This function runs the RAG agent in a loop, allowing for continuous interaction with the user. It takes user input, feeds it into the agent, and processes the agent's responses, including any tool calls made by the LLM. The function maintains a session state that is updated with each interaction, enabling the agent to have context across multiple turns of conversation. The loop continues until the user decides to exit by typing 'exit'. During each iteration, the function also prints out the AI's responses and any tool calls that are made, providing visibility into the agent's decision-making process and actions taken based on the user's input.
    Args:
        - rag_agent: The compiled RAG agent that has been created using the state graph.
    Returns:
        - None. The function runs indefinitely until the user decides to exit, at which point it terminates the conversation loop.
    """
    config = {"configurable": {"thread_id": "sesion_1"}}
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() == 'exit': break
        
        input_data = {"messages": [HumanMessage(content=user_input)]}
        
        # stream
        for event in rag_agent.stream(input_data, config=config):
            for node_name, output in event.items():
                #print(f"\n--- Executing node: {node_name} ---")
                
                if "messages" in output:
                    last_msg = output["messages"][-1]
                    if isinstance(last_msg, AIMessage) and last_msg.content:
                        print(f"AI: {last_msg.content}")
                    elif hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                        print(f"LLM calling Tool: {last_msg.tool_calls[0]['name']}")
                    elif isinstance(last_msg, ToolMessage) and last_msg.content:
                        print(f"Tool: {last_msg.content}")

        final_state = rag_agent.get_state(config)
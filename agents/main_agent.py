from typing import TypedDict, Annotated, Sequence
from operator import add as add_messages

from agents.agents_config import llm
from tools.tools import tools, tools_dict

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage
from langgraph.graph import StateGraph, END

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
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0


system_prompt = """
You are an intelligent AI assistant who answers questions.
Use the retriever tool available to answer questions about europe context.
Use the location tool available to get the current location of the user.
Use the get weather tool available to get the weather of a province code. You can use the get province code tool to get the province code.
You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""


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
    messages = [SystemMessage(content=system_prompt)] + messages
    response = llm.invoke(messages)
    
    state['messages'].append(AIMessage(content=response.content))
    
    return state


# Retriever Agent
def take_action(state: AgentState) -> AgentState:
    """
    Execute tool calls from the LLM's response.
    Args:
        - state: The current state of the agent, which includes a sequence of messages. The last message in this sequence is expected to be the LLM's response, which may contain tool calls that need to be executed.
    Returns:
        - The updated state of the agent after executing the tool calls, which includes new messages with the results of the tool executions. The function iterates through the tool calls in the last message, executes the corresponding tools based on their names, and appends the results as new ToolMessage instances to the messages in the state. If a tool name is not recognized, it appends an error message instead. After processing all tool calls, the updated state is returned, allowing the conversation to continue with the new information obtained from the tools. This enables a dynamic interaction where the LLM can request specific actions to be taken and receive the results in real-time, facilitating a more informed and context-aware response from the LLM in subsequent turns.
    """

    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        
        if not t['name'] in tools_dict: # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")
            

        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}


def create_graph():
    """
    This function creates a state graph for the RAG agent. It defines the states and transitions between them based on the LLM's responses and the execution of tool calls. The graph starts with the LLM state, where the LLM is called to generate a response based on the current conversation history. If the LLM's response contains tool calls, the graph transitions to the retriever agent state, where the specified tools are executed to obtain relevant information. After executing the tools, the graph transitions back to the LLM state, allowing the LLM to generate a new response based on the results from the tools. If there are no tool calls in the LLM's response, the graph transitions to an end state, concluding the conversation. This structure enables a dynamic interaction where the LLM can request specific actions to be taken and receive real-time results from those actions, facilitating a more informed and context-aware response in subsequent turns.
    Returns:
        - The compiled RAG agent based on the defined state graph, which can be invoked to handle conversations with the ability to execute tools as needed based on the LLM's responses.
    """
    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("retriever_agent", take_action)

    graph.add_conditional_edges(
        "llm",
        should_continue,
        {True: "retriever_agent", False: END}
    )
    graph.add_edge("retriever_agent", "llm")
    graph.set_entry_point("llm")

    rag_agent = graph.compile()

    return rag_agent


def running_agent(rag_agent):
    """
    This function runs the RAG agent in a loop, allowing for continuous interaction with the user. It maintains a conversation history that is updated with each user input and the corresponding responses from the agent. The loop continues until the user types 'exit', at which point the conversation ends. During each iteration, the user's input is added to the conversation history as a HumanMessage, and the RAG agent is invoked with the current state of the conversation. The agent processes the messages, potentially executing tools as needed, and generates a response that is printed to the user. The conversation history is then updated with the new messages from the agent's response, allowing for a dynamic and context-aware interaction in subsequent turns.
    Args:
        - rag_agent: The compiled RAG agent that will be invoked to handle the conversation with the user. This agent is expected to manage the flow of the conversation, including calling the LLM and executing tools based on the user's input and the LLM's responses.
    """
    conversation_history = []
    
    user_input = input("\nUser: ")
    while user_input.lower() != 'exit':
        conversation_history.append(HumanMessage(content=user_input))
        result = rag_agent.invoke({"messages": conversation_history})

        print(f'AI: {result['messages'][-1].content}')
        
        conversation_history = result["messages"]
        user_input = input("\nUser: ")
        
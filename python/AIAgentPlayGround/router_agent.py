import operator
from typing import Annotated, TypedDict, List
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

# 1. Configuration
UNRAID_IP = "192.168.50.200"  # e.g., 192.168.1.50
PORT = "4567"
MODEL_NAME = "deepseek-r1"  # e.g., qwen2.5 or deepseek-r1:8b


# 2. Define the State (Our "Clipboard")
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    category: str


# 3. Define the Nodes (The Workers)
def categorize_message(state: AgentState):
    last_message = state["messages"][-1].content
    llm = ChatOllama(base_url=f"http://{UNRAID_IP}:{PORT}", model=MODEL_NAME)

    prompt = (
        "Analyze the following user input and determine if it is related to 'Math' or 'General' topics. "
        "Output ONLY the single word 'Math' or 'General'. "
        f"\n\nUser Input: {last_message}"
    )

    response = llm.invoke(prompt)
    category = response.content.strip()
    return {"category": category}


def handle_math(state: AgentState):
    return {"messages": [AIMessage(content="I am a Math Agent. I can help you calculate that.")]}


def handle_general(state: AgentState):
    return {"messages": [AIMessage(content="I am a General Agent.")]}


# 4. Define Routing Logic
def route_after_prediction(state: AgentState):
    if state["category"].lower() == "math":
        return "math"
    return "general"


# 5. Build the Graph
workflow = StateGraph(AgentState)

workflow.add_node("categorizer", categorize_message)
workflow.add_node("math", handle_math)
workflow.add_node("general", handle_general)

workflow.set_entry_point("categorizer")

# This connects categorizer to either math or general based on the function
workflow.add_conditional_edges(
    "categorizer",
    route_after_prediction,
    {
        "math": "math",
        "general": "general"
    }
)

# End the conversation after the response
workflow.add_edge("math", END)
workflow.add_edge("general", END)

# 6. Compile and Run
app = workflow.compile()

# Test it out!
inputs = {"messages": [HumanMessage(content="what is the colour of a red apple?")]}
config = {"recursion_limit": 10}

for output in app.stream(inputs, config):
    for key, value in output.items():
        print(f"Node '{key}':")
        print(value)
        print("-" * 20)
from typing import TypedDict, Any, List, Literal

from langgraph.graph import StateGraph, END
from agent.dspy_signatures import RouterModule, SQLGenerator, Synthesizer
from agent.tools.sqlite_tool import SqliteTool
from agent.rag.retrieval import BM25Retriever


# Define the state for our agent
class AgentState(TypedDict):
    question: str
    format_hint: str
    route: str
    rag_context: List[dict]
    sql_query: str
    sql_results: List[dict]
    final_answer: Any
    citations: List[str]
    errors: List[str]
    repair_count: int


# Node functions
def route_node(state: AgentState) -> AgentState:
    """Determines the path to take (RAG, SQL, or Hybrid)."""
    print("---ROUTING---")
    # This would be a call to a DSPy module
    # router = RouterModule()
    # result = router(question=state['question'])
    # For now, we can manually set it for testing
    if "average order value" in state["question"].lower():
        state["route"] = "hybrid"
    elif "policy" in state["question"].lower() or "calendar" in state["question"].lower():
        state["route"] = "rag"
    else:
        state["route"] = "sql"
    return state

def retrieve_node(state: AgentState) -> AgentState:
    """Retrieves documents using BM25."""
    print("---RETRIEVING (RAG)---")
    retriever = BM25Retriever()
    state["rag_context"] = retriever.search(query=state["question"])
    return state

def sql_gen_node(state: AgentState) -> AgentState:
    """Generates an SQL query from the user's question."""
    print("---GENERATING SQL---")
    # This would be a call to a DSPy module
    # generator = SQLGenerator()
    # with SqliteTool() as db:
    #     schema = db.get_all_schemas()
    # result = generator(question=state['question'], schema=schema)
    # state['sql_query'] = result.sql_query
    state['sql_query'] = "SELECT 'dummy query'" # Placeholder
    return state

def execute_sql_node(state: AgentState) -> AgentState:
    """Executes the generated SQL query."""
    print("---EXECUTING SQL---")
    try:
        with SqliteTool() as db:
            state["sql_results"] = db.execute_sql(state["sql_query"])
    except Exception as e:
        print(f"  -> SQL execution error: {e}")
        state["errors"].append(str(e))
    return state

def synthesize_node(state: AgentState) -> AgentState:
    """Synthesizes the final answer from the context."""
    print("---SYNTHESIZING ANSWER---")
    # This would be a call to a DSPy module
    # synthesizer = Synthesizer()
    # result = synthesizer(...)
    state["final_answer"] = "This is a synthesized answer." # Placeholder
    state["citations"] = ["doc1::chunk1", "Orders"] # Placeholder
    return state

def validate_node(state: AgentState) -> AgentState:
    """Validates the output and checks for the need for repair."""
    print("---VALIDATING---")
    # For now, assume no repair is needed if there are no errors
    if state.get("errors") and len(state["errors"]) > 0:
        state["repair_count"] = state.get("repair_count", 0) + 1
    return state

def repair_node(state: AgentState) -> AgentState:
    """Repairs the query or answer based on errors."""
    print(f"---REPAIRING (Attempt {state.get('repair_count', 1)})---")
    # In a real scenario, you would analyze state['errors'] and retry
    # a specific node, like sql_gen_node or synthesizer.
    # For now, we'll just clear errors to avoid an infinite loop.
    state["errors"] = []
    return state


# Conditional edge logic
def route_decision(state: AgentState) -> Literal["retrieve_node", "sql_gen_node"]:
    """Decides the next step after the router."""
    if state["route"] in ("rag", "hybrid"):
        return "retriever"
    elif state["route"] == "sql":
        return "sql_generator"

def after_retrieval_decision(state: AgentState) -> Literal["sql_gen_node", "synthesizer"]:
    """Decides the next step after the retriever."""
    if state["route"] == "hybrid":
        return "sql_generator"
    else: # RAG-only
        return "synthesizer"
        
def needs_repair(state: AgentState) -> Literal["repair", "__end__"]:
    """Decides if the agent needs to repair its output."""
    if state.get("errors") and state.get("repair_count", 0) < 2:
        return "repair"
    else:
        return "__end__"


def create_graph():
    """Creates the LangGraph agent."""
    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("router", route_node)
    workflow.add_node("retriever", retrieve_node)
    workflow.add_node("sql_generator", sql_gen_node)
    workflow.add_node("executor", execute_sql_node)
    workflow.add_node("synthesizer", synthesize_node)
    workflow.add_node("validator", validate_node)
    workflow.add_node("repair", repair_node)

    # Edges
    workflow.set_entry_point("router")
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {"retriever": "retriever", "sql_generator": "sql_generator"},
    )
    workflow.add_conditional_edges(
        "retriever",
        after_retrieval_decision,
        {"sql_generator": "sql_generator", "synthesizer": "synthesizer"}
    )
    workflow.add_edge("sql_generator", "executor")
    workflow.add_edge("executor", "synthesizer")
    workflow.add_edge("synthesizer", "validator")
    
    workflow.add_conditional_edges(
        "validator", needs_repair, {"repair": "repair", "__end__": END}
    )
    
    # The repair loop logic
    # For now, a simple repair just re-runs the synthesizer
    workflow.add_edge("repair", "synthesizer")

    return workflow.compile()
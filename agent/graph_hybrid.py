from typing import TypedDict, Any, List, Literal
import dspy
from pathlib import Path

from langgraph.graph import StateGraph, END
# Note: We are importing the *module* classes now, not just the signatures
from agent.dspy_signatures import PlannerModule, RouterModule, SQLGenerator, Synthesizer
from agent.tools.sqlite_tool import SqliteTool
from agent.rag.retrieval import BM25Retriever
import re


# Define the state for our agent
class AgentState(TypedDict):
    question: str
    format_hint: str
    route: str
    constraints: str
    rag_context: List[dict]
    sql_query: str
    sql_results: List[dict]
    final_answer: Any
    citations: List[str]
    errors: List[str]
    feedback: str # New field for repair feedback
    repair_count: int
    sql_tables: List[str] # Extracted tables for citation
    confidence: float
    explanation: str


# Node functions
def route_node(state: AgentState) -> AgentState:
    """Determines the path to take (RAG, SQL, or Hybrid)."""
    print("---ROUTING---")
    
    # Check if LM is configured
    if dspy.settings.lm is None:
        print("WARNING: DSPy LM is not configured. Using fallback routing.")
        # Fallback logic
        if "average order value" in state["question"].lower():
            route = "hybrid"
        elif "policy" in state["question"].lower() or "calendar" in state["question"].lower():
            route = "rag"
        else:
            route = "sql"
    else:
        try:
            router = RouterModule()
            result = router(question=state['question'])
            print(f"  -> Router raw result: {result}")
            # Normalize the output to lowercase and match expected values
            if result.route:
                route = result.route.lower().strip()
                if "rag" in route:
                    route = "rag"
                elif "hybrid" in route:
                    route = "hybrid"
                else:
                    route = "sql"
            else:
                print("  -> Router returned None for route.")
                raise ValueError("Router returned None")
        except Exception as e:
            print(f"Routing failed: {e}. Using fallback.")
            # Fallback logic
            if "average order value" in state["question"].lower():
                route = "hybrid"
            elif "policy" in state["question"].lower() or "calendar" in state["question"].lower():
                route = "rag"
            else:
                route = "sql"

    print(f"  -> Route: {route}")
    state["route"] = route
    # Initialize constraints and other fields
    state["constraints"] = ""
    state["rag_context"] = []
    return state

def retrieve_node(state: AgentState) -> AgentState:
    """Retrieves documents using BM25."""
    print("---RETRIEVING (RAG)---")
    retriever = BM25Retriever()
    state["rag_context"] = retriever.search(query=state["question"])
    return state

def plan_node(state: AgentState) -> AgentState:
    """Extracts constraints from the retrieved context."""
    print("---PLANNING---")
    
    if dspy.settings.lm is None:
        state['constraints'] = "No specific constraints found (LM not configured)."
        return state

    try:
        planner = PlannerModule()
        # Convert rag_context to string for the LLM
        context_str = "\n\n".join([f"Document: {doc['content']}" for doc in state['rag_context']])
        
        result = planner(question=state['question'], rag_context=context_str)
        state['constraints'] = result.constraints
        print(f"  -> Constraints: {state['constraints']}")
    except Exception as e:
        print(f"Planning failed: {e}")
        state['constraints'] = "No specific constraints found (Error)."

    return state

# Globals for DSPy modules, loaded once
_sql_generator_module = None

def _load_sql_generator():
    """
    Helper to instantiate the SQLGenerator module.
    
    Since optimization is failing with the small model, we will bypass loading
    the file and just instantiate a fresh (unoptimized) module.
    """
    global _sql_generator_module
    if _sql_generator_module is None:
        optimized_path = Path("scripts/sql_generator_optimized.json")
        if optimized_path.exists():
            print(f"Loading optimized SQLGenerator from {optimized_path}")
            _sql_generator_module = SQLGenerator()
            try:
                _sql_generator_module.load(str(optimized_path))
            except Exception as e:
                print(f"Failed to load optimized module: {e}. Using fresh module.")
                _sql_generator_module = SQLGenerator()
        else:
            print("Optimized module not found. Instantiating fresh SQLGenerator.")
            _sql_generator_module = SQLGenerator()
    return _sql_generator_module

def sql_gen_node(state: AgentState) -> AgentState:
    """Generates an SQL query from the user's question using the DSPy SQLGenerator."""
    print("---GENERATING SQL---")
    # Ensure DSPy's LM is configured globally or passed appropriately
    if dspy.settings.lm is None:
        print("WARNING: DSPy LM is not configured. SQLGenerator will not function.")
        state['errors'].append("DSPy LM not configured for SQL generation.")
        state['sql_query'] = "SELECT 'Error: LM not configured';"
        return state

    generator = _load_sql_generator()
    
    with SqliteTool() as db:
        schema = db.get_all_schemas()
    
    try:
        # Pass feedback if available (from repair loop)
        result = generator(question=state['question'], schema=schema, feedback=state.get("feedback"), response_format={"type": "text"})
        state['sql_query'] = result.sql_query
        
        if state['sql_query']:
            # Clean the SQL query
            clean_query = state['sql_query'].strip()
            
            # 1. Remove markdown code blocks
            if "```" in clean_query:
                # Find the content inside the first code block
                try:
                    clean_query = clean_query.split("```")[1]
                    if clean_query.startswith("sql"):
                        clean_query = clean_query[3:]
                except IndexError:
                    pass # Fallback to raw string if split fails
            
            # 2. Regex to find the SELECT statement (Case-insensitive, DOTALL)
            # This handles "SQL: SELECT...", "Here is the query: SELECT...", etc.
            match = re.search(r'(SELECT\s+.*)', clean_query, re.IGNORECASE | re.DOTALL)
            if match:
                clean_query = match.group(1)
            
            # 3. Remove any remaining leading/trailing whitespace or garbage
            clean_query = clean_query.strip().lstrip("] \n\t")
            
            # 4. Ensure it ends with a semicolon (and remove multiple semicolons)
            clean_query = clean_query.rstrip(";") + ";"
            
            state['sql_query'] = clean_query
            print(f"  -> Generated SQL: {state['sql_query']}")
        
        if state['sql_query'] is None or state['sql_query'] == "":
            print("  -> DSPy SQL generation returned None or empty.")
            state['errors'].append("DSPy SQL generation returned None or empty.")
            state['sql_query'] = "SELECT 'Error: SQL generation returned None';"
    except Exception as e:
        print(f"  -> DSPy SQL generation error: {e}")
        state['errors'].append(f"DSPy SQL generation failed: {e}")
        state['sql_query'] = "SELECT 'Error: DSPy SQL generation failed';"
    
    return state

def execute_sql_node(state: AgentState) -> AgentState:
    """Executes the generated SQL query and extracts table names for citations."""
    print("---EXECUTING SQL---")
    try:
        with SqliteTool() as db:
            state["sql_results"] = db.execute_sql(state["sql_query"])
        
        # Extract table names from the SQL query for citations
        if state.get("sql_query"):
            # Find table names after FROM and JOIN keywords
            # Pattern matches: FROM TableName, JOIN TableName, FROM "Table Name"
            pattern = r'(?:FROM|JOIN)\s+(?:"([^"]+)"|([^\s,]+))'
            matches = re.findall(pattern, state["sql_query"], re.IGNORECASE)
            
            # Extract table names (matches is list of tuples, either quoted or unquoted)
            table_names = []
            for quoted, unquoted in matches:
                table_name = quoted if quoted else unquoted
                # Remove alias logic is handled by regex for unquoted (stops at space)
                # For quoted, we want the full name (e.g. "Order Details")
                if table_name and table_name.upper() not in ['AS', 'ON', 'WHERE']:
                    table_names.append(table_name)
            
            # Store unique table names in state for citation
            state["sql_tables"] = list(set(table_names))
            
    except Exception as e:
        print(f"  -> SQL execution error: {e}")
        state["errors"].append(str(e))
    return state

def synthesize_node(state: AgentState) -> AgentState:
    """Synthesizes the final answer from the context."""
    print("---SYNTHESIZING ANSWER---")
    
    if dspy.settings.lm is None:
        state["final_answer"] = "I cannot generate an answer because the LLM is not configured."
        state["citations"] = []
        return state

    try:
        synthesizer = Synthesizer()
        
        # Prepare inputs
        context_str = ""
        if state.get('rag_context'):
            context_str = "\n\n".join([
                f"Document ID: {doc.get('id', 'unknown')}\nContent: {doc.get('content', '')}" 
                for doc in state['rag_context'] 
                if isinstance(doc, dict)
            ])
        
        sql_results_str = str(state.get('sql_results', []))
        
        result = synthesizer(
            question=state['question'], 
            format_hint=state['format_hint'], 
            constraints=state['constraints'], 
            rag_context=context_str, 
            sql_results=sql_results_str,
            feedback=state.get("feedback")
        )
        
        print(f"  -> Synthesizer raw result: {result}")
        
        state["final_answer"] = getattr(result, 'final_answer', "No answer generated.")
        
        # Extract confidence and explanation
        try:
            state["confidence"] = float(getattr(result, 'confidence', 0.0))
        except (ValueError, TypeError):
            state["confidence"] = 0.0
            
        state["explanation"] = getattr(result, 'explanation', "")
        
        # Combine RAG citations with SQL table citations
        rag_citations = getattr(result, 'citations', [])
        sql_tables = state.get('sql_tables', [])
        
        # Format: [doc citations] + [SQL: table1, table2, ...]
        all_citations = []
        
        # Handle RAG citations (could be string or list)
        if rag_citations:
            if isinstance(rag_citations, str):
                # If it's a string representation of a list, try to clean it
                cleaned = rag_citations.strip("[]'\" ")
                if cleaned:
                    all_citations.append(cleaned)
            elif isinstance(rag_citations, list):
                all_citations.extend(rag_citations)
                
        if sql_tables:
            tables_str = f"SQL: {', '.join(sql_tables)}"
            all_citations.append(tables_str)
        
        state["citations"] = all_citations if all_citations else []
        
        print(f"  -> Answer: {str(state['final_answer'])[:100]}...")
    except Exception as e:
        print(f"Synthesis failed: {e}")
        state["final_answer"] = "I encountered an error while generating the answer."
        state["citations"] = []
        state["errors"].append(f"Synthesis failed: {e}")

    return state

def validate_node(state: AgentState) -> AgentState:
    """Validates the output and checks for the need for repair."""
    print("---VALIDATING---")
    
    # Existing error check
    if state.get("errors") and len(state["errors"]) > 0:
        state["repair_count"] = state.get("repair_count", 0) + 1
        return state

    # Format Validation
    answer = state.get("final_answer")
    hint = state.get("format_hint", "").lower()
    
    if not answer:
        state["errors"].append("Final answer is empty.")
    elif hint:
        try:
            if "int" in hint:
                # Try to find an integer in the string if it's not purely an int
                import re
                if not re.search(r'\d+', str(answer)):
                     state["errors"].append(f"Answer '{answer}' does not contain an integer as required by hint '{hint}'.")
            elif "float" in hint:
                import re
                if not re.search(r'\d+\.\d+', str(answer)) and not re.search(r'\d+', str(answer)):
                    state["errors"].append(f"Answer '{answer}' does not contain a float/number as required by hint '{hint}'.")
            elif "list" in hint:
                if not (str(answer).strip().startswith("[") and str(answer).strip().endswith("]")):
                     state["errors"].append(f"Answer '{answer}' does not look like a list as required by hint '{hint}'.")
            elif "json" in hint or "{" in hint:
                if not (str(answer).strip().startswith("{") and str(answer).strip().endswith("}")):
                     state["errors"].append(f"Answer '{answer}' does not look like a JSON object as required by hint '{hint}'.")
        except Exception as e:
            state["errors"].append(f"Validation failed: {e}")

    if state.get("errors") and len(state["errors"]) > 0:
        print(f"  -> Validation failed: {state['errors']}")
        state["repair_count"] = state.get("repair_count", 0) + 1
    else:
        print("  -> Validation passed.")
        
    return state

def repair_node(state: AgentState) -> AgentState:
    """Repairs the query or answer based on errors."""
    print(f"---REPAIRING (Attempt {state.get('repair_count', 1)})---")
    
    # Move errors to feedback so the next node can see them
    if state.get("errors"):
        state["feedback"] = f"Previous attempt failed with errors: {'; '.join(state['errors'])}"
    else:
        state["feedback"] = "Previous attempt failed."
        
    state["errors"] = []
    return state


# Conditional edge logic
def route_decision(state: AgentState) -> Literal["retriever", "sql_generator"]:
    """Decides the next step after the router."""
    if state["route"] in ("rag", "hybrid"):
        return "retriever"
    else: # SQL-only
        return "sql_generator"

def after_retrieval_decision(state: AgentState) -> Literal["planner"]:
    """After retrieval, always go to the planner."""
    return "planner"

def after_planner_decision(state: AgentState) -> Literal["sql_generator", "synthesizer"]:
    """After planning, decide whether to generate SQL or synthesize."""
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


def repair_routing(state: AgentState) -> Literal["sql_generator", "synthesizer"]:
    """Decides where to go after repair."""
    feedback = state.get("feedback", "").lower()
    # If the feedback contains SQL errors, go back to SQL generation
    if "sql" in feedback or "execution" in feedback or "database" in feedback or "syntax" in feedback:
        return "sql_generator"
    # Otherwise (e.g., format errors), go to synthesizer
    return "synthesizer"


def create_graph():
    """Creates the LangGraph agent."""
    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("router", route_node)
    workflow.add_node("retriever", retrieve_node)
    workflow.add_node("planner", plan_node)
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
    workflow.add_edge("retriever", "planner")
    
    workflow.add_conditional_edges(
        "planner",
        after_planner_decision,
        {"sql_generator": "sql_generator", "synthesizer": "synthesizer"},
    )
    
    workflow.add_edge("sql_generator", "executor")
    workflow.add_edge("executor", "synthesizer")
    workflow.add_edge("synthesizer", "validator")
    
    workflow.add_conditional_edges(
        "validator", needs_repair, {"repair": "repair", "__end__": END}
    )
    
    # New conditional edge from repair
    workflow.add_conditional_edges(
        "repair",
        repair_routing,
        {"sql_generator": "sql_generator", "synthesizer": "synthesizer"}
    )

    return workflow.compile()

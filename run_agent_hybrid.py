from agent.graph_hybrid import create_graph, AgentState
import logging

# Set up basic logging to see outputs from the agent's tools
logging.basicConfig(level=logging.INFO)

def main():
    """
    Main function to create and run the agent graph with sample inputs.
    This serves as a basic integration test to see the flow of the agent.
    """
    app = create_graph()

    # --- Test Case 1: RAG Route ---
    print("\n" + "="*30)
    print("--- RUNNING TEST 1: RAG ---")
    print("="*30)
    question_rag = "what is the return policy for beverages?"
    inputs_rag = {
        "question": question_rag,
        "format_hint": "A simple string answer.",
        "errors": [],
        "repair_count": 0,
    }
    # .stream() is also useful for seeing the flow live
    for s in app.stream(inputs_rag):
        print(s)
        print("----")

    # --- Test Case 2: SQL Route ---
    print("\n" + "="*30)
    print("--- RUNNING TEST 2: SQL ---")
    print("="*30)
    question_sql = "what are the top 3 selling products?"
    inputs_sql = {
        "question": question_sql,
        "format_hint": "A list of product names.",
        "errors": [],
        "repair_count": 0,
    }
    for s in app.stream(inputs_sql):
        print(s)
        print("----")


    # --- Test Case 3: Hybrid Route ---
    print("\n" + "="*30)
    print("--- RUNNING TEST 3: HYBRID ---")
    print("="*30)
    question_hybrid = "what was the average order value during the 'Winter Classics 1997' promotion?"
    inputs_hybrid = {
        "question": question_hybrid,
        "format_hint": "A float value.",
        "errors": [],
        "repair_count": 0,
    }
    for s in app.stream(inputs_hybrid):
        print(s)
        print("----")


if __name__ == "__main__":
    main()

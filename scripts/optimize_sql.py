import dspy
import json
from dspy.teleprompt import BootstrapFewShot
from agent.dspy_signatures import SQLGenerator
from agent.tools.sqlite_tool import SqliteTool
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_training_set(file_path: str, db_schema: str) -> list[dspy.Example]:
    """Loads the hand-written SQL training data and formats it as dspy.Example objects."""
    training_set = []
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            example = dspy.Example(
                question=data["question"],
                schema=db_schema,
                query=data["query"]  # This is the ground truth
            ).with_inputs("question", "schema") # Removed constraints
            training_set.append(example)
    logging.info(f"Loaded {len(training_set)} examples from {file_path}")
    return training_set


def execution_accuracy(example, pred, trace=None):
    """
    A DSPy metric that checks if the predicted SQL query executes successfully.
    """
    predicted_query = pred.sql_query
    
    # Check if pred.sql_query is None or empty, which can happen if LLM fails
    if not predicted_query or not isinstance(predicted_query, str):
        logging.error(f"FAILED: Predicted query is invalid or empty. Query: {predicted_query}")
        return False

    try:
        with SqliteTool() as db:
            db.execute_sql(predicted_query)
        logging.info(f"SUCCESS: Query executed successfully.\n  -> Query: {predicted_query}")
        return True  # The query executed without errors
    except Exception as e:
        logging.error(f"FAILED: Query execution failed: {e}\n  -> Query: {predicted_query}")
        return False # The query failed


def main():
    """
    Main function to run the DSPy optimization process for the SQLGenerator module.
    """
    # --- 1. Configure DSPy ---
    try:
        # User's configured Ollama LM
        ollama_model = dspy.LM(
            "devstral-small-2507",
            # "phi3.5:3.8b-mini-instruct-q4_K_M",
            api_base="http://127.0.0.1:1234/v1",
            api_key="lm-studio",
            model_type='text')
        dspy.configure(lm=ollama_model)
        logging.info("DSPy configured with phi3.5 Ollama model.")
    except Exception as e:
        logging.error(f"Could not configure DSPy. Please ensure your API keys or model setup is correct. Error: {e}")
        logging.warning("Proceeding without a configured LLM. The script will fail if an LLM call is made.")
        return

    # --- 2. Load Data ---
    with SqliteTool() as db:
        db_schema = db.get_all_schemas()
    
    trainset = load_training_set("data/sql_training_set.jsonl", db_schema)
    
    # --- Add manual initial demos to guide the LLM ---
    # These examples include the 'reasoning' field to provide a ChainOfThought example
    manual_demos = [
        dspy.Example(
            question="Top 3 products by total revenue all-time.",
            schema=db_schema,
            # Removed constraints
            query="SELECT T2.ProductName, SUM(T1.UnitPrice * T1.Quantity * (1 - T1.Discount)) AS TotalRevenue FROM 'Order Details' AS T1 INNER JOIN Products AS T2 ON T1.ProductID = T2.ProductID GROUP BY T2.ProductName ORDER BY TotalRevenue DESC LIMIT 3;"
        ).with_inputs("question", "schema"), # Removed constraints
        dspy.Example(
            question="Total revenue from the 'Beverages' category during 'Summer Beverages 1997' dates.",
            schema=db_schema,
            # Removed constraints
            query="SELECT SUM(T2.UnitPrice * T2.Quantity * (1 - T2.Discount)) FROM Orders AS T1 INNER JOIN 'Order Details' AS T2 ON T1.OrderID = T2.OrderID INNER JOIN Products AS T3 ON T2.ProductID = T3.ProductID INNER JOIN Categories AS T4 ON T3.CategoryID = T4.CategoryID WHERE T1.OrderDate BETWEEN '1997-06-01' AND '1997-06-30' AND T4.CategoryName = 'Beverages';"
        ).with_inputs("question", "schema"), # Removed constraints
    ]
    
    # Prepend manual demos to the training set
    trainset = manual_demos + trainset
    logging.info(f"Augmented training set with {len(manual_demos)} manual demos. Total examples: {len(trainset)}")

    # --- 3. Set up the Optimizer (Teleprompter) ---
    # Increase max_bootstrapped_demos to give the LLM more chances to find good examples.
    # Set max_rounds to 1 for now to just try and get initial traces.
    teleprompter = BootstrapFewShot(metric=execution_accuracy, max_bootstrapped_demos=8, max_rounds=2)
    
    # --- 4. Run the Compilation ---
    logging.info("Starting DSPy compilation... This may take a few minutes depending on your model and data.")
    try:
        optimized_sql_generator = teleprompter.compile(SQLGenerator(), trainset=trainset)
    except Exception as e:
        logging.error(f"DSPy compilation failed. This usually happens if the LLM is not configured correctly or is unreachable, or if it consistently fails to produce valid outputs. Error: {e}", exc_info=True)
        return
        
    # --- 5. Save the Optimized Module ---
    output_path = "sql_generator_optimized.json"
    optimized_sql_generator.save(output_path)
    logging.info(f"Successfully compiled and saved the optimized module to {output_path}")


if __name__ == "__main__":
    main()

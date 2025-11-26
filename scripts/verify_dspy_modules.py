import dspy
import logging
from agent.dspy_signatures import RouterModule, PlannerModule, SQLGenerator, Synthesizer
from agent.tools.sqlite_tool import SqliteTool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_dspy():
    try:
        from dotenv import load_dotenv
        import os
        load_dotenv()
        
        api_base = os.getenv("LLM_API_BASE", "http://127.0.0.1:1234/v1")
        api_key = os.getenv("LLM_API_KEY", "lm-studio")
        model_name = os.getenv("LLM_MODEL_NAME", "devstral-small-2507")

        # User's configured Ollama LM
        ollama_model = dspy.LM(
            model_name,
            api_base=api_base,
            api_key=api_key,
            model_type='text')
        dspy.configure(lm=ollama_model)
        logger.info(f"DSPy configured with model: {model_name}")
        return True
    except Exception as e:
        logger.error(f"Could not configure DSPy: {e}")
        return False

def test_router():
    print("\n--- Testing Router ---")
    router = RouterModule()
    questions = [
        "what is the return policy?",
        "what are the top 3 products?",
        "average order value during the promotion"
    ]
    for q in questions:
        try:
            print(f"Q: {q}")
            result = router(question=q)
            print(f"A: {result.route} (Reasoning: {getattr(result, 'reasoning', 'None')})")
        except Exception as e:
            print(f"Error: {e}")

def test_planner():
    print("\n--- Testing Planner ---")
    planner = PlannerModule()
    question = "average order value during 'Summer 2024'"
    context = "Document: Promotion 'Summer 2024' ran from 2024-06-01 to 2024-08-31."
    try:
        print(f"Q: {question}")
        print(f"Context: {context}")
        result = planner(question=question, rag_context=context)
        print(f"Constraints: {result.constraints}")
    except Exception as e:
        print(f"Error: {e}")

def test_sql_generator():
    print("\n--- Testing SQL Generator ---")
    generator = SQLGenerator()
    try:
        generator.load("scripts/sql_generator_optimized.json")
        print("Loaded optimized SQL generator.")
    except:
        print("Using fresh SQL generator.")

    question = "How many products are there?"
    with SqliteTool() as db:
        schema = db.get_table_schema("Products")
    
    try:
        print(f"Q: {question}")
        result = generator(question=question, schema=schema)
        print(f"SQL: {result.sql_query}")
    except Exception as e:
        print(f"Error: {e}")

def test_synthesizer():
    print("\n--- Testing Synthesizer ---")
    synthesizer = Synthesizer()
    question = "How many products?"
    result_sql = "[{'Count': 77}]"
    try:
        print(f"Q: {question}")
        result = synthesizer(
            question=question,
            format_hint="A number",
            constraints="",
            rag_context="",
            sql_results=result_sql
        )
        print(f"Answer: {result.final_answer}")
        print(f"Citations: {result.citations}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if setup_dspy():
        test_router()
        test_planner()
        test_sql_generator()
        test_synthesizer()

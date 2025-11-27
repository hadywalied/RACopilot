import click
import json
from agent.graph_hybrid import create_graph, AgentState
import logging
import dspy
from pathlib import Path

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@click.command()
@click.option('--batch', type=click.Path(exists=True), required=True, help='Path to input JSONL file.')
@click.option('--out', type=click.Path(), required=True, help='Path to output JSONL file.')
def main(batch, out):
    """
    Run the Retail Analytics Copilot on a batch of questions.
    """
    # --- Configure DSPy Language Model ---
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
            model_type='chat',
            response_format={"type": "text"}
        )
        dspy.configure(lm=ollama_model)
        logging.info(f"DSPy configured with model: {model_name}")
    except Exception as e:
        logging.error(f"Could not configure DSPy LM: {e}")
        return

    # Create the agent graph
    app = create_graph()
    
    input_path = Path(batch)
    output_path = Path(out)
    
    print(f"DEBUG: Input path: {input_path}")
    print(f"DEBUG: Output path: {output_path}")
    
    logging.info(f"Processing {input_path} -> {output_path}")
    
    results = []
    
    if not input_path.exists():
        print(f"ERROR: Input file {input_path} does not exist.")
        return

    try:
        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:
            
            print("DEBUG: Files opened successfully.")
            
            for line in f_in:
                print(f"DEBUG: Reading line: {line[:50]}...")
                if not line.strip():
                    continue
                    
                data = json.loads(line)
                question = data.get("question")
                format_hint = data.get("format_hint", "A string answer.")
                
                logging.info(f"Processing question: {question}")
                
                inputs = {
                    "question": question,
                    "format_hint": format_hint,
                    "errors": [],
                    "repair_count": 0,
                }
                
                try:
                    # Run the agent
                    # We use invoke instead of stream for batch processing to get the final state
                    final_state = app.invoke(inputs)
                    
                    output_data = {
                        "id": data.get("id", "unknown"),
                        "final_answer": final_state.get("final_answer"),
                        "sql": final_state.get("sql_query", ""),
                        "confidence": final_state.get("confidence", 0.0),
                        "explanation": final_state.get("explanation", ""),
                        "citations": final_state.get("citations", [])
                    }
                    
                    # Write result immediately
                    f_out.write(json.dumps(output_data) + "\n")
                    f_out.flush()
                    
                except Exception as e:
                    logging.error(f"Error processing question '{question}': {e}")
                    error_data = {
                        "question": question,
                        "error": str(e)
                    }
                    f_out.write(json.dumps(error_data) + "\n")
                    f_out.flush()

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        logging.error(f"Critical error during batch processing: {e}")

    logging.info("Batch processing complete.")

if __name__ == "__main__":
    main()

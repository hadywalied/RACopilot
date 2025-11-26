import dspy
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_ollama_connection():
    try:
        # User's configured Ollama LM
        ollama_model = dspy.LM(
            "phi3.5:3.8b-mini-instruct-q4_K_M",
            api_base="http://127.0.0.1:11434/v1",
            api_key="ollama",
            model_type='text')
        dspy.configure(lm=ollama_model)
        logging.info("DSPy configured with phi3.5 Ollama model.")

        # Define a simple DSPy Signature
        class SimpleTask(dspy.Signature):
            question = dspy.InputField(desc="A simple question")
            answer = dspy.OutputField(desc="A concise answer")

        # Create a dspy.Predict module using the signature
        predict_module = dspy.Predict(SimpleTask)

        # Make a simple prediction
        question = "What is the capital of France?"
        logging.info(f"Attempting prediction for: '{question}'")
        
        # Increased timeout might be necessary for local LLMs
        with dspy.settings.context(lm=ollama_model, bypass_lm_price_check=True):
             response = predict_module(question=question)
        
        logging.info(f"Prediction successful! Answer: {response.answer}")

    except Exception as e:
        logging.error(f"Failed to connect to Ollama or make a prediction. Error: {e}", exc_info=True)
        logging.error("Please ensure:")
        logging.error("1. Ollama server is running on http://127.0.0.1:11434.")
        logging.error("2. The model 'phi3.5:3.8b-mini-instruct-q4_K_M' is pulled (`ollama pull phi3.5:3.8b-mini-instruct-q4_K_M`).")
        logging.error("3. The model name in the script matches the pulled model name exactly.")

if __name__ == "__main__":
    test_ollama_connection()

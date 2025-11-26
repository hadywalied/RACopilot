import dspy


class RouteQuery(dspy.Signature):
    """Classify the user's question into one of three routes: 'rag', 'sql', or 'hybrid'.
    
    Routes:
    - 'rag': Questions about policies, definitions, textual information, or general knowledge (e.g., "return policy", "who is the CEO").
    - 'sql': Questions requiring calculation, aggregation, or retrieval of structured data from the database (e.g., "total sales", "top products", "count of orders").
    - 'hybrid': Questions that need both specific constraints from text AND data from the database (e.g., "average order value during the 'Summer Promotion'").
    """

    question = dspy.InputField(desc="The user's question.")
    route = dspy.OutputField(desc="The classification: 'rag', 'sql', or 'hybrid'.")


class Planner(dspy.Signature):
    """Analyze the user's question and the retrieved text context to extract specific constraints for a database query.
    
    Focus on:
    - Date ranges (start and end dates).
    - Specific product names or categories mentioned in the text.
    - KPI definitions (how to calculate something).
    
    If no constraints are found in the text, state "No specific constraints found."
    """
    question = dspy.InputField(desc="The user's question.")
    rag_context = dspy.InputField(desc="Retrieved document chunks that may contain definitions or dates.")
    constraints = dspy.OutputField(desc="A concise summary of the extracted constraints.")


class NL2SQL(dspy.Signature):
    """Generate a valid SQLite query to answer the question.
    
    CRITICAL INSTRUCTIONS:
    1. Output ONLY the raw SQL query. Do NOT use markdown code blocks (```sql). Do NOT add explanations.
    2. Start the query immediately with 'SELECT'.
    3. Use the provided schema.
    4. Use double quotes "Table Name" for tables/columns with spaces.
    5. For dates, use SQLite's STRFTIME format.
    """
    question = dspy.InputField(desc="The question to be answered via SQL.")
    schema = dspy.InputField(desc="The SQLite database schema.")
    feedback = dspy.InputField(desc="Error message from previous SQL execution attempt.", optional=True)
    sql_query = dspy.OutputField(desc="The raw SQL query string. Nothing else.")


class SynthesizeAnswer(dspy.Signature):
    """Synthesize a final answer based on the provided context and SQL results.
    
    Guidelines:
    - Be direct and professional.
    - Use the SQL results to provide exact numbers.
    - Use the RAG context to explain policies or definitions.
    - Cite sources using the format [SourceID].
    - If the SQL result is empty or error, explain that data could not be retrieved.
    """

    question = dspy.InputField(desc="The original question.")
    format_hint = dspy.InputField(desc="Hint about the expected answer format.")
    constraints = dspy.InputField(desc="Constraints used in the process.")
    rag_context = dspy.InputField(desc="Text context used.")
    sql_results = dspy.InputField(desc="Data returned from the database.")
    feedback = dspy.InputField(desc="Feedback from previous attempts (e.g., format errors).", optional=True)
    final_answer = dspy.OutputField(desc="The comprehensive answer.")
    citations = dspy.OutputField(desc="List of document IDs used as sources.")


# --- DSPy Modules ---

class RouterModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought(RouteQuery)

    def forward(self, question):
        return self.classify(question=question)


class PlannerModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(Planner)

    def forward(self, question, rag_context):
        return self.extract(question=question, rag_context=rag_context)


class SQLGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        # Simplified to dspy.Predict for direct SQL string output
        self.generate = dspy.Predict(NL2SQL)

    def forward(self, question, schema, feedback=None):
        return self.generate(question=question, schema=schema, feedback=feedback or "")


class Synthesizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(SynthesizeAnswer)

    def forward(self, question, format_hint, constraints, rag_context, sql_results, feedback=None):
        return self.generate(
            question=question,
            format_hint=format_hint,
            constraints=constraints,
            rag_context=rag_context,
            sql_results=sql_results,
            feedback=feedback or ""
        )
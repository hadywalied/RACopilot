import dspy


class RouteQuery(dspy.Signature):
    """Classify if query needs: rag, sql, or hybrid"""

    question = dspy.InputField(desc="User question")
    route = dspy.OutputField(desc="One of: rag, sql, hybrid")


class Planner(dspy.Signature):
    """From a user question and retrieved context, extract constraints for SQL generation.
    
    This includes date ranges, KPI formulas, categories, or other entities.
    """
    question = dspy.InputField(desc="User question")
    rag_context = dspy.InputField(desc="Retrieved document chunks with potential constraints")
    constraints = dspy.OutputField(desc="A summary of constraints like dates, KPIs, or categories.")


class NL2SQL(dspy.Signature):
    """Convert natural language to *correct and runnable* SQLite query.
- IMPORTANT: ALWAYS use double quotes around table and column names that contain spaces (e.g., "Order Details").
- Use 'AS T1', 'AS T2', etc. for table aliases.
- For date comparisons, prefer STRFTIME('%Y-%m-%d', date_column) for full dates, or STRFTIME('%Y', date_column) for years.
- Ensure the query is valid SQLite syntax. Avoid non-standard SQL keywords (like ILIKE if not supported)."""

    question = dspy.InputField()
    schema = dspy.InputField(desc="Database schema")
    sql_query = dspy.OutputField(desc="Valid SQLite query")


class SynthesizeAnswer(dspy.Signature):
    """Produce typed answer with citations"""

    question = dspy.InputField()
    format_hint = dspy.InputField(desc="Expected output type")
    constraints = dspy.InputField(desc="Summary of constraints that were applied")
    rag_context = dspy.InputField(desc="Retrieved document chunks")
    sql_results = dspy.InputField(desc="SQL execution results")
    final_answer = dspy.OutputField(desc="Answer matching format_hint")
    citations = dspy.OutputField(desc="List of sources used")


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
        # Changed to dspy.Predict for simpler output expected from LLM
        self.generate = dspy.ChainOfThought(NL2SQL)
        # self.generate = dspy.Predict(NL2SQL)

    def forward(self, question, schema): # Removed constraints
        return self.generate(question=question, schema=schema) # Removed constraints


class Synthesizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(SynthesizeAnswer)

    def forward(self, question, format_hint, constraints, rag_context, sql_results):
        return self.generate(
            question=question,
            format_hint=format_hint,
            constraints=constraints,
            rag_context=rag_context,
            sql_results=sql_results,
        )

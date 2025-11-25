import dspy


class RouteQuery(dspy.Signature):
    """Classify if query needs: rag, sql, or hybrid"""

    question = dspy.InputField(desc="User question")
    route = dspy.OutputField(desc="One of: rag, sql, hybrid")


class NL2SQL(dspy.Signature):
    """Convert natural language to SQL"""

    question = dspy.InputField()
    schema = dspy.InputField(desc="Database schema")
    sql_query = dspy.OutputField(desc="Valid SQLite query")


class SynthesizeAnswer(dspy.Signature):
    """Produce typed answer with citations"""

    question = dspy.InputField()
    format_hint = dspy.InputField(desc="Expected output type")
    rag_context = dspy.InputField(desc="Retrieved document chunks")
    sql_results = dspy.InputField(desc="SQL execution results")
    final_answer = dspy.OutputField(desc="Answer matching format_hint")
    citations = dspy.OutputField(desc="List of sources used")


class RouterModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought(RouteQuery)

    def forward(self, question):
        return self.classify(question=question)


class SQLGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(NL2SQL)

    def forward(self, question, schema):
        return self.generate(question=question, schema=schema)


class Synthesizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(SynthesizeAnswer)

    def forward(self, question, format_hint, rag_context, sql_results):
        return self.generate(
            question=question,
            format_hint=format_hint,
            rag_context=rag_context,
            sql_results=sql_results,
        )
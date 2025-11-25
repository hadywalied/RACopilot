import pytest
from pathlib import Path
from agent.rag.retrieval import BM25Retriever

@pytest.fixture
def temp_docs_dir(tmp_path):
    """Fixture to create a temporary directory with dummy markdown files."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    
    # Document with two chunks (header and paragraph)
    (docs_dir / "doc1.md").write_text("# Document 1\n\nThis is the first document about blue things.")
    
    # Document that will be split into three chunks
    (docs_dir / "doc2.md").write_text("# Document 2\n\nThis is a document about yellow things.\n\nIt has a second paragraph about a yellow banana.")
    
    # Document with two chunks
    (docs_dir / "doc3.md").write_text("# Document 3\n\nThis document is about red cars.")
    
    return docs_dir

def test_bm25_retriever_initialization_and_chunking(temp_docs_dir):
    """Test that the BM25 retriever loads and chunks documents correctly."""
    retriever = BM25Retriever(docs_path=str(temp_docs_dir))
    
    # We expect 7 chunks in total: 2 from doc1, 3 from doc2, 2 from doc3
    assert len(retriever.documents) == 7
    assert len(retriever.doc_ids) == 7
    
    expected_ids = [
        "doc1::chunk0", "doc1::chunk1",
        "doc2::chunk0", "doc2::chunk1", "doc2::chunk2",
        "doc3::chunk0", "doc3::chunk1"
    ]
    assert sorted(retriever.doc_ids) == sorted(expected_ids)

    # Check content of a specific chunk
    doc2_chunk1_index = retriever.doc_ids.index("doc2::chunk1")
    assert retriever.documents[doc2_chunk1_index] == "This is a document about yellow things."

def test_bm25_retriever_search_top_k(temp_docs_dir):
    """Test that the search function returns the correct number of results."""
    retriever = BM25Retriever(docs_path=str(temp_docs_dir))
    
    # Search for "banana" to uniquely identify the correct chunk
    results = retriever.search("banana", top_k=2)
    assert len(results) == 1
    
    # The first result should be the chunk containing "banana"
    assert results[0]["id"] == "doc2::chunk2"
    assert "banana" in results[0]["content"]

def test_bm25_retriever_search_content_and_score(temp_docs_dir):
    """Test the content and score of search results."""
    retriever = BM25Retriever(docs_path=str(temp_docs_dir))
    
    # Search for "blue"
    results = retriever.search("blue", top_k=1)
    assert len(results) == 1
    
    result = results[0]
    # The content is in chunk1 because chunk0 is the markdown header
    assert result["id"] == "doc1::chunk1"
    assert "blue" in result["content"]
    assert isinstance(result["score"], float)
    assert result["score"] > 0

def test_bm25_retriever_search_no_match(temp_docs_dir):
    """Test a query that should not match any document well."""
    retriever = BM25Retriever(docs_path=str(temp_docs_dir))
    
    results = retriever.search("unrelated topic like elephants")
    # BM25 scores for non-matching terms are often 0, so we expect an empty list
    assert len(results) == 0

def test_bm25_retriever_empty_docs_directory(tmp_path):
    """Test that the retriever handles an empty docs directory gracefully."""
    empty_docs_dir = tmp_path / "empty_docs"
    empty_docs_dir.mkdir()
    
    retriever = BM25Retriever(docs_path=str(empty_docs_dir))
    assert retriever.bm25 is None
    assert len(retriever.documents) == 0
    
    # Search should return an empty list
    results = retriever.search("any query")
    assert results == []

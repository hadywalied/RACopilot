from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from typing import List, Dict, Any
import logging
import re
from rank_bm25 import BM25Okapi

# Set up logging
logger = logging.getLogger(__name__)

class TfidfRetriever:
    """
    A TF-IDF based retriever that loads documents, splits them into
    paragraph-level chunks, and performs a similarity search.
    """
    def __init__(self, docs_path: str = "docs"):
        self.docs_path = Path(__file__).parent.parent.parent / docs_path
        self.documents: List[str] = []
        self.doc_ids: List[str] = []
        
        self._load_and_chunk_documents()
        
        self.vectorizer = TfidfVectorizer()
        if self.documents:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
            logger.info(f"TfidfRetriever initialized with {len(self.documents)} chunks from {self.docs_path}.")
        else:
            logger.warning("No documents were loaded.")
            self.tfidf_matrix = None

    def _load_and_chunk_documents(self):
        """
        Loads documents from the specified path and splits them into
        paragraph-level chunks.
        """
        if not self.docs_path.exists() or not self.docs_path.is_dir():
            logger.error(f"Docs path does not exist or is not a directory: {self.docs_path}")
            return
            
        for filepath in self.docs_path.glob("*.md"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Split by two or more newlines (handles different OS line endings)
                    chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
                    
                    for i, chunk in enumerate(chunks):
                        self.documents.append(chunk)
                        # The ID is filename::chunk<index>, as per spec
                        self.doc_ids.append(f"{filepath.stem}::chunk{i}")
            except Exception as e:
                logger.error(f"Failed to read or process file {filepath}: {e}")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Performs a TF-IDF search for the given query.
        """
        if self.tfidf_matrix is None:
            return []
            
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Get top_k results, ensuring we don't go out of bounds
        k = min(top_k, len(self.documents))
        if k == 0:
            return []
        
        # Get indices of top_k scores, handling the case where k=0 or k > num_docs
        top_k_indices = similarities.argsort()[-k:][::-1]

        results = [
            {
                "id": self.doc_ids[i],
                "content": self.documents[i],
                "score": similarities[i],
            }
            for i in top_k_indices if similarities[i] > 0
        ]
        return results


class BM25Retriever:
    """
    A BM25 based retriever that loads documents, splits them into
    paragraph-level chunks, and performs a similarity search.
    """
    def __init__(self, docs_path: str = "docs"):
        self.docs_path = Path(__file__).parent.parent.parent / docs_path
        self.documents: List[str] = []
        self.doc_ids: List[str] = []
        
        self._load_and_chunk_documents()
        
        if self.documents:
            # BM25 requires tokenized documents
            tokenized_corpus = [self._tokenize(doc) for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info(f"BM25Retriever initialized with {len(self.documents)} chunks from {self.docs_path}.")
        else:
            logger.warning("No documents were loaded. Retriever will not find any results.")
            self.bm25 = None

    def _tokenize(self, text: str) -> List[str]:
        """
        A simple tokenizer that lowercases, removes non-alphanumeric characters,
        and splits by space.
        """
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        return text.split()

    def _load_and_chunk_documents(self):
        """
        Loads documents from the specified path and splits them into
        paragraph-level chunks.
        """
        if not self.docs_path.exists() or not self.docs_path.is_dir():
            logger.error(f"Docs path does not exist or is not a directory: {self.docs_path}")
            return
            
        for filepath in self.docs_path.glob("*.md"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Split by two or more newlines (handles different OS line endings)
                    chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
                    
                    for i, chunk in enumerate(chunks):
                        self.documents.append(chunk)
                        # The ID is filename::chunk<index>, as per spec
                        self.doc_ids.append(f"{filepath.stem}::chunk{i}")
            except Exception as e:
                logger.error(f"Failed to read or process file {filepath}: {e}")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Performs a BM25 search for the given query.
        """
        if self.bm25 is None:
            return []
        
        tokenized_query = self._tokenize(query)
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top_k results, ensuring we don't go out of bounds
        k = min(top_k, len(self.documents))
        if k == 0:
            return []
        
        # Get indices of top_k scores
        top_k_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:k]

        results = [
            {
                "id": self.doc_ids[i],
                "content": self.documents[i],
                "score": doc_scores[i],
            }
            for i in top_k_indices if doc_scores[i] > 0
        ]
        return results

# Example Usage:
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO)
#     
#     # TF-IDF
#     tfidf_retriever = TfidfRetriever()
#     results_tfidf = tfidf_retriever.search("what is the return policy for beverages?")
#     print("TF-IDF Results:", results_tfidf)
#
#     # BM25
#     bm25_retriever = BM25Retriever()
#     results_bm25 = bm25_retriever.search("what is the return policy for beverages?")
#     print("BM25 Results:", results_bm25)

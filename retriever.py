from sentence_transformers import SentenceTransformer
from database import ECGDatabase

class KnowledgeRetriever:

    def __init__(self):

        self.db = ECGDatabase()

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def search(
        self,
        query,
        top_k=8
    ):

        embedding = self.model.encode(
            query,
            normalize_embeddings=True
        )

        return self.db.search_knowledge(
            embedding,
            top_k
        )
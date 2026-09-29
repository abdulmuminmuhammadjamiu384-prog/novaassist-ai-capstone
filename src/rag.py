import os
import json
import math
import re
from collections import Counter

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "novapay_kb.json")
STOP_WORDS = {
    "what", "is", "the", "a", "an", "in", "on", "to", "do", "does",
    "can", "i", "who", "are", "for", "of", "and", "or", "novapay",
    "today", "current", "system", "tell", "me", "how", "long"
}

def clean_token(token: str) -> str:
    token = token.lower()
    if token.endswith("ments"):
        return token[:-1]
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("es") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token

def tokenize(text: str):
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [clean_token(t) for t in tokens if t not in STOP_WORDS]

def load_knowledge_base(file_path: str = DATA_PATH):
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8-sig") as f:
        return json.load(f)

class RAGEngine:
    def __init__(self, kb_path: str = DATA_PATH):
        self.documents = load_knowledge_base(kb_path)
        self.idf = {}
        self.doc_vectors = []
        self._build_index()

    def _build_index(self):
        if not self.documents:
            return

        tokenized_corpus = [tokenize(f"{d['title']} {d['content']}") for d in self.documents]
        n_docs = len(self.documents)

        df = Counter()
        for doc in tokenized_corpus:
            df.update(set(doc))

        self.idf = {term: math.log((1 + n_docs) / (1 + count)) + 1 for term, count in df.items()}

        self.doc_vectors = []
        for doc in tokenized_corpus:
            tf = Counter(doc)
            total = len(doc) or 1
            vec = {term: (tf[term] / total) * self.idf[term] for term in doc}
            self.doc_vectors.append(vec)

    def retrieve(self, query: str, top_k: int = 1):
        tokens = tokenize(query)
        if not tokens or not self.doc_vectors:
            return []

        tf = Counter(tokens)
        total = len(tokens) or 1
        query_vec = {t: (tf[t] / total) * self.idf.get(t, 0) for t in tokens if t in self.idf}

        scores = []
        q_norm = math.sqrt(sum(v ** 2 for v in query_vec.values()))

        for idx, d_vec in enumerate(self.doc_vectors):
            common = set(query_vec.keys()) & set(d_vec.keys())
            if not common or q_norm == 0:
                continue
            dot = sum(query_vec[t] * d_vec[t] for t in common)
            d_norm = math.sqrt(sum(v ** 2 for v in d_vec.values()))
            similarity = dot / (q_norm * d_norm) if d_norm > 0 else 0
            scores.append((similarity, len(common), self.documents[idx]))

        scores.sort(key=lambda x: (x[0], x[1]), reverse=True)

        # Strictly requires at least 2 distinct matching terms and >= 0.10 similarity
        valid_matches = [
            doc for score, common_count, doc in scores
            if common_count >= 2 and score >= 0.10
        ]
        return valid_matches[:top_k]

    def answer_query(self, query: str) -> dict:
        matches = self.retrieve(query)
        if not matches:
            return {
                "answer": "I do not have verified documentation in NovaPay's knowledge base to answer this query accurately.",
                "sources": [],
                "grounded": False
            }

        answer = f"According to NovaPay verified records:\n{matches[0]['content']}"
        return {
            "answer": answer,
            "sources": [d["title"] for d in matches],
            "grounded": True
        }

if __name__ == "__main__":
    engine = RAGEngine()
    test_q = "What is the settlement schedule for merchants?"
    result = engine.answer_query(test_q)
    print(f"Grounded: {result['grounded']}")
    print(f"Response: {result['answer']}")
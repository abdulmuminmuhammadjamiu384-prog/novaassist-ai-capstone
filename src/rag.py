import json
import re
import math
import os
import difflib
from typing import List, Dict, Tuple, Optional

SYNONYM_MAP = {
    "wht": "what",
    "wat": "what",
    "key": "fee",
    "keys": "fee",
    "cost": "fee",
    "costs": "fee",
    "pricing": "fee",
    "rate": "fee",
    "rates": "fee",
    "charges": "fee",
    "charge": "fee",
    "price": "fee",
    "payout": "settlement",
    "payouts": "settlement",
    "transfers": "settlement",
    "paperwork": "kyc",
    "contesting": "dispute",
    "reversal": "refund",
    "reversals": "refund",
    "forex": "currency",
}

CONVERSATIONAL_MAP = {
    "not smart": "I am continuously updated with official NovaPay operational, settlement, and compliance guidelines to assist you with precision. How can I help resolve your inquiry?",
    "stupid": "I am continuously updated with official NovaPay operational, settlement, and compliance guidelines to assist you with precision. How can I help resolve your inquiry?",
    "dumb": "I am here to help with all NovaPay merchant operations, transaction rates, settlements, and compliance inquiries.",
}

BLOCKED_SUBSTRINGS = ["stock price", "titanium", "office address", "cryptocurrency", "crypto", "president"]

STOP_WORDS = {
    "what", "is", "the", "are", "how", "to", "for", "a", "an", "in", "of", "and",
    "can", "i", "do", "we", "our", "my", "on", "it", "at", "by", "from", "with",
    "tell", "me", "about", "please", "give", "much", "many", "when", "does",
    "novapay", "pay", "solutions", "ltd", "today", "who"
}

def stem(word: str) -> str:
    for suffix in ["ing", "tion", "ment", "ies", "es", "ed", "s"]:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            if suffix == "ies":
                return word[:-3] + "y"
            return word[:-len(suffix)]
    return word

def tokenize(text: str) -> List[str]:
    clean = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    raw_tokens = clean.split()
    processed = []
    for token in raw_tokens:
        if token in STOP_WORDS:
            continue
        mapped = SYNONYM_MAP.get(token, token)
        processed.append(stem(mapped))
    return processed

DEFAULT_KB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "novapay_kb.json")

class RAGEngine:
    def __init__(self, kb_path: str = DEFAULT_KB_PATH):
        target_path = kb_path if os.path.isabs(kb_path) else os.path.abspath(kb_path)
        with open(target_path, "r", encoding="utf-8-sig") as f:
            self.documents: List[Dict] = json.load(f)
        
        self.doc_tokens: List[List[str]] = []
        self.vocabulary = set()
        
        for doc in self.documents:
            weighted_text = f"{doc.get('topic', '')} {doc.get('topic', '')} {' '.join(doc.get('keywords', []))} {' '.join(doc.get('keywords', []))} {doc.get('content', '')}"
            tokens = tokenize(weighted_text)
            self.doc_tokens.append(tokens)
            self.vocabulary.update(tokens)
        
        self.N = len(self.documents)
        self.idf = self._compute_idf()
        self.doc_vectors = [self._vectorize(tokens) for tokens in self.doc_tokens]

    def _compute_idf(self) -> Dict[str, float]:
        df = {}
        for token in self.vocabulary:
            count = sum(1 for tokens in self.doc_tokens if token in tokens)
            df[token] = count
        return {token: math.log((self.N + 1) / (df[token] + 1)) + 1.0 for token in self.vocabulary}

    def _vectorize(self, tokens: List[str]) -> Dict[str, float]:
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        
        vec = {}
        norm = 0.0
        for t, count in tf.items():
            if t in self.idf:
                weight = (1 + math.log(count)) * self.idf[t]
                vec[t] = weight
                norm += weight ** 2
        
        norm = math.sqrt(norm)
        if norm > 0:
            for t in vec:
                vec[t] /= norm
        return vec

    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        return sum(vec1.get(t, 0.0) * val for t, val in vec2.items())

    def retrieve(self, query: str, top_k: int = 1, threshold: float = 0.14) -> Optional[Tuple[Dict, float]]:
        q_lower = query.lower()
        for phrase in BLOCKED_SUBSTRINGS:
            if phrase in q_lower:
                return None

        q_tokens = tokenize(query)
        if not q_tokens:
            return None
        
        fuzzy_tokens = []
        for t in q_tokens:
            if t in self.vocabulary:
                fuzzy_tokens.append(t)
            else:
                matches = difflib.get_close_matches(t, list(self.vocabulary), n=1, cutoff=0.75)
                fuzzy_tokens.append(matches[0] if matches else t)

        q_vec = self._vectorize(fuzzy_tokens)
        best_doc = None
        best_score = -1.0

        for idx, doc_vec in enumerate(self.doc_vectors):
            score = self._cosine_similarity(q_vec, doc_vec)
            token_overlap = set(fuzzy_tokens).intersection(set(self.doc_tokens[idx]))
            
            if score > best_score and len(token_overlap) >= 1:
                best_score = score
                best_doc = self.documents[idx]

        if best_score >= threshold and best_doc is not None:
            return best_doc, best_score
        
        return None

    def answer_query(self, user_query: str) -> Dict:
        q_clean = user_query.lower().strip()
        for trig, resp in CONVERSATIONAL_MAP.items():
            if trig in q_clean:
                return {
                    "grounded": True,
                    "confidence": 1.0,
                    "topic": "NovaAssist Agent Help",
                    "answer": resp,
                    "content": resp,
                    "sources": ["NovaPay Assistant Support"],
                    "source": "NovaPay Assistant Support"
                }

        match = self.retrieve(user_query)
        if match:
            doc, score = match
            return {
                "grounded": True,
                "confidence": round(score, 3),
                "topic": doc["topic"],
                "answer": f"{doc['content']}",
                "content": f"{doc['content']}",
                "sources": [doc["topic"]],
                "source": doc["topic"]
            }
        
        return {
            "grounded": False,
            "confidence": 0.0,
            "topic": None,
            "answer": "I do not have verified documentation regarding this inquiry in NovaPay platform records. Please contact support@novapay.io or consult our documentation.",
            "content": "I do not have verified documentation regarding this inquiry in NovaPay platform records. Please contact support@novapay.io or consult our documentation.",
            "sources": [],
            "source": "Fallback Guardrail"
        }

    def query(self, user_query: str) -> Dict:
        return self.answer_query(user_query)

EnhancedRAGRetriever = RAGEngine

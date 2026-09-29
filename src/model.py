import os
import math
import json
import re
from collections import Counter

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "nova_model.json")

# Baseline training data for NovaAssist query intent classification
TRAINING_DATA = [
    ("hello hi hey good morning greetings sup", "greeting"),
    ("help me what can you do commands instructions capabilities", "help"),
    ("what is the weather forecast temperature outside rain sunny", "weather"),
    ("system status check diagnostics report health uptime cpu memory", "status"),
    ("goodbye bye see you later exit quit close terminate", "farewell")
]

def tokenize(text: str):
    """Basic tokenizer removing punctuation and lowering case."""
    return re.findall(r"\b\w+\b", text.lower())

def compute_tfidf_vectors(documents):
    """Computes TF-IDF vectors for documents using pure Python."""
    tokenized_docs = [tokenize(doc) for doc in documents]
    num_docs = len(documents)
    
    # Document frequency
    df = Counter()
    for doc in tokenized_docs:
        df.update(set(doc))
        
    vocab = sorted(list(df.keys()))
    idf = {term: math.log((1 + num_docs) / (1 + df[term])) + 1 for term in vocab}
    
    vectors = []
    for doc in tokenized_docs:
        tf = Counter(doc)
        total_words = len(doc) or 1
        vec = {term: (tf[term] / total_words) * idf[term] for term in doc}
        vectors.append(vec)
        
    return vectors, idf, vocab

def cosine_similarity(vec1, vec2):
    """Calculates cosine similarity between two sparse vector dicts."""
    common_terms = set(vec1.keys()) & set(vec2.keys())
    if not common_terms:
        return 0.0
    dot_product = sum(vec1[term] * vec2[term] for term in common_terms)
    norm1 = math.sqrt(sum(val ** 2 for val in vec1.values()))
    norm2 = math.sqrt(sum(val ** 2 for val in vec2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def train_and_save_model(model_path: str = MODEL_PATH):
    """Builds and serializes the pure-Python intent model."""
    texts = [item[0] for item in TRAINING_DATA]
    labels = [item[1] for item in TRAINING_DATA]
    
    vectors, idf, vocab = compute_tfidf_vectors(texts)
    
    model_data = {
        "labels": labels,
        "training_vectors": vectors,
        "idf": idf,
        "vocab": vocab
    }
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(model_data, f, indent=2)
    return model_data

def load_or_train_model(model_path: str = MODEL_PATH):
    """Loads model from disk or generates one if missing."""
    if os.path.exists(model_path):
        with open(model_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return train_and_save_model(model_path)

def predict_intent(text: str, model_path: str = MODEL_PATH) -> str:
    """Predicts the highest-confidence intent for a given query."""
    if not text or not text.strip():
        return "unknown"
        
    model = load_or_train_model(model_path)
    tokens = tokenize(text)
    tf = Counter(tokens)
    total_words = len(tokens) or 1
    
    query_vec = {
        term: (tf[term] / total_words) * model["idf"].get(term, 0)
        for term in tokens if term in model["idf"]
    }
    
    best_intent = "unknown"
    highest_score = 0.0
    
    for idx, train_vec in enumerate(model["training_vectors"]):
        score = cosine_similarity(query_vec, train_vec)
        if score > highest_score:
            highest_score = score
            best_intent = model["labels"][idx]
            
    return best_intent if highest_score > 0.1 else "unknown"

if __name__ == "__main__":
    print("Training baseline NovaAssist model...")
    train_and_save_model()
    print(f"Model saved successfully to: {MODEL_PATH}")
    
    sample_query = "Hey there, good morning!"
    prediction = predict_intent(sample_query)
    print(f"Test query: '{sample_query}' -> Predicted Intent: '{prediction}'")
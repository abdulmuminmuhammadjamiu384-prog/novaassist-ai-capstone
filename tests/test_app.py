import pytest
import os
import sys

# Ensure src/ is discoverable by test runner
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from model import predict_intent, load_or_train_model
from rag import RAGEngine
from app import generate_response

def test_model_loads_properly():
    """Verify that the intent classification model loads with expected keys."""
    model = load_or_train_model()
    assert "labels" in model
    assert "training_vectors" in model
    assert len(model["labels"]) > 0

def test_intent_classification():
    """Verify conversational intent detection."""
    assert predict_intent("hello there") == "greeting"
    assert predict_intent("help me please") == "help"

def test_rag_knowledge_retrieval():
    """Test grounded retrieval for known NovaPay documentation."""
    rag = RAGEngine()
    result = rag.answer_query("What is the domestic transaction fee?")
    assert result["grounded"] is True
    assert "1.4%" in result["answer"]
    assert "NovaPay Core Platform & Fees" in result["sources"]

def test_rag_hallucination_prevention():
    """Test out-of-domain queries return ungrounded fallback."""
    rag = RAGEngine()
    result = rag.answer_query("Can I buy cryptocurrency on NovaPay?")
    assert result["grounded"] is False
    assert "do not have verified documentation" in result["answer"]

def test_app_integrated_response():
    """Test full conversational routing and grounding pipeline."""
    # Intent route
    greet_res = generate_response("hi")
    assert "NovaAssist" in greet_res

    # RAG knowledge route
    doc_res = generate_response("What is the turnaround time for Tier 1 verification?")
    assert "24 to 48 business hours" in doc_res
    assert "Sources:" in doc_res
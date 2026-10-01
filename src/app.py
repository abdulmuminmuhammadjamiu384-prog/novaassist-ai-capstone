

import sys
import os
from dotenv import load_dotenv

sys.path.append(os.path.dirname(__file__))
from model import predict_intent
from rag import RAGEngine

load_dotenv()
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "NovaAssist")

engine = RAGEngine()

INTENT_RESPONSES = {
    "greeting": f"Hello! I am {ASSISTANT_NAME}, your NovaPay business copilot. How can I help you today?",
    "help": "You can ask about NovaPay fees, merchant onboarding, KYC, disputes, webhooks, or system compliance. Type 'exit' to quit.",
    "status": "System Status: NovaPay RAG Engine is online, indexed, and operational.",
    "farewell": f"Goodbye! Ending {ASSISTANT_NAME} session."
}

def generate_response(user_input: str) -> str:
    """Combines intent routing with RAG knowledge base retrieval."""
    intent = predict_intent(user_input)
    
    if intent in ["greeting", "help", "status", "farewell"]:
        return INTENT_RESPONSES[intent]
    
    # Query the RAG engine for knowledge-grounded answers
    rag_result = engine.answer_query(user_input)
    if rag_result["grounded"]:
        sources_str = ", ".join(rag_result["sources"])
        return f"{rag_result['answer']}\n\n[Sources: {sources_str}]"
    
    return "I am sorry, but I do not have verified documentation in the NovaPay knowledge base to answer this question accurately."

def run_assistant():
    print(f"\n==========================================")
    print(f"      {ASSISTANT_NAME} - NovaPay Copilot CLI     ")
    print(f"==========================================")
    print("Ask any question regarding NovaPay services (type 'exit' to quit).\n")

    while True:
        try:
            user_input = input("You > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                print(f"{ASSISTANT_NAME} > {INTENT_RESPONSES['farewell']}")
                break

            response = generate_response(user_input)
            print(f"\n{ASSISTANT_NAME} > {response}\n")

        except (KeyboardInterrupt, EOFError):
            print(f"\n{ASSISTANT_NAME} > Session terminated.")
            break

if __name__ == "__main__":
    run_assistant()

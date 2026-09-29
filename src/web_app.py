import os
import sys
import streamlit as st

sys.path.append(os.path.dirname(__file__))

from model import predict_intent
from rag import RAGEngine

st.set_page_config(
    page_title="NovaAssist | NovaPay Business Copilot",
    page_icon="💳",
    layout="wide"
)

@st.cache_resource
def get_rag_engine():
    return RAGEngine()

rag_engine = get_rag_engine()

with st.sidebar:
    st.title("💳 NovaAssist Copilot")
    st.caption("AI-Powered Merchant Operations Assistant")
    st.markdown("---")
    st.markdown("**Platform Status:** Operational 🟢")
    st.markdown("**Compliance:** PCI-DSS Level 1 | ISO 27001")
    st.markdown("**Retrieval Grounding:** Active (Anti-Hallucination)")
    st.markdown("---")
    st.markdown("### Quick Inquiries")
    st.info(
        "- What is the domestic transaction fee?\n"
        "- What are merchant KYC requirements?\n"
        "- How long do I have to contest a dispute?\n"
        "- What is the payout settlement schedule?"
    )
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

st.title("NovaAssist Business Operations Copilot")
st.write(
    "Ground-truth AI copilot for NovaPay Solutions Ltd. Provides verified guidance on fee structures, settlements, compliance, and integration."
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am NovaAssist, your verified business copilot for NovaPay Solutions. How can I assist your operations today?"
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about NovaPay platform operations..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    intent = predict_intent(prompt)

    if intent == "greeting":
        answer = "Hello! I am NovaAssist, your NovaPay business copilot. How can I assist you with your gateway or merchant operations today?"
    elif intent == "help":
        answer = "You can ask about transaction fees, KYC merchant onboarding, settlement timelines, dispute policies, or API security standards."
    elif intent == "farewell":
        answer = "Goodbye! Thank you for using NovaAssist."
    else:
        result = rag_engine.answer_query(prompt)
        if result["grounded"]:
            sources = ", ".join(result["sources"])
            answer = f"{result['answer']}\n\n**Verified Sources:** {sources}"
        else:
            answer = "⚠️️ **Verification Guard:** I do not have verified documentation in the NovaPay knowledge base to answer this question accurately."

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

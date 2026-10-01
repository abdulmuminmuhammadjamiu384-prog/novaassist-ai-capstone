import os
import sys

# Optional import for headless environments / pytest
try:
    import streamlit as st
except ImportError:
    st = None

# Ensure current module directory is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag import RAGEngine

# Engine instance
engine = RAGEngine()

# Comprehensive phrase & intent mapping
CONVERSATIONAL_INTENTS = [
    (["how are you", "how r u", "how do you do"],
     "I'm operating at peak performance and ready to assist you! What can I check for you regarding NovaPay transaction fees, settlements, or onboarding policies?"),

    (["who are you", "what is your name", "what are you"],
     "I am NovaAssist, an intelligent business operations copilot engineered specifically for NovaPay merchant solutions, settlements, disputes, and compliance."),

    (["what can you do", "what can you for me", "what do you do", "help me", "what can i ask", "services you offer", "what are your services", "services do you offer"],
     "I provide verified operational answers for NovaPay merchants. You can ask about transaction fees, settlement payout cycles (T+1/T+3), Tier 1 & Tier 2 KYC verification, dispute chargeback deadlines, and webhook API integration."),

    (["only about that", "is it only", "can i only ask"],
     "Yes, I specialize strictly in NovaPay merchant and fintech operations (pricing, payouts, onboarding, disputes, and compliance) to ensure complete accuracy without hallucination."),

    (["not smart", "stupid", "dumb", "useless"],
     "I am specialized strictly in official NovaPay documentation. If you share your specific operational or transaction question, I will fetch the exact policy for you."),

    (["thank you", "thanks", "thx"],
     "You are very welcome! Let me know if you need anything else regarding your NovaPay merchant account.")
]

def generate_response(user_input: str) -> str:
    cleaned = user_input.lower().strip()

    # 1. Check conversational intents first
    for triggers, response_text in CONVERSATIONAL_INTENTS:
        if any(t in cleaned for t in triggers):
            return response_text

    # 2. Short greetings
    if any(g == cleaned or cleaned.startswith(g + " ") for g in ["hi", "hello", "hey", "good morning", "good afternoon"]):
        return "Hello! I am NovaAssist, how can I assist your NovaPay merchant operations today?"

    # 3. Fall back to RAG knowledge base search
    rag_result = engine.answer_query(user_input)
    if rag_result["grounded"]:
        sources = rag_result.get("sources", [rag_result.get("source", "NovaPay Platform Records")])
        return f"{rag_result['answer']}\n\nVerified Sources: {', '.join(sources)}"
    else:
        return f"⚠️ Verification Guard: {rag_result['answer']}"

if st is not None:
    st.set_page_config(
        page_title="NovaAssist | NovaPay Business Operations",
        page_icon="💳",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # CSS to hide the Streamlit header, GitHub links, Fork button, and bottom Cloud badges
    st.markdown("""
    <style>
    /* Hide top header bar completely */
    header[data-testid="stHeader"],
    header,
    [data-testid="stToolbar"],
    [data-testid="stDecoration"] {
        display: none !important;
        visibility: hidden !important;
        height: 0px !important;
    }

    /* Hide MainMenu and Footer */
    #MainMenu, footer {
        display: none !important;
        visibility: hidden !important;
    }

    /* Target GitHub links and fork buttons specifically */
    a[href*="github.com"],
    a[href*="github"],
    button[title*="Fork"],
    div[title*="Fork"],
    button[title*="GitHub"] {
        display: none !important;
        visibility: hidden !important;
    }

    /* Hide Streamlit Community Cloud bottom viewer badges */
    div[class*="viewerBadge"],
    div[class*="manage-app"],
    [data-testid="manage-app-button"],
    .stDeployButton {
        display: none !important;
        visibility: hidden !important;
    }

    .block-container {
        padding-top: 1.5rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am NovaAssist, your verified business copilot for NovaPay Solutions. How can I assist your operations today?"}
        ]

    with st.sidebar:
        st.markdown("### 💳 NovaAssist Copilot")
        st.caption("AI-Powered Merchant Operations Assistant")
        st.markdown("---")
        st.markdown("**Platform Status:** Operational 🟢")
        st.markdown("**Compliance:** PCI-DSS Level 1 | ISO 27001")
        st.markdown("**Retrieval Grounding:** Active (Anti-Hallucination)")
        st.markdown("---")
        st.markdown("#### Quick Inquiries")
        sample_queries = [
            "What is the domestic transaction fee?",
            "What are merchant KYC requirements?",
            "How long do I have to contest a dispute?",
            "What is the payout settlement schedule?"
        ]
        for q in sample_queries:
            if st.button(q, key=f"btn_{q}"):
                st.session_state.messages.append({"role": "user", "content": q})
                res = generate_response(q)
                st.session_state.messages.append({"role": "assistant", "content": res})
                st.rerun()

    st.title("NovaAssist Business Operations Copilot")
    st.markdown("Ground-truth AI copilot for NovaPay Solutions Ltd. Provides verified guidance on fee structures, settlements, compliance, and integration.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="💳" if msg["role"] == "assistant" else "👤"):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask a question about NovaPay platform operations..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.write(prompt)
        
        reply = generate_response(prompt)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant", avatar="💳"):
            st.write(reply)

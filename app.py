import os
import sys

# Optional import for local pytest environments
try:
    import streamlit as st
except ImportError:
    st = None

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag import RAGEngine

# Global Engine Instance
engine = RAGEngine()

# Smart Conversational Knowledge & Politeness Mapping
SMART_CONVERSATIONAL = {
    "how are you": "I'm operating smoothly and ready to assist you! What can I check for you regarding NovaPay transaction fees, payouts, or onboarding?",
    "who are you": "I am NovaAssist, an intelligent operations copilot built specifically for NovaPay merchant solutions, settlement workflows, disputes, and compliance.",
    "what can you do": "I can provide exact data on NovaPay's transaction fee caps, payout settlement timelines (T+1 / T+3), KYC verification steps, dispute deadlines, and webhook API integration.",
    "only about that": "My core specialization is NovaPay financial operations and merchant policies—so I focus on fees, payouts, verification, chargebacks, and API integrations to give you verified accuracy without guessing.",
    "is it only": "I specialize in NovaPay merchant and fintech operations. If you need details on fees, chargeback timelines, verification requirements, or developer endpoints, I have direct answers.",
    "thank you": "You're very welcome! Let me know if you need help with anything else regarding your NovaPay merchant account.",
    "thanks": "Glad to help! Feel free to ask any other questions about payouts, rates, or account setup."
}

def predict_intent(user_input: str) -> str:
    cleaned = user_input.lower().strip()
    if any(k in cleaned for k in ["hello", "hi", "hey", "good morning", "good afternoon"]):
        return "greeting"
    return "query"

def generate_response(user_input: str) -> str:
    cleaned = user_input.lower().strip()
    
    # 1. Smart Conversational Check
    for phrase, answer in SMART_CONVERSATIONAL.items():
        if phrase in cleaned:
            return answer
            
    # 2. Greeting Check
    if predict_intent(user_input) == "greeting" and len(cleaned.split()) <= 2:
        return "Hello! I am NovaAssist, how can I assist your NovaPay merchant operations today?"
        
    # 3. Knowledge Base RAG Retrieval
    rag_result = engine.answer_query(user_input)
    if rag_result["grounded"]:
        sources = rag_result.get("sources", [rag_result.get("source", "NovaPay Platform Records")])
        sources_str = ", ".join(sources)
        return f"{rag_result['answer']}\n\nSources: {sources_str}"
    else:
        return f"⚠️️ Verification Guard: {rag_result['answer']}"

# Streamlit Interface Rendering
if st is not None:
    st.set_page_config(
        page_title="NovaAssist | NovaPay Business Operations",
        page_icon="💳",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Aggressive CSS to hide: Top Header, Fork, GitHub link, 3 dots, and Streamlit Community Cloud Host badges
    st.markdown("""
    <style>
    /* 1. Hide entire top Streamlit header bar & GitHub icon */
    header[data-testid="stHeader"], header, [data-testid="stToolbar"], [data-testid="stDecoration"] {
        display: none !important;
        visibility: hidden !important;
        height: 0px !important;
    }

    /* 2. Hide MainMenu and Footer */
    #MainMenu, footer {
        display: none !important;
        visibility: hidden !important;
    }

    /* 3. Hide bottom-right Community Cloud Host / Profile / Manage app badges */
    div[class*="viewerBadge"],
    div[class*="manage-app"],
    [data-testid="manage-app-button"],
    div[class*="stAppDeployButton"],
    .stDeployButton,
    iframe[title*="manage"],
    div[style*="z-index: 1000001"],
    div[style*="z-index: 999999"],
    div[style*="position: fixed"][style*="bottom"] {
        display: none !important;
        visibility: hidden !important;
    }

    /* 4. Hide any stray GitHub / Fork links */
    a[href*="github.com"], button[title*="Fork"], button[title*="GitHub"] {
        display: none !important;
        visibility: hidden !important;
    }

    /* 5. Adjust layout padding */
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

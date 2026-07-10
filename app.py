"""
Streamlit UI for Agentic Research Memory System
================================================
Run with:
    streamlit run app.py
"""

import asyncio
import streamlit as st
from datetime import datetime
from pathlib import Path
from typing import Optional
import json

from src.agents.session_manager import session_manager
from src.agents.query_agent import query_agent
from src.agents.monitor_agent import monitor_agent
from src.tools.research_index import research_index
from src.models.session import ResearchSession


# =========================================================
# Page Config
# =========================================================
st.set_page_config(
    page_title="Research Memory",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a cleaner look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .paper-card {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        background: #f8fafc;
    }
    .source-chip {
        display: inline-block;
        background: #e0e7ff;
        color: #3730a3;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        margin-right: 0.4rem;
        margin-bottom: 0.3rem;
    }
    .stChatMessage {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# Helpers
# =========================================================
def run_async(coro):
    """Helper to run async code inside Streamlit"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def get_or_create_session(topic: str) -> ResearchSession:
    return session_manager.get_or_create_session(topic)


# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.markdown("### 🧠 Research Memory")
    st.caption("Agentic Research Assistant")

    st.divider()

    # ---- Session Management ----
    st.subheader("Sessions")

    sessions = session_manager.list_sessions()
    session_options = {f"{s.topic[:40]} ({s.session_id})": s for s in sessions}

    if "current_session" not in st.session_state:
        st.session_state.current_session = None

    # Select existing session
    if session_options:
        selected_label = st.selectbox(
            "Open existing session",
            options=["— Select —"] + list(session_options.keys()),
            key="session_selector"
        )
        if selected_label != "— Select —":
            st.session_state.current_session = session_options[selected_label]

    # Create new session
    with st.expander("➕ New Research Session", expanded=not sessions):
        new_topic = st.text_input("Research Topic", placeholder="e.g. agentic RAG memory systems")
        col1, col2 = st.columns(2)
        with col1:
            ingest_now = st.checkbox("Ingest papers now", value=True)
        with col2:
            create_btn = st.button("Create", type="primary", use_container_width=True)

        if create_btn and new_topic.strip():
            with st.spinner("Creating session..."):
                session = session_manager.create_session(new_topic.strip())
                if ingest_now:
                    with st.spinner("Ingesting papers from arXiv (this can take a few minutes)..."):
                        session = run_async(session_manager.ensure_papers_ingested(session))
                st.session_state.current_session = session
                st.success(f"Session created with {len(session.papers_ingested)} papers")
                st.rerun()

    st.divider()

    # ---- Current Session Info ----
    if st.session_state.current_session:
        s = st.session_state.current_session
        st.markdown("**Active Session**")
        st.markdown(f"**Topic:** {s.topic}")
        st.markdown(f"**Papers:** {len(s.papers_ingested)}")
        st.markdown(f"**Messages:** {len(s.conversation)}")
        st.caption(f"ID: `{s.session_id}`")

        if st.button("🔄 Re-ingest papers", use_container_width=True):
            with st.spinner("Re-running ingestion..."):
                s = run_async(session_manager.ensure_papers_ingested(s, force=True))
                st.session_state.current_session = s
                st.success(f"Now have {len(s.papers_ingested)} papers")
                st.rerun()

        if st.button("📡 Run Continuous Monitor", use_container_width=True):
            with st.spinner("Checking arXiv for new papers..."):
                results = run_async(monitor_agent.run_once(topics=[s.topic]))
                if results and results[0]["successfully_ingested"] > 0:
                    st.success(f"Ingested {results[0]['successfully_ingested']} new papers!")
                else:
                    st.info("No new papers found.")
                st.rerun()

    st.divider()

    # ---- Global Stats ----
    stats = research_index.stats()
    st.markdown("**Knowledge Base**")
    st.metric("Total Papers", stats["total_papers"])
    st.metric("Tracked Topics", stats["total_topics"])


# =========================================================
# Main Area
# =========================================================
st.markdown('<div class="main-header">Research Memory</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Talk to your personal research knowledge base</div>', unsafe_allow_html=True)


if st.session_state.current_session is None:
    st.info("👈 Create a new research session or select an existing one from the sidebar to start chatting.")
    st.stop()


session: ResearchSession = st.session_state.current_session


# ---- Tabs ----
tab_chat, tab_papers, tab_history = st.tabs(["💬 Chat", "📄 Papers", "📜 History"])


# =========================================================
# TAB 1: Chat
# =========================================================
with tab_chat:
    # Display previous conversation
    for msg in session.conversation:
        with st.chat_message(msg.role):
            st.markdown(msg.content)
            if msg.role == "assistant" and msg.sources:
                with st.expander("Sources used"):
                    for src in msg.sources:
                        st.markdown(
                            f"- **[{src.get('paper_id')}]({src.get('arxiv_url')})** — {src.get('title', '')[:80]}"
                        )

    # Chat input
    if prompt := st.chat_input("Ask anything about this research topic..."):
        # Show user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Save user message
        session_manager.add_message("user", prompt)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking over your research notes..."):
                result = run_async(query_agent.answer(prompt, topic=session.topic))

            st.markdown(result["answer"])

            if result["sources"]:
                st.markdown("**Sources:**")
                for src in result["sources"]:
                    score = f" (score: {src['score']:.2f})" if src.get("score") else ""
                    st.markdown(
                        f'<span class="source-chip">[{src["paper_id"]}]({src["arxiv_url"]}){score}</span> {src["title"][:70]}',
                        unsafe_allow_html=True
                    )

        # Save assistant message
        session_manager.add_message("assistant", result["answer"], sources=result["sources"])

        # Refresh session object
        st.session_state.current_session = session_manager.load_session(session.session_id)
        st.rerun()


# =========================================================
# TAB 2: Papers
# =========================================================
with tab_papers:
    st.subheader(f"Papers in this session ({len(session.papers_ingested)})")

    if not session.papers_ingested:
        st.warning("No papers ingested yet. Use the sidebar to ingest papers.")
    else:
        for arxiv_id in session.papers_ingested:
            # Try to load metadata from artifact store
            meta_path = Path("papers") / arxiv_id / "metadata.json"
            note_path = Path("papers") / arxiv_id / "knowledge_note.json"

            title = arxiv_id
            abstract_snippet = ""
            one_sentence = ""

            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    title = meta.get("title", arxiv_id)
                except Exception:
                    pass

            if note_path.exists():
                try:
                    note = json.loads(note_path.read_text(encoding="utf-8"))
                    one_sentence = note.get("one_sentence_summary", "")
                except Exception:
                    pass

            with st.container():
                st.markdown(f"### [{arxiv_id}](https://arxiv.org/abs/{arxiv_id})")
                st.markdown(f"**{title}**")
                if one_sentence:
                    st.caption(one_sentence)
                st.markdown(f"[Open on arXiv](https://arxiv.org/abs/{arxiv_id})  •  [PDF](https://arxiv.org/pdf/{arxiv_id}.pdf)")
                st.divider()


# =========================================================
# TAB 3: History
# =========================================================
with tab_history:
    st.subheader("Conversation History")

    if not session.conversation:
        st.info("No messages yet. Start chatting!")
    else:
        for i, msg in enumerate(session.conversation):
            role_icon = "🧑" if msg.role == "user" else "🤖"
            with st.expander(f"{role_icon} {msg.role.capitalize()} — {msg.timestamp.strftime('%Y-%m-%d %H:%M')}"):
                st.markdown(msg.content)
                if msg.sources:
                    st.markdown("**Sources:**")
                    for src in msg.sources:
                        st.markdown(f"- [{src.get('paper_id')}]({src.get('arxiv_url')}) {src.get('title', '')[:60]}")


# =========================================================
# Footer
# =========================================================
st.divider()
st.caption(f"Session `{session.session_id}`  •  Topic: {session.topic}  •  Powered by LangGraph + Ollama")


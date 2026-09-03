import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from src.answer import answer_query

st.set_page_config(page_title="CBSE Grade 8 Science Tutor", page_icon="🔬")

st.title("🔬 CBSE Grade 8 Science Tutor")
st.caption("Ask anything from your NCERT Grade 7 & 8 Science books")

grade = st.selectbox("Your grade", [7, 8], index=1)

examples = [
    "What is a cell made of?",
    "How does yeast help make bread?",
    "What is photosynthesis?",
    "How do we see the phases of the moon?",
]

st.write("**Try an example:**")
cols = st.columns(2)
for i, ex in enumerate(examples):
    if cols[i % 2].button(ex, use_container_width=True):
        st.session_state["question"] = ex

question = st.text_input("Or ask your own question", key="question")

if question:
    with st.spinner("Thinking..."):
        result = answer_query(question, user_grade=grade)

    st.divider()
    if result["status"] == "answered":
        st.markdown(result["answer"])
        st.caption("📖 Sources: " + ", ".join(result["sources"]))
    else:
        st.info(result["message"])
import streamlit as st
from dotenv import load_dotenv

from document_reader import read_pdf
from rag_pipeline import create_rag

from test_results_analyzer import (
    analyze_csv
)

from requirements_extractor import (
    extract_requirements,
    save_excel
)

load_dotenv()

st.set_page_config(
    page_title="Interior Development AI Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Interior Development AI Assistant")

st.markdown(
    """
    AI-powered assistant for engineering and automotive documents.
    """
)

tab1, tab2, tab3 = st.tabs(
    [
        "RAG Assistant",
        "Requirements Extraction",
        "Test Results Analysis"
    ]
)

# ==================================================
# TAB 1 : RAG ASSISTANT
# ==================================================

with tab1:

    pdf = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        key="rag_pdf"
    )

    if pdf:

        with st.spinner("Processing document..."):

            docs = read_pdf(pdf)

            st.write(
                "Pages extracted:",
                len(docs)
            )

            retriever, llm = create_rag(docs)

        st.success(
            "Document processed successfully!"
        )

        question = st.text_input(
            "Ask a question"
        )

        if question:

            retrieved_docs = retriever.invoke(
                question
            )

            if len(retrieved_docs) == 0:

                st.error(
                    "No relevant content found."
                )

                st.stop()

            context = "\n".join(
                [
                    doc.page_content
                    for doc in retrieved_docs
                ]
            )

            prompt = f"""
You are a helpful engineering assistant.

Answer ONLY using the provided context.

Context:
{context}

Question:
{question}
"""

            try:

                response = llm.invoke(
                    prompt
                )

                st.subheader(
                    "Answer"
                )

                answer = ""

                try:

                    if isinstance(
                        response.content,
                        list
                    ):

                        for item in response.content:

                            if isinstance(
                                item,
                                dict
                            ):

                                answer += item.get(
                                    "text",
                                    ""
                                )

                    else:

                        answer = str(
                            response.content
                        )

                except Exception:

                    answer = str(
                        response
                    )

                st.markdown(answer)

            except Exception as e:

                st.error(
                    f"Gemini Error: {e}"
                )

            st.subheader(
                "📄 Sources"
            )

            seen = set()

            for doc in retrieved_docs:

                source = doc.metadata.get(
                    "source",
                    "Unknown File"
                )

                page = doc.metadata.get(
                    "page",
                    "?"
                )

                key = (
                    f"{source}-{page}"
                )

                if key not in seen:

                    seen.add(key)

                    st.write(
                        f"📌 {source} | Page {page}"
                    )

# ==================================================
# TAB 2 : REQUIREMENTS EXTRACTION
# ==================================================

with tab2:

    st.subheader(
        "Extract Requirements from Specification Documents"
    )

    req_pdf = st.file_uploader(
        "Upload Specification PDF",
        type=["pdf"],
        key="req_pdf"
    )

    if req_pdf:

        try:

            docs = read_pdf(
                req_pdf
            )

            text = "\n".join(
                [
                    doc.page_content
                    for doc in docs
                ]
            )

            st.info(
                "Extracting requirements..."
            )

            requirements = (
                extract_requirements(
                    text
                )
            )

            st.subheader(
                "Extracted Requirements"
            )

            st.text_area(
    "Requirements",
    str(requirements),
    height=400
)

            excel_file = save_excel(
                requirements
            )

            st.success(
                f"Excel file created: {excel_file}"
            )

        except Exception as e:

            st.error(
                f"Requirements Extraction Error: {e}"
            )

with tab3:

    st.subheader(
        "Analyze Engineering Test Results"
    )

    csv_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        key="csv_upload"
    )

    if csv_file:

        result = analyze_csv(
            csv_file
        )

        st.subheader(
            "Analysis"
        )

        st.markdown(result)        
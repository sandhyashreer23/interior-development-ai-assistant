from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


def create_rag(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)
    chunks = [
       chunk
    for chunk in chunks
    if chunk.page_content.strip()
   ]

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = FAISS.from_documents(
        chunks,
        embeddings
    )
    print("Total chunks:", len(chunks))
    if len(chunks) == 0:
      raise Exception(
        "No text chunks generated."
    )


    retriever = vectordb.as_retriever()

    llm = ChatGoogleGenerativeAI(
         model="gemini-3.8-flash",
        temperature=0
    )

    return retriever, llm
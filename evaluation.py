import json
import time
from dotenv import load_dotenv

load_dotenv()

from document_reader import read_pdf
from rag_pipeline import create_rag


import glob

pdf_files = glob.glob("*.pdf")

print("PDFs found:", pdf_files)

PDF_FILE = pdf_files[0]


# Load PDF
with open(PDF_FILE, "rb") as f:

    docs = read_pdf(f)

retriever, llm = create_rag(docs)


# Load questions
with open(
    "evaluation_questions.json",
    "r",
    encoding="utf-8"
) as f:

    questions = json.load(f)


correct = 0
total_time = 0


for item in questions:

    question = item["question"]

    expected = item["expected_answer"]

    start = time.time()

    retrieved_docs = retriever.invoke(question)

    context = "\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    prompt = f"""
Answer only from the context.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    elapsed = time.time() - start

    total_time += elapsed

    try:

        if isinstance(
            response.content,
            list
        ):

            answer = ""

            for block in response.content:

                if isinstance(block, dict):

                    answer += block.get(
                        "text",
                        ""
                    )

        else:

            answer = str(response.content)

    except:

        answer = str(response)

    print("\n---------------------")
    print("Question:", question)
    print("Expected:", expected)
    print("Actual:", answer)

    if expected.lower() in answer.lower():

        correct += 1

        print("✓ Correct")

    else:

        print("✗ Incorrect")


accuracy = (
    correct /
    len(questions)
) * 100

avg_time = (
    total_time /
    len(questions)
)

print("\n====================")
print("Total Questions:", len(questions))
print("Correct:", correct)
print("Accuracy:", round(accuracy, 2), "%")
print(
    "Average Latency:",
    round(avg_time, 2),
    "seconds"
)
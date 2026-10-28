import os
import pandas as pd

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def extract_requirements(text):

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0
    )

    prompt = f"""
You are an engineering requirements analyst.

Extract all requirements.

Return ONLY a markdown table:

| ID | Requirement | Category |
|----|-------------|----------|

Categories:
- Safety
- Comfort
- Ergonomics
- Material
- Regulatory

Document Text:

{text[:15000]}
"""

    response = llm.invoke(prompt)

    try:

        content = response.content

        # force everything into a string
        if isinstance(content, list):

            output = ""

            for item in content:

                if isinstance(item, dict):

                    output += item.get(
                        "text",
                        ""
                    )

                else:

                    output += str(item)

            return str(output)

        return str(content)

    except Exception:

        return str(response)

def save_excel(data):

    import pandas as pd
    import os

    # Convert list to string
    if isinstance(data, list):

        text = ""

        for item in data:

            if isinstance(item, dict):

                text += item.get(
                    "text",
                    ""
                )

    else:

        text = str(data)

    lines = text.split("\n")

    rows = []

    for line in lines:

        if "REQ-" in line:

            parts = [
                p.strip()
                for p in line.split("|")
                if p.strip()
            ]

            if len(parts) >= 3:

                rows.append(
                    [
                        parts[0],
                        parts[1],
                        parts[2]
                    ]
                )

    if not rows:

        rows.append(
            [
                "REQ-001",
                "No requirements parsed",
                "Regulatory"
            ]
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "ID",
            "Requirement",
            "Category"
        ]
    )

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    file_path = "outputs/requirements.xlsx"

    df.to_excel(
        file_path,
        index=False
    )

    return file_path
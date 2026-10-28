import pandas as pd

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def analyze_csv(csv_file):

    df = pd.read_csv(csv_file)

    stats = df.describe(
        include="all"
    ).to_string()

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0
    )

    prompt = f"""
You are an engineering analyst.

Analyze these test results and provide:

## 1. Key Findings
## 2. Trends
## 3. Anomalies
## 4. Recommendations

Data Summary:

{stats}
"""

    response = llm.invoke(prompt)

    try:

        if isinstance(response.content, list):

            output = ""

            for item in response.content:

                if isinstance(item, dict):

                    output += item.get(
                        "text",
                        ""
                    )

                else:

                    output += str(item)

            return output

        return str(response.content)

    except Exception:

        return str(response)
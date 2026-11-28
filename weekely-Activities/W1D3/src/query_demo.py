
import pandas as pd
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate

DATA_PATH = "data/cleaned/states_cleaned.csv"
DB_PATH = "chroma_db"



def answer_numeric_question(question):
    """Answer numerical questions using the latest record per state."""

    df = pd.read_csv(DATA_PATH)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["State"] = df["State"].astype(str).str.strip()

    # Exclude national totals and unassigned records
    df = df[
        ~df["State"].str.lower().isin(["india", "state unassigned"])
    ]

    # Remove rows with invalid dates
    df = df.dropna(subset=["Date"])

    # Select the latest record for each state
    df = df.sort_values("Date")
    latest_records = df.groupby("State").tail(1).copy()

    latest_records["Confirmed"] = pd.to_numeric(
        latest_records["Confirmed"], errors="coerce"
    )
    latest_records = latest_records.dropna(subset=["Confirmed"])

    if latest_records.empty:
        return "No valid state records found."

    question = question.lower()

    if "highest" in question and "confirmed" in question:
        row = latest_records.loc[
            latest_records["Confirmed"].idxmax()
        ]
        return (
            f"{row['State']} has the highest confirmed cases: "
            f"{int(row['Confirmed'])}."
        )

    if "lowest" in question and "confirmed" in question:
        row = latest_records.loc[
            latest_records["Confirmed"].idxmin()
        ]
        return (
            f"{row['State']} has the lowest confirmed cases: "
            f"{int(row['Confirmed'])}."
        )

    if "average" in question and "confirmed" in question:
        average = latest_records["Confirmed"].mean()
        return (
            f"The average confirmed cases across the latest "
            f"state records is {average:.2f}."
        )

    return None



def query_data(question):
    """Use Pandas for numerical queries and RAG for general questions."""

    numeric_answer = answer_numeric_question(question)

    if numeric_answer:
        print("\nAnswer:")
        print(numeric_answer)
        return

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vector_store = Chroma(
        collection_name="covid_state_data",
        embedding_function=embeddings,
        persist_directory=DB_PATH
    )

    results = vector_store.similarity_search(question, k=3)

    context = "\n".join(
        document.page_content for document in results
    )

    prompt = ChatPromptTemplate.from_template(
        """
        Answer using only the context below.

        If the answer is not available, clearly say so.

        Context:
        {context}

        Question:
        {question}
        """
    )

    model = ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )

    chain = prompt | model

    response = chain.invoke({
        "context": context,
        "question": question
    })

    print("\nRetrieved records:")
    for document in results:
        print("-", document.page_content)

    print("\nAI Answer:")
    print(response.content)


if __name__ == "__main__":
    question = input("Ask a question about the dataset: ")
    query_data(question)

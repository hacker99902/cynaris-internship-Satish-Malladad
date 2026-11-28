import pandas as pd
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DATA_PATH = "data/cleaned/states_cleaned.csv"
DB_PATH = "chroma_db"


def create_documents(df):
    """Convert the latest valid record for each state into documents."""

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Remove national totals and unassigned records
    df["State"] = df["State"].str.strip()

    df = df[
        ~df["State"].str.lower().isin(["india", "state unassigned"])
    ]

    # Remove rows without a valid date
    df = df.dropna(subset=["Date"])

    # Keep the latest record for each state
    df = df.sort_values("Date")
    latest_records = df.groupby("State").tail(1)

    documents = []

    for _, row in latest_records.iterrows():
        content = (
            f"State: {row['State']}. "
            f"Date: {row['Date'].date()}. "
            f"Confirmed cases: {row['Confirmed']}. "
            f"Recovered cases: {row['Recovered']}. "
            f"Deceased: {row['Deceased']}. "
            f"Other cases: {row['Other']}. "
            f"Total tests: {row['Tested']}."
        )

        documents.append(
            Document(
                page_content=content,
                metadata={"state": str(row["State"])}
            )
        )

    return documents

def build_vector_store():
    """Generate embeddings and store documents in ChromaDB."""

    df = pd.read_csv(DATA_PATH)

    documents = create_documents(df)

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    vector_store = Chroma(
        collection_name="covid_state_data",
        embedding_function=embeddings,
        persist_directory=DB_PATH
    )

    # Clear old records before rebuilding the index
    existing_ids = vector_store.get()["ids"]

    if existing_ids:
        vector_store.delete(ids=existing_ids)

    vector_store.add_documents(documents)

    print(f"Created {len(documents)} documents.")
    print("Embeddings generated and stored in ChromaDB.")

    return vector_store

if __name__ == "__main__":
    build_vector_store()

#Importing libraries
import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()

FAISS_PATH = "faiss_food_index"

# --- CSV Loader ---
def load_csv(path):
    df = pd.read_csv(path, encoding="utf-8-sig")

    if len(df.columns) == 1:
        df = pd.read_csv(path, sep="\t", encoding="utf-8-sig")

    df.columns = df.columns.str.strip().str.lower()
    print(f"{path} -> COLUMNS:", df.columns.tolist())

    return df

# --- Convert CSV → Documents ---
def load_food_docs():
    allergy_df = load_csv("Data/Allergy.csv")
    sub_df = load_csv("Data/Substitution.csv")

    docs = []

    for _, row in allergy_df.iterrows():
        docs.append(Document(
            page_content=f"ingredient: {row.get('ingredient', '')}\nallergy: {row.get('allergy', '')}",
            metadata={"type": "allergy"}
        ))

    for _, row in sub_df.iterrows():
        docs.append(Document(
            page_content=(
                f"ingredient: {row.get('ingredient', '')}\n"
                f"substitution: {row.get('substitution', '')}\n"
                f"notes: {row.get('notes', '')}"
            ),
            metadata={"type": "substitution"}
        ))

    return docs

# --- Build + Save FAISS ---
def build_vectorstore():
    docs = load_food_docs()

    # Light chunking (important if notes get long)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    split_docs = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    vectorstore.save_local(FAISS_PATH)
    print("FAISS index saved.")

    return vectorstore

# --- Load Existing Index ---
def load_vectorstore():
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

# --- Retriever ---
def get_retriever():
    if os.path.exists(FAISS_PATH):
        print("Loading existing FAISS index...")
        vectorstore = load_vectorstore()
    else:
        print("Building FAISS index...")
        vectorstore = build_vectorstore()

    return vectorstore.as_retriever(search_kwargs={"k": 3})
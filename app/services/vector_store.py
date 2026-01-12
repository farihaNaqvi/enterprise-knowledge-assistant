from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


def get_vector_store(chunks):
    # Using text-embedding-3-small is 5x cheaper than the older version
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # This creates a local database folder called 'chroma_db'
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vector_db
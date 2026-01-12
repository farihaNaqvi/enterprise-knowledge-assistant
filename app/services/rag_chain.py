from langchain_openai import ChatOpenAI
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


def create_rag_assistant(vector_db):
    # temperature=0 makes the AI factual and consistent
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    prompt = ChatPromptTemplate.from_template("""
    You are a professional corporate assistant. Answer the question 
    using ONLY the provided context. If the answer isn't there, 
    say "I don't have that information in my database."

    Context: {context}
    Question: {input}
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    return create_retrieval_chain(retriever, document_chain)
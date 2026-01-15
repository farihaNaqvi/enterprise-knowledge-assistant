import boto3
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


class IngestionService:
    def __init__(self):
        # We use boto3 here to show 'Enterprise' readiness (cloud storage)
        self.s3 = boto3.client('s3')
        # Recursive splitter is better than CharacterSplitter because it
        # tries to keep paragraphs and sentences together.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )

    async def process_pdf(self, file_path: str):
        # 1. Load the PDF from a path
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # 2. Split into chunks
        chunks = self.splitter.split_documents(docs)

        # 3. Add Metadata for Citations (Crucial for Enterprise apps)
        for chunk in chunks:
            chunk.metadata["source"] = file_path.split("/")[-1]

        return chunks

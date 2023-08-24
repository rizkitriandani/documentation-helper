import os
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings,GooglePalmEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment=os.environ.get("PINECONE_ENVIRONMENT_REGION"),
)


def ingest_docs() -> None:
    loader = ReadTheDocsLoader(path="langchain-docs/api.python.langchain.com/en/latest")
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents) }documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Splitted into {len(documents)} chunks")

    for doc in documents:
        old_path = doc.metadata["source"]
        new_url = old_path.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to insert {len(documents)} to Pinecone")
    # embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    embeddings = GooglePalmEmbeddings(google_api_key=os.environ.get("GOOGLE_PALM_API_KEY"))
    Pinecone.from_documents(
        documents, embeddings, index_name="langchain-doc-index"
    )
    print("****** Added to Pinecone vectorstore vectors")


if __name__ == "__main__":
    ingest_docs()


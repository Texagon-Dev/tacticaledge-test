# Uncomment the following line if you need to initialize FAISS with no AVX2 optimization
# os.environ['FAISS_NO_AVX2'] = '1'

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def embed_docs(pdf):

    loader = UnstructuredPDFLoader(pdf)
    data=loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=750,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
    )

    docs=text_splitter.split_documents(data)
    
    print(docs)
    
    db = FAISS.from_documents(docs, embeddings)
    return db

def save_db_locally(doc1 , doc2):
    db1 = embed_docs(doc1)
    db2 = embed_docs(doc2)

    db1.save_local("Embeddings/mobily")
    db2.save_local("Embeddings/operation_and_maintainance")

def get_db(db_name):
    FAISS.allow_dangerous_deserialization = True
    if db_name == "mobily":
        db = FAISS.load_local(r"D:\Texagon\Urgent Project\Embeddings\mobily", embeddings , allow_dangerous_deserialization=True)

        return db
    elif db_name == "operation":
        db = FAISS.load_local(r"D:\Texagon\Urgent Project\Embeddings\operation_and_maintainance", embeddings , allow_dangerous_deserialization=True)

        return db


def main():
    doc1 = "data/mobily.pdf"
    doc2 = "data/operation_and_maintainance.pdf"

    save_db_locally(doc1, doc2)

# print(get_db("mobily"))
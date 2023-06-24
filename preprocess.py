from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from chromadb.config import Settings
from langchain.document_loaders import PDFMinerLoader

from constant import (PERSIST_DIRECTORY,
                      CHROMA_SETTINGS)


def pdf_loader(pdf_file):

    loader = PDFMinerLoader("uploaded_files/" + pdf_file)
    return loader.load()
    

    # pdf_reader = PdfReader(pdf_file)
    # print(pdf_reader)

    # text = ""
    # for page in pdf_reader.pages:
    #     text += page.extract_text()

    

def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(text)
    return docs
    
def create_knowledge(chunks):
    embedding = HuggingFaceInstructEmbeddings(
            model_name = "hkunlp/instructor-large"

            #TODO: Decide on using device type
            , model_kwargs={"device": "cuda"}
        )

    db = Chroma.from_documents(chunks,
                               embedding,
                               persist_directory=PERSIST_DIRECTORY,
                               client_settings=CHROMA_SETTINGS,)
    
    db.persist()
    db = None
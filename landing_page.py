import streamlit as st
from preprocess import pdf_loader, split_text, create_knowledge

from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from constant import (PERSIST_DIRECTORY,
                      CHROMA_SETTINGS)

from model import load_model
from langchain.chains import RetrievalQA
import os

def main():

    with st.sidebar:

        st.title("Document ChatBot")
        document = st.file_uploader('Upload PDF files', type=['pdf'])
        print(type(document))
        

        if document is not None:
            st.info("Generating Knowledge Base.")
            with open(os.path.join("uploaded_files",document.name),"wb") as f: 
                f.write(document.getbuffer()) 
                st.success("Saved File")
            

    
    if document is not None:

        text = pdf_loader(document.name)
        text_chunks = split_text(text)
        create_knowledge(text_chunks)
        st.info("Done Generating Knowlegde Base")

        query = st.text_input("Ask questions here")

        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large", model_kwargs={"device": "cuda"})

        db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
            client_settings=CHROMA_SETTINGS,
        )

        retriever = db.as_retriever()

        model_id = "TheBloke/WizardLM-7B-uncensored-GPTQ"
        model_basename = "WizardLM-7B-uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors"
        llm = load_model(device_type="cuda", model_id=model_id, model_basename=model_basename)

        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

        res = qa(query)
        answer, docs = res["result"], res["source_documents"]

        st.text(answer)

if __name__ == '__main__':
    main()
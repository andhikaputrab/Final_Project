import streamlit as st
from haystack import Document
from src.utils.styling import load_css
from src.utils.config import config
from src.connection.db_connector import MongoDBConnector
from src.data.data_loader import WikipediaFetcher
from src.data.haystack_pipeline import HaystackPipeline

st.set_page_config(
    page_title=config.get('PAGE_TITLE_CHAT'),
    page_icon='🤖',
    layout=config.get('LAYOUT_CHAT')
)

load_css()

# Inisialisasi koneksi MongoDB dan pipeline
document_store = MongoDBConnector().document_store
haystack_pipeline = HaystackPipeline(document_store)

def main():
    st.title('Question Answering Application')
    
    # Input pertanyaan pengguna
    question = st.text_input("Enter your question:")
    
    if question:
        # Mendapatkan artikel dari Wikipedia
        wikipedia_search = WikipediaFetcher(language='id')
        wikipedia_documents = wikipedia_search.fetch_wikipedia_articles(question)
        
        document_objects = [Document(content=doc["content"], meta=doc["meta"]) for doc in wikipedia_documents]
        if not document_objects:
            raise ValueError("No documents to store.")
        
        # Menyimpan dokumen ke MongoDB
        pipeline_storing_mongodb = haystack_pipeline.document_store_pipeline()
        pipeline_storing_mongodb.run({"documents": document_objects})

        # Menjalankan pipeline untuk menghasilkan jawaban
        answer_pipeline = haystack_pipeline.answer_generator_pipeline()
        answers = answer_pipeline.run({
            "embedder": {"text": question}, 
            "builder": {"query": question}
        })
        
        # Menampilkan jawaban
        st.write("Answer:", answers['generator']['replies'][0])

if __name__ == "__main__":
    main()
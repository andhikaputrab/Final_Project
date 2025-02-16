import streamlit as st
from src.utils.config import config
from src.utils.styling import load_css
from src.utils.logger import default_logger as logger
from src.data.data_loader import WikipediaFetcher

st.set_page_config(
    page_title=config.get('PAGE_TITLE'),
    page_icon=config.get('PAGE_ICON'),
    layout=config.get('LAYOUT')
)

load_css()

st.title("🔍 Overview of the Biology Question Answering App")
st.markdown("""
**This app provides quick and accurate answers to your biology questions using advanced technologies. 
It pulls data from the Wikipedia API, giving access to a large, up-to-date knowledge base. 
The app also uses Ollama’s Large Language Model (LLM), an AI that understands natural language and generates clear, relevant explanations. 
By combining Wikipedia’s reliable data with Ollama’s AI, the app offers an easy and intuitive way to explore biology concepts and answer your questions.**

⚙️ How it Works:
- User Input: Type your question in the provided box.
- Data Processing: The app queries the Wikipedia API and uses Ollama’s LLM to process and generate a relevant response.
- Get Answer: Receive an accurate, well-explained answer!
""")

wikipedia_fetcher = WikipediaFetcher(language='id')

documents = wikipedia_fetcher.fetch_wikipedia_articles('fotosintesis')
logger.info('Fetching wikipedia article')

if documents:
    logger.info('Fetching wikipedia article successfully')
    st.subheader("Relevant Wikipedia Articles:")
    for doc in documents:
        st.markdown(f"**{doc['meta']['title']}**")
        st.write(doc['content'][:532])  # Show a snippet of the article content
else:
    st.write("No articles found for the given question.")
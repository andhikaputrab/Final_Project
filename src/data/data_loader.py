import wikipediaapi
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords

class WikipediaFetcher:
    def __init__(self, language):
        self.stop_words = stopwords.words('indonesian')
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.wikipedia = wikipediaapi.Wikipedia(user_agent='final_project', language=language, extract_format=wikipediaapi.ExtractFormat.WIKI)
        
    def extract_keywords(self, question, top_n=5):
        """
        Extract keywords with countVectorizer and cosine similarity.
        """
        
        count = CountVectorizer(ngram_range=(1, 2), stop_words=self.stop_words).fit([question])
        candidates = count.get_feature_names_out()
        
        doc_embedding = self.model.encode([question])
        candidate_embeddings = self.model.encode(candidates)
        
        # Hitung jarak cosine untuk setiap kandidat
        distances = cosine_similarity(doc_embedding, candidate_embeddings)
        keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]
        return keywords
    
    def fetch_wikipedia_articles(self, question):
        """
        Get wikipedia documents based on keywords from question
        """
        keywords = self.extract_keywords(question, top_n=5)
        documents = []
        
        for keyword in keywords:
            try:
                page = self.wikipedia.page(keyword)
                if page.exists():
                    documents.append({"content": page.text, "meta": {"title": keyword}})
            except wikipediaapi.exceptions.DisambiguationError as e:
                documents.append({"content": e.options[0], "meta": {"title": keyword}})
            except wikipediaapi.exceptions.HTTPTimeoutError:
                documents.append([])

        return documents
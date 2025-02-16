from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.mongodb_atlas import MongoDBAtlasEmbeddingRetriever
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaGenerator
from src.connection.db_connector import MongoDBConnector

class HaystackPipeline:
    def __init__(self, document_store):
        self.document_store = document_store
        self.pipeline = self.document_store_pipeline()

    def document_store_pipeline(self):
        pipeline_storing = Pipeline()

        # Preprocessing pipeline (cleaning, splitting, embedding, storing)
        pipeline_storing.add_component("cleaner", DocumentCleaner())
        pipeline_storing.add_component("splitter", DocumentSplitter(split_by="word", split_length=256, split_overlap=100))
        pipeline_storing.add_component("embedder", SentenceTransformersDocumentEmbedder())
        pipeline_storing.add_component("writer", DocumentWriter(document_store=self.document_store, policy=DuplicatePolicy.SKIP))

        # Connecting components in sequence
        pipeline_storing.connect("cleaner", "splitter")
        pipeline_storing.connect("splitter", "embedder")
        pipeline_storing.connect("embedder", "writer")

        return pipeline_storing

    def answer_generator_pipeline(self):
        # Template for prompt generation
        template = """
            given these documents, answer the question based on these documents in a sentence. Documents:
            {% for document in documents %}
                {{ document.content }}
            {% endfor %}
            Question: {{ query }}
        """

        # Ollama generator setup (with your chosen model)
        generator = OllamaGenerator(
            model="llama3.2",
            url="http://localhost:11434/api/generate"
        )
        
        # Answer generation pipeline (embedding, retrieval, prompt building, generation)
        pipeline_generate_answers = Pipeline()

        # Adding components to the pipeline
        pipeline_generate_answers.add_component('embedder', SentenceTransformersTextEmbedder())
        pipeline_generate_answers.add_component('retriever', MongoDBAtlasEmbeddingRetriever(document_store=self.document_store))
        pipeline_generate_answers.add_component('builder', PromptBuilder(template=template))
        pipeline_generate_answers.add_component('generator', generator)

        # Connecting components in sequence
        pipeline_generate_answers.connect("embedder", "retriever")
        pipeline_generate_answers.connect("retriever", "builder")
        pipeline_generate_answers.connect("builder", "generator")
        
        return pipeline_generate_answers
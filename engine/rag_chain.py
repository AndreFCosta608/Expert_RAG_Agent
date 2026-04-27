from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings

class RAGManager:
    
    def __init__(self, persist_directory="./db"):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = Chroma(
            persist_directory=persist_directory, 
            embedding_function=self.embeddings
        )
        self.llm = Ollama(model="mistral")

    def get_response(self, query, k=5):
        template = """You are an expert technical assistant. 
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know.

        {context}

        Question: {question}
        Expert Answer:"""
        
        qa_chain = RetrievalQA.from_chain_type(
            self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": k}),
            chain_type_kwargs={"prompt": PromptTemplate.from_template(template)},
            return_source_documents=True
        )
        
        return qa_chain({"query": query})
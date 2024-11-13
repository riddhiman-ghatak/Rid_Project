# agents.py
import os
from typing import List, Dict
import arxiv
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Set up the Groq API key
os.environ["GROQ_API_KEY"] = "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"

# Initialize models
llm = ChatGroq(model="llama3-8b-8192")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

class SearchAgent:
    def search_papers(self, topic: str, max_results: int = 10) -> List[Dict]:
        """Fetch papers from Arxiv based on a topic."""
        search = arxiv.Search(
            query=topic,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        papers = []
        for result in search.results():
            paper = {
                'id': result.entry_id,
                'title': result.title,
                'authors': [author.name for author in result.authors],
                'summary': result.summary,
                'published': result.published.strftime('%Y-%m-%d'),
                'url': result.pdf_url
            }
            papers.append(paper)
        
        return papers

class QAAgent:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Initialize the QA prompt template
        system_prompt = """You are a research paper analysis assistant. 
        Use the following pieces of retrieved context to answer the question about the research paper. 
        If you don't know the answer, say that you don't know. 
        Base your answer only on the provided context.
        
        Context:
        {context}"""
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        self.vectorstore = None
        
    def _create_vectorstore(self, paper_content: str):
        """Create a vector store from paper content."""
        # Split the content into chunks
        splits = self.text_splitter.create_documents([paper_content])
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        
    def answer_question(self, question: str, context: str) -> str:
        """Answer a question based on paper context using RAG."""
        try:
            # Create or update vector store with the paper content
            self._create_vectorstore(context)
            
            # Create the retrieval chain
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": 3}
            )
            
            # Create the QA chain
            doc_chain = create_stuff_documents_chain(llm, self.prompt)
            retrieval_chain = create_retrieval_chain(retriever, doc_chain)
            
            # Get the answer
            response = retrieval_chain.invoke({
                "input": question
            })
            
            return response["answer"]
            
        except Exception as e:
            print(f"Error in QA process: {e}")
            return f"Error generating answer: {str(e)}"

class FutureWorksAgent:
    def generate_future_directions(self, papers: List[Dict]) -> str:
        """Generate future research directions based on paper summaries."""
        summaries = "\n".join([f"Title: {p['title']}\nSummary: {p['summary']}\n" for p in papers])
        
        prompt = f"""Based on these recent research papers:
        {summaries}
        
        Please analyze the current trends and suggest 3-5 promising future research directions.
        For each direction, explain:
        1. The motivation
        2. Potential impact
        3. Technical challenges to overcome"""

        try:
            response = llm.invoke(prompt)
            return response
        except Exception as e:
            print(f"Error generating future directions: {e}")
            return "Error generating future research directions."
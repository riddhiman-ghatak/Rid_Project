import os
import arxiv
from typing import List, Dict
from database import Neo4jDatabase
from langchain_groq import ChatGroq

# Set up the Groq API key
os.environ["GROQ_API_KEY"] = "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"

# Initialize the LLM model for question answering and future works
llm = ChatGroq(model="llama3-8b-8192")

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

def store_papers_in_db(papers: List[Dict]):
    """Store a list of papers in the Neo4j database."""
    db = Neo4jDatabase()
    for paper in papers:
        db.save_paper(paper)
    db.close()

class QAAgent:
    def answer_question(self, question: str, context: str) -> str:
        """Answer a question based on a given context from the research paper."""
        prompt = f"""Given the following research paper context:
        {context}
        
        Please answer this question:
        {question}
        
        Provide a concise and accurate answer based only on the information given."""

        try:
            response = llm.generate(prompt)
            return response['response']
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "There was an error generating the answer."

class FutureWorksAgent:
    def generate_future_directions(self, papers: List[Dict]) -> str:
        """Generate future research directions based on a list of paper summaries."""
        summaries = "\n".join([f"Title: {p['title']}\nSummary: {p['summary']}\n" for p in papers])
        
        prompt = f"""Based on these recent research papers:
        {summaries}
        
        Please analyze the current trends and suggest 3-5 promising future research directions.
        For each direction, explain:
        1. The motivation
        2. Potential impact
        3. Technical challenges to overcome"""

        try:
            response = llm.generate(prompt)
            return response['response']
        except Exception as e:
            print(f"Error generating future directions: {e}")
            return "There was an error generating future research directions."

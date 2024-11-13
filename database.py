from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

class Neo4jDatabase:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def save_paper(self, paper_data):
        with self.driver.session() as session:
            session.execute_write(self._create_paper, paper_data)

    @staticmethod
    def _create_paper(tx, paper_data):
        query = '''
        CREATE (p:Paper {
            id: $id,
            title: $title,
            authors: $authors,
            summary: $summary,
            published: $published,
            url: $url
        })
        '''
        tx.run(query, paper_data)

    def get_papers_by_topic(self, topic: str):
        with self.driver.session() as session:
            result = session.execute_read(self._get_papers, topic)
            if not result:
                raise ValueError("No papers found for the specified topic.")
            return result

    @staticmethod
    def _get_papers(tx, topic: str):
        query = '''
        MATCH (p:Paper)
        WHERE p.title CONTAINS $topic OR p.summary CONTAINS $topic
        RETURN p
        ORDER BY p.published DESC
        '''
        results = tx.run(query, topic=topic)
        papers = [dict(record['p']) for record in results]
        
        for paper in papers:
            if 'summary' not in paper:
                raise ValueError("Missing 'summary' field in paper data.")
        
        return papers

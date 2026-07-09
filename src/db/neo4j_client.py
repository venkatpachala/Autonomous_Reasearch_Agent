"""
Neo4j Graph DB Client - Safe initialization
"""

from typing import Optional, Dict, Any
from loguru import logger
from neo4j import GraphDatabase, Driver

from src.config import settings


class Neo4jClient:
    def __init__(self):
        self.uri = settings.neo4j_uri
        self.user = settings.neo4j_user
        self.password = settings.neo4j_password
        self.driver: Optional[Driver] = None
        self._connect()

    def _connect(self):
        try:
            self.driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )
            self.driver.verify_connectivity()
            logger.success("Connected to Neo4j")
            self._setup_schema()
        except Exception as e:
            logger.warning(f"Neo4j not available: {e}. Graph storage will be skipped.")
            self.driver = None

    def _setup_schema(self):
        if not self.driver:
            return
        with self.driver.session() as session:
            session.run("""
                CREATE CONSTRAINT paper_arxiv_id IF NOT EXISTS
                FOR (p:Paper) REQUIRE p.arxiv_id IS UNIQUE
            """)
            session.run("""
                CREATE CONSTRAINT author_name IF NOT EXISTS
                FOR (a:Author) REQUIRE a.name IS UNIQUE
            """)

    def is_connected(self) -> bool:
        return self.driver is not None

    def create_paper_node(self, paper_data: Dict[str, Any]):
        if not self.driver:
            return
        query = """
        MERGE (p:Paper {arxiv_id: $arxiv_id})
        SET p += $paper_data
        """
        with self.driver.session() as session:
            session.run(query, arxiv_id=paper_data["arxiv_id"], paper_data=paper_data)

    def create_author_relationship(self, arxiv_id: str, author_name: str):
        if not self.driver:
            return
        query = """
        MATCH (p:Paper {arxiv_id: $arxiv_id})
        MERGE (a:Author {name: $author_name})
        MERGE (p)-[:AUTHORED_BY]->(a)
        """
        with self.driver.session() as session:
            session.run(query, arxiv_id=arxiv_id, author_name=author_name)

    def close(self):
        if self.driver:
            self.driver.close()


# Safe global instance
neo4j_client = Neo4jClient()
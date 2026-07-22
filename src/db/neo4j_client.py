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
            # Index names on Method, Dataset, Metric, Concept
            for label in ["Method", "Dataset", "Metric", "Concept"]:
                session.run(f"""
                    CREATE INDEX {label.lower()}_name IF NOT EXISTS
                    FOR (n:{label}) ON (n.name)
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

    def write_extracted_graph(self, paper_id: str, entities: list, relationships: list):
        """
        Merge extracted entity nodes and relationships to build property graph triplets.
        """
        if not self.driver:
            return

        with self.driver.session() as session:
            # 1. Merge entities and link to source Paper node
            for ent in entities:
                # Restrict labels to alphanumeric
                clean_type = "".join(c for c in ent.type if c.isalnum())
                if clean_type not in {"Method", "Dataset", "Metric", "Concept"}:
                    clean_type = "Concept"

                query_ent = f"MERGE (e:{clean_type} {{name: $name}}) SET e.description = $description"
                session.run(query_ent, name=ent.name, description=ent.description)

                # Create link from Paper node to Entity
                query_link = f"""
                MATCH (p:Paper {{arxiv_id: $paper_id}})
                MATCH (e:{clean_type} {{name: $name}})
                MERGE (p)-[:MENTIONS]->(e)
                """
                session.run(query_link, paper_id=paper_id, name=ent.name)

            # 2. Merge entity relationships
            for rel in relationships:
                clean_rel = "".join(c for c in rel.relation if c.isalnum() or c == "_").upper()
                if not clean_rel:
                    clean_rel = "RELATED_TO"

                # Match generically across any nodes (or fallback nodes)
                query_rel = f"""
                MATCH (s {{name: $source}}), (t {{name: $target}})
                MERGE (s)-[r:{clean_rel}]->(t)
                SET r.value = $value
                """
                session.run(query_rel, source=rel.source, target=rel.target, value=rel.value)

    def get_related_triplets(self, entity_names: List[str]) -> List[str]:
        """
        Query Neo4j for 1-hop and 2-hop connected relationships involving the specified entities.
        """
        if not self.driver or not entity_names:
            return []

        query = """
        MATCH (s)-[r]->(t)
        WHERE s.name IN $names OR t.name IN $names
        RETURN s.name as source, type(r) as relation, t.name as target, r.value as value, 
               labels(s)[0] as source_type, labels(t)[0] as target_type
        LIMIT 30
        """
        triplets = []
        try:
            with self.driver.session() as session:
                result = session.run(query, names=entity_names)
                for rec in result:
                    val = f" ({rec['value']})" if rec['value'] else ""
                    source_lbl = rec['source_type'] or "Node"
                    target_lbl = rec['target_type'] or "Node"
                    triplets.append(
                        f"({rec['source']}:{source_lbl}) -[:{rec['relation']}{val}]-> ({rec['target']}:{target_lbl})"
                    )
        except Exception as e:
            logger.warning(f"Failed to fetch related triplets from Neo4j: {e}")
        return triplets

    def close(self):
        if self.driver:
            self.driver.close()


# Safe global instance
neo4j_client = Neo4jClient()
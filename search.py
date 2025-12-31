import ollama
from db import Database
from typing import List, Dict
import json

class SearchEngine:
    def __init__(self):
        self.db = Database()
        self.embed_model = "nomic-embed-text"
        self.llm_model = "gpt-oss:20b-cloud"

    def hybrid_search(self, query: str, top_k: int = 5) -> Dict:
        # 1. Vector Search
        query_embedding = self.get_embedding(query)
        vector_results = self.vector_search(query_embedding, top_k)
        
        # 2. Extract Entities for Graph Search
        entities = self.extract_entities(query)
        graph_results = self.graph_search(entities)
        
        # 3. Combine Context
        context = "### Vector Context:\n"
        for res in vector_results:
            context += f"- {res}\n"
        
        context += "\n### Graph Context:\n"
        for res in graph_results:
            context += f"- {res}\n"
            
        # 4. Generate Answer
        answer = self.generate_answer(query, context)
        
        return {
            "answer": answer,
            "sources": {
                "vector_count": len(vector_results),
                "graph_count": len(graph_results),
                "entities_found": entities
            }
        }

    def get_embedding(self, text: str) -> List[float]:
        response = ollama.embeddings(model=self.embed_model, prompt=text)
        return response['embedding']

    def vector_search(self, embedding: List[float], top_k: int) -> List[str]:
        conn = self.db.connect_pg()
        if not conn: return []
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT content FROM chunks 
                    ORDER BY embedding <=> %s::vector 
                    LIMIT %s
                """, (embedding, top_k))
                rows = cur.fetchall()
                return [row[0] for row in rows]
        finally:
            self.db.release_pg(conn)

    def extract_entities(self, query: str) -> List[str]:
        prompt = f"Extract key entities (nouns) from this query. Return as a comma-separated list: {query}"
        response = ollama.generate(model=self.llm_model, prompt=prompt)
        entities = [e.strip() for e in response['response'].split(',')]
        return entities

    def graph_search(self, entities: List[str]) -> List[str]:
        driver = self.db.connect_neo4j()
        results = []
        with driver.session() as session:
            for entity in entities:
                # 2-hop traversal to find deeper context and relationships
                # Using toLower for better matching flexibility
                query = """
                MATCH (e:Entity)-[r*1..2]-(neighbor)
                WHERE toLower(e.name) CONTAINS toLower($name)
                RETURN DISTINCT e.name as s, type(r[0]) as p, neighbor.name as o
                LIMIT 15
                """
                res = session.run(query, name=entity)
                for record in res:
                    results.append(f"{record['s']} {record['p']} {record['o']}")
        return list(set(results)) # Deduplicate

    def generate_answer(self, query: str, context: str) -> str:
        prompt = f"""
        You are a helpful assistant. Use the following context to answer the user query.
        If the context does not contain enough information, say so.
        
        Context:
        {context}
        
        User Query: {query}
        """
        response = ollama.generate(model=self.llm_model, prompt=prompt)
        return response['response']

    def get_all_graph_data(self):
        driver = self.db.connect_neo4j()
        nodes = []
        edges = []
        with driver.session() as session:
            # Get nodes
            node_query = "MATCH (n:Entity) RETURN n.name as id, n.name as label, labels(n)[0] as type LIMIT 100"
            node_res = session.run(node_query)
            for record in node_res:
                nodes.append({
                    "id": record["id"],
                    "label": record["label"],
                    "type": record["type"]
                })
            
            # Get edges
            edge_query = "MATCH (s:Entity)-[r]->(o:Entity) RETURN s.name as source, type(r) as label, o.name as target LIMIT 100"
            edge_res = session.run(edge_query)
            for record in edge_res:
                edges.append({
                    "source": record["source"],
                    "label": record["label"],
                    "target": record["target"]
                })
        return nodes, edges

    def close(self):
        self.db.close()

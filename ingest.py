import os
import ollama
import json
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from db import Database
from typing import List, Dict

class Ingestor:
    def __init__(self):
        self.db = Database()
        self.converter = DocumentConverter()
        self.chunker = HybridChunker()
        self.embed_model = "nomic-embed-text"
        self.llm_model = "gpt-oss:20b-cloud"

    def process_file(self, file_path: str):
        print(f"Processing {file_path}...")
        result = self.converter.convert(file_path)
        doc = result.document
        chunks = list(self.chunker.chunk(doc))
        
        from concurrent.futures import ThreadPoolExecutor
        
        print(f"  Ingesting {len(chunks)} chunks in parallel...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(lambda x: self._process_chunk(x[1], x[0], file_path), enumerate(chunks)))

    def _process_chunk(self, chunk, index, file_path):
        text = chunk.text
        # 1. Store in Vector DB (PostgreSQL)
        embedding = self.get_embedding(text)
        self.store_vector(text, embedding, {"source": file_path, "chunk_id": index})
        
        # 2. Extract Triplets and Store in Graph DB (Neo4j)
        triplets = self.extract_triplets(text)
        self.store_graph(triplets)

    def get_embedding(self, text: str) -> List[float]:
        response = ollama.embeddings(model=self.embed_model, prompt=text)
        return response['embedding']

    def extract_triplets(self, text: str) -> List[Dict]:
        prompt = f"""
        Extract semantic triplets (Subject, Predicate, Object) from the following text.
        Return ONLY a JSON list of objects with "subject", "predicate", and "object" keys.
        Do not include any explanation.
        
        Text: {text}
        """
        try:
            response = ollama.generate(model=self.llm_model, prompt=prompt, format="json")
            data = json.loads(response['response'])
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "triplets" in data:
                return data["triplets"]
            return []
        except Exception as e:
            print(f"Error extracting triplets: {e}")
            return []

    def store_vector(self, text: str, embedding: List[float], metadata: Dict):
        conn = self.db.connect_pg()
        if not conn: return
        try:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO chunks (content, metadata, embedding) VALUES (%s, %s, %s)",
                    (text, json.dumps(metadata), embedding)
                )
        finally:
            self.db.release_pg(conn)

    def store_graph(self, triplets: List[Dict]):
        driver = self.db.connect_neo4j()
        with driver.session() as session:
            for t in triplets:
                # Basic cleaning
                s = str(t.get('subject', '')).strip()
                p = str(t.get('predicate', '')).strip().upper().replace(" ", "_")
                o = str(t.get('object', '')).strip()
                
                if s and p and o:
                    query = f"""
                    MERGE (s:Entity {{name: $s_name}})
                    MERGE (o:Entity {{name: $o_name}})
                    MERGE (s)-[r:{p}]->(o)
                    """
                    session.run(query, s_name=s, o_name=o)

    def close(self):
        self.db.close()

if __name__ == "__main__":
    # Test with a dummy file if needed, but primarily used by app.py
    pass

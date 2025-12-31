import os
import psycopg2
from psycopg2 import pool
from neo4j import GraphDatabase
from dotenv import load_dotenv
import time

load_dotenv()

# PostgreSQL configuration
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PWD = os.getenv("PG_PWD", "password")
PG_DB = os.getenv("PG_DB", "graphrag")

# Neo4j configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PWD = os.getenv("NEO4J_PWD", "password")

class Database:
    _pg_pool = None

    def __init__(self):
        self.neo4j_driver = None
        self._init_pg_pool()

    def _init_pg_pool(self):
        if Database._pg_pool is None:
            try:
                Database._pg_pool = pool.SimpleConnectionPool(
                    1, 10,
                    host=PG_HOST,
                    port=PG_PORT,
                    user=PG_USER,
                    password=PG_PWD,
                    dbname=PG_DB
                )
                print("PostgreSQL connection pool initialized")
            except Exception as e:
                print(f"Error initializing PG pool: {e}")

    def connect_pg(self):
        if Database._pg_pool:
            return Database._pg_pool.getconn()
        return None

    def release_pg(self, conn):
        if Database._pg_pool and conn:
            Database._pg_pool.putconn(conn)

    def connect_neo4j(self):
        retries = 5
        while retries > 0:
            try:
                if self.neo4j_driver is None:
                    self.neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))
                    self.neo4j_driver.verify_connectivity()
                    print("Connected to Neo4j")
                return self.neo4j_driver
            except Exception as e:
                print(f"Error connecting to Neo4j: {e}. Retrying...")
                time.sleep(2)
                retries -= 1
        raise Exception("Failed to connect to Neo4j")

    def init_db(self):
        # Initialize PostgreSQL
        conn = self.connect_pg()
        if conn:
            try:
                conn.autocommit = True
                with conn.cursor() as cur:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS chunks (
                            id SERIAL PRIMARY KEY,
                            content TEXT,
                            metadata JSONB,
                            embedding vector(768)
                        );
                    """)
            finally:
                self.release_pg(conn)
        
        # Initialize Neo4j constraints
        driver = self.connect_neo4j()
        with driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
            session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type)")

    def close(self):
        if self.pg_conn:
            self.pg_conn.close()
        if self.neo4j_driver:
            self.neo4j_driver.close()

if __name__ == "__main__":
    db = Database()
    db.init_db()
    db.close()

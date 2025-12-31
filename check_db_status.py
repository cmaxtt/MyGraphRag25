from db import Database

def check_neo4j():
    db = Database()
    try:
        driver = db.connect_neo4j()
        with driver.session() as session:
            # Check for generic Entity nodes
            result = session.run("MATCH (n) RETURN count(n) as count")
            total_nodes = result.single()["count"]
            
            # Check for relations
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            total_rels = result.single()["count"]
            
            # Sample some data
            result = session.run("MATCH (n) RETURN n.name as name, labels(n)[0] as label LIMIT 5")
            samples = result.values()
            
            print(f"Total Nodes: {total_nodes}")
            print(f"Total Relationships: {total_rels}")
            if samples:
                print("Sample Nodes:")
                for s in samples:
                    print(f" - [{s[1]}] {s[0]}")
            else:
                print("No nodes found in database.")
                
    except Exception as e:
        print(f"Error checking Neo4j: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    check_neo4j()

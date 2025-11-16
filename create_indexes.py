#!/usr/bin/env python3
"""
Script to manually create vector indexes for Neo4j Reddit database
"""
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def create_vector_indexes():
    """Create vector indexes for Reddit content using mxbai-embed-large (1024 dimensions)"""

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "research2025")

    driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        with driver.session() as session:
            # Create Reddit content embeddings index (1024 dimensions for mxbai-embed-large)
            print("Creating Reddit content embeddings vector index...")
            session.run("""
                CREATE VECTOR INDEX reddit_content_embeddings IF NOT EXISTS
                FOR (r:RedditContent)
                ON r.content_embedding
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 1024,
                        `vector.similarity_function`: 'cosine'
                    }
                }
            """)

            # Create topic embeddings index
            print("Creating topic embeddings vector index...")
            session.run("""
                CREATE VECTOR INDEX topic_embeddings IF NOT EXISTS
                FOR (t:Topic)
                ON t.embedding
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 1024,
                        `vector.similarity_function`: 'cosine'
                    }
                }
            """)

            print("✓ All vector indexes created successfully!")

    except Exception as e:
        print(f"Error creating vector indexes: {e}")
        return False

    finally:
        driver.close()

    return True

if __name__ == "__main__":
    success = create_vector_indexes()
    if success:
        print("\n✅ Vector indexes creation completed!")
    else:
        print("\n❌ Vector indexes creation failed!")

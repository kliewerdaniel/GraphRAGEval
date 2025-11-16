#!/usr/bin/env python3
"""
Script to create missing thread relationships (REPLIES_TO and BELONGS_TO_THREAD)
in the Neo4j database for Reddit data.
"""
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def create_thread_relationships():
    """Create REPLIES_TO and BELONGS_TO_THREAD relationships for Reddit content"""

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "research2025")

    driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        with driver.session() as session:
            print("Creating REPLIES_TO relationships...")

            # Create REPLIES_TO relationships where parent_id exists and is different from current id
            result = session.run("""
                MATCH (child:RedditContent)
                WHERE child.parent_id IS NOT NULL
                AND child.parent_id <> child.id
                AND child.parent_id <> ""
                WITH child,
                     CASE
                       WHEN child.parent_id STARTS WITH 't1_' THEN substring(child.parent_id, 3)
                       WHEN child.parent_id STARTS WITH 't3_' THEN substring(child.parent_id, 3)
                       ELSE child.parent_id
                     END AS processed_parent_id
                // Find parent nodes that have IDs starting with the processed parent_id
                MATCH (parent:RedditContent)
                WHERE parent.id STARTS WITH processed_parent_id + '_'
                AND parent.id <> child.id
                MERGE (child)-[:REPLIES_TO]->(parent)
                RETURN count(child) as replies_created
            """).single()

            replies_count = result['replies_created'] if result else 0
            print(f"Created {replies_count} REPLIES_TO relationships")

            print("Creating BELONGS_TO_THREAD relationships...")

            # Create BELONGS_TO_THREAD relationships where link_id exists and is different from current id
            result = session.run("""
                MATCH (comment:RedditContent)
                WHERE comment.link_id IS NOT NULL
                AND comment.link_id <> comment.id
                AND comment.link_id <> ""
                WITH comment,
                     CASE
                       WHEN comment.link_id STARTS WITH 't3_' THEN substring(comment.link_id, 3)
                       ELSE comment.link_id
                     END AS processed_thread_id
                // Find thread nodes that have IDs starting with the processed thread_id
                MATCH (thread:RedditContent)
                WHERE thread.id STARTS WITH processed_thread_id + '_'
                AND thread.id <> comment.id
                MERGE (comment)-[:BELONGS_TO_THREAD]->(thread)
                RETURN count(comment) as thread_links_created
            """).single()

            thread_count = result['thread_links_created'] if result else 0
            print(f"Created {thread_count} BELONGS_TO_THREAD relationships")

            print("✓ Thread relationships creation completed!")

    except Exception as e:
        print(f"Error creating thread relationships: {e}")
        return False

    finally:
        driver.close()

    return True

if __name__ == "__main__":
    success = create_thread_relationships()
    if success:
        print("\n✅ Thread relationships creation completed successfully!")
    else:
        print("\n❌ Thread relationships creation failed!")

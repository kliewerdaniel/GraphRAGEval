import ollama
from neo4j import GraphDatabase
from typing import List, Dict, Any
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class HybridRetriever:
    def __init__(self,
                 neo4j_uri=None,
                 neo4j_user=None,
                 neo4j_password=None,
                 ollama_model="mistral"):

        # Use environment variables if not provided
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USERNAME")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD")

        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password)
        )
        self.ollama_model = ollama_model

    def retrieve_context(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval combining:
        1. Vector similarity search
        2. Graph traversal for related concepts
        3. Citation network expansion
        """

        # Generate query embedding
        try:
            query_embedding = ollama.embeddings(
                model='nomic-embed-text',
                prompt=query
            )['embedding']
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return []

        with self.driver.session() as session:
            # Vector similarity search
            vector_results = session.run("""
                CALL db.index.vector.queryNodes(
                    'paper_abstracts',
                    $limit,
                    $query_embedding
                )
                YIELD node, score
                MATCH (node)<-[:AUTHORED]-(author:Author)
                OPTIONAL MATCH (node)-[:DISCUSSES]->(concept:Concept)

                RETURN
                    node.title AS title,
                    node.abstract AS abstract,
                    node.year AS year,
                    score AS relevance_score,
                    collect(DISTINCT author.name) AS authors,
                    collect(DISTINCT concept.name) AS concepts,
                    'vector_search' AS retrieval_method
                ORDER BY score DESC
                """,
                query_embedding=query_embedding,
                limit=limit
            ).data()

            # Graph traversal for cited papers
            graph_results = []
            if vector_results:
                try:
                    top_paper_title = vector_results[0]['title']

                    try:
                        cited_papers = session.run("""
                            MATCH (seed:Paper {title: $seed_title})
                            OPTIONAL MATCH (seed)-[:CITES]->(cited:Paper)
                            WHERE cited IS NOT NULL
                            RETURN DISTINCT cited.title AS title
                            """,
                            seed_title=top_paper_title
                        ).data()

                        # For now, skip graph traversal due to citation data absence
                        graph_results = []

                    except Exception as e:
                        print(f"Graph traversal failed: {e}")
                        graph_results = []
                except Exception as e:
                    print(f"Error in graph traversal: {e}")
                    graph_results = []

            # Combine and deduplicate
            all_results = vector_results + graph_results
            seen_titles = set()
            unique_results = []

            for result in all_results:
                if result.get('title') and result['title'] not in seen_titles:
                    seen_titles.add(result['title'])
                    unique_results.append(result)

            return sorted(unique_results,
                         key=lambda x: x.get('relevance_score', 0),
                         reverse=True)[:limit]

    def _extract_query_concepts(self, query: str) -> List[str]:
        """Extract key concepts from query using LLM"""
        try:
            response = ollama.generate(
                model=self.ollama_model,
                prompt=f"Extract 3-5 key technical concepts from this query: {query}. Return as comma-separated list.",
                options={'temperature': 0.1}
            )
            return [c.strip() for c in response['response'].split(',') if c.strip()]
        except Exception as e:
            print(f"Error extracting query concepts: {e}")
            # Fallback: simple keyword extraction
            return query.split()[:3]

    def search_by_concept(self, concepts: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """Search papers by specific concepts with graph traversal"""
        with self.driver.session() as session:
            results = session.run("""
                UNWIND $concepts AS concept_name
                MATCH (c:Concept {name: concept_name})
                MATCH (p:Paper)-[:DISCUSSES]->(c)
                MATCH (p)<-[:AUTHORED]-(a:Author)
                OPTIONAL MATCH (p)-[:DISCUSSES]->(related:Concept)
                WHERE related <> c

                RETURN
                    p.title AS title,
                    p.abstract AS abstract,
                    p.year AS year,
                    count(c) AS concept_matches,
                    collect(DISTINCT a.name) AS authors,
                    collect(DISTINCT related.name) AS related_concepts,
                    'concept_search' AS retrieval_method
                ORDER BY concept_matches DESC, p.year DESC
                LIMIT $limit
                """,
                concepts=concepts,
                limit=limit
            ).data()

            return results

    def find_similar_papers(self, paper_title: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find papers similar to the given paper using citation patterns"""
        with self.driver.session() as session:
            results = session.run("""
                MATCH (source:Paper {title: $paper_title})
                MATCH (source)-[:DISCUSSES]->(concept:Concept)<-[:DISCUSSES]-(similar:Paper)
                WHERE similar <> source
                MATCH (similar)<-[:AUTHORED]-(author:Author)

                RETURN
                    similar.title AS title,
                    similar.abstract AS abstract,
                    similar.year AS year,
                    count(concept) AS shared_concepts,
                    collect(DISTINCT author.name) AS authors,
                    collect(DISTINCT concept.name) AS concepts,
                    'similarity_search' AS retrieval_method
                ORDER BY shared_concepts DESC
                LIMIT $limit
                """,
                paper_title=paper_title,
                limit=limit
            ).data()

            return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test hybrid retrieval")
    parser.add_argument("--query", type=str, help="Query to test")
    parser.add_argument("--limit", type=int, default=5, help="Number of results")

    args = parser.parse_args()

    retriever = HybridRetriever()

    if args.query:
        results = retriever.retrieve_context(args.query, args.limit)

        print(f"Query: {args.query}")
        print(f"Results: {len(results)}")
        for i, result in enumerate(results):
            print(f"{i+1}. {result['title']} (Score: {result.get('relevance_score', 0):.3f})")
            print(f"   Authors: {', '.join(result.get('authors', []))}")
            print(f"   Method: {result.get('retrieval_method', 'unknown')}")
            if result.get('concepts'):
                print(f"   Concepts: {', '.join(result.get('concepts', [])[:3])}")
            print()
    else:
        print("Provide a query with --query")

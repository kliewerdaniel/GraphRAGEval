import ollama
from neo4j import GraphDatabase
from typing import List, Dict, Any
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class RedditRetriever:
    def __init__(self,
                 neo4j_uri=None,
                 neo4j_user=None,
                 neo4j_password=None,
                 ollama_model="llama3.2:3b",
                 embedding_model="mxbai-embed-large:latest"):

        # Use environment variables if not provided
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD")

        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password)
        )
        self.ollama_model = ollama_model
        self.embedding_model = embedding_model

    def retrieve_context(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval for Reddit data combining:
        1. Vector similarity search on content
        2. Topic-based graph traversal
        3. Author and subreddit relationships
        """

        # Generate query embedding
        try:
            query_embedding = ollama.embeddings(
                model=self.embedding_model,
                prompt=query
            )['embedding']
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return []

        with self.driver.session() as session:
            # Vector similarity search on Reddit content
            vector_results = session.run("""
                CALL db.index.vector.queryNodes(
                    'reddit_content_embeddings',
                    $limit,
                    $query_embedding
                )
                YIELD node, score
                MATCH (node)-[:AUTHORED_BY]->(author:RedditUser)
                OPTIONAL MATCH (node)-[:DISCUSSES]->(topic:Topic)
                OPTIONAL MATCH (node)-[:POSTED_IN]->(subreddit:Subreddit)
                OPTIONAL MATCH (node)-[:MENTIONS]->(entity:Entity)

                RETURN
                    node.id AS id,
                    node.raw_content AS content,
                    node.author AS author,
                    node.subreddit AS subreddit,
                    node.score AS score,
                    node.sentiment AS sentiment,
                    node.has_question AS has_question,
                    node.content_type_extracted AS content_type,
                    node.created_utc AS created_utc,
                    score AS relevance_score,
                    collect(DISTINCT topic.name) AS topics,
                    collect(DISTINCT entity.name) AS entities,
                    'vector_search' AS retrieval_method,
                    node.file_path AS file_path
                ORDER BY score DESC
                """,
                query_embedding=query_embedding,
                limit=limit
            ).data()

            # Topic-based expansion
            topic_expansion_results = []
            if vector_results and vector_results[0].get('topics'):
                try:
                    top_topics = vector_results[0]['topics'][:3]  # Top 3 topics

                    topic_results = session.run("""
                        UNWIND $topics AS topic_name
                        MATCH (t:Topic {name: topic_name})
                        MATCH (r:RedditContent)-[:DISCUSSES]->(t)
                        WHERE r.id <> $exclude_id
                        MATCH (r)-[:AUTHORED_BY]->(author:RedditUser)
                        OPTIONAL MATCH (r)-[:POSTED_IN]->(subreddit:Subreddit)

                        RETURN
                            r.id AS id,
                            r.raw_content AS content,
                            r.author AS author,
                            r.subreddit AS subreddit,
                            r.score AS score,
                            r.sentiment AS sentiment,
                            r.created_utc AS created_utc,
                            count(t) AS topic_matches,
                            collect(DISTINCT author.username) AS authors,
                            collect(DISTINCT subreddit.name) AS subreddits,
                            'topic_expansion' AS retrieval_method,
                            r.file_path AS file_path
                        ORDER BY topic_matches DESC, r.score DESC
                        LIMIT $limit
                        """,
                        topics=top_topics,
                        exclude_id=vector_results[0]['id'],
                        limit=limit // 2
                    ).data()

                    topic_expansion_results = topic_results

                except Exception as e:
                    print(f"Topic expansion failed: {e}")
                    topic_expansion_results = []

            # Thread and conversation expansion
            thread_results = []
            if vector_results:
                try:
                    # Find replies to top result and parent comments
                    thread_data = session.run("""
                        MATCH (seed:RedditContent {id: $seed_id})
                        OPTIONAL MATCH (seed)<-[:REPLIES_TO]-(reply:RedditContent)
                        OPTIONAL MATCH (seed)-[:REPLIES_TO]->(parent:RedditContent)
                        OPTIONAL MATCH (seed)-[:BELONGS_TO_THREAD]->(thread:RedditContent)

                        WITH seed, reply, parent, thread
                        WHERE reply IS NOT NULL OR parent IS NOT NULL OR thread IS NOT NULL

                        MATCH (content)
                        WHERE content = reply OR content = parent OR content = thread
                        MATCH (content)-[:AUTHORED_BY]->(author:RedditUser)
                        OPTIONAL MATCH (content)-[:POSTED_IN]->(subreddit:Subreddit)

                        RETURN DISTINCT
                            content.id AS id,
                            content.raw_content AS content,
                            content.author AS author,
                            content.subreddit AS subreddit,
                            content.score AS score,
                            content.sentiment AS sentiment,
                            content.created_utc AS created_utc,
                            'thread_context' AS retrieval_method,
                            content.file_path AS file_path,
                            CASE WHEN content = reply THEN 'reply' WHEN content = parent THEN 'parent' ELSE 'thread' END AS relationship_type
                        LIMIT $limit
                        """,
                        seed_id=vector_results[0]['id'],
                        limit=limit // 3
                    ).data()

                    thread_results = thread_data

                except Exception as e:
                    print(f"Thread expansion failed: {e}")
                    thread_results = []

            # Combine and deduplicate results
            all_results = vector_results + topic_expansion_results + thread_results
            seen_ids = set()
            unique_results = []

            for result in all_results:
                result_id = result.get('id')
                if result_id and result_id not in seen_ids:
                    seen_ids.add(result_id)
                    # Normalize content for display (truncate if too long)
                    if 'content' in result and result['content']:
                        result['content_preview'] = result['content'][:500] + '...' if len(result['content']) > 500 else result['content']
                    unique_results.append(result)

            # Sort by relevance score, prioritizing vector search results
            def sort_key(result):
                base_score = result.get('relevance_score', 0)
                method_bonus = 1.0 if result.get('retrieval_method') == 'vector_search' else 0.8
                return base_score * method_bonus

            unique_results.sort(key=sort_key, reverse=True)

            return unique_results[:limit]

    def _extract_query_concepts(self, query: str) -> List[str]:
        """Extract key concepts from query using LLM"""
        try:
            response = ollama.generate(
                model=self.ollama_model,
                prompt=f"Extract 3-5 key concepts from this Reddit search query: {query}. Return as comma-separated list.",
                options={'temperature': 0.1}
            )
            return [c.strip() for c in response['response'].split(',') if c.strip()]
        except Exception as e:
            print(f"Error extracting query concepts: {e}")
            # Fallback: simple keyword extraction
            return query.split()[:3]

    def search_by_topic(self, topics: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """Search Reddit content by specific topics"""
        with self.driver.session() as session:
            results = session.run("""
                UNWIND $topics AS topic_name
                MATCH (t:Topic {name: topic_name})
                MATCH (r:RedditContent)-[:DISCUSSES]->(t)
                MATCH (r)-[:AUTHORED_BY]->(author:RedditUser)
                OPTIONAL MATCH (r)-[:POSTED_IN]->(subreddit:Subreddit)

                RETURN
                    r.id AS id,
                    r.raw_content AS content,
                    r.author AS author,
                    r.subreddit AS subreddit,
                    r.score AS score,
                    r.sentiment AS sentiment,
                    r.has_question AS has_question,
                    r.created_utc AS created_utc,
                    count(t) AS topic_matches,
                    collect(DISTINCT author.username) AS authors,
                    collect(DISTINCT subreddit.name) AS subreddits,
                    'topic_search' AS retrieval_method,
                    r.file_path AS file_path
                ORDER BY topic_matches DESC, r.score DESC
                LIMIT $limit
                """,
                topics=topics,
                limit=limit
            ).data()

            return results

    def search_by_author(self, author: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search Reddit content by specific author"""
        with self.driver.session() as session:
            results = session.run("""
                MATCH (a:RedditUser {username: $author})
                MATCH (r:RedditContent)-[:AUTHORED_BY]->(a)
                OPTIONAL MATCH (r)-[:POSTED_IN]->(subreddit:Subreddit)
                OPTIONAL MATCH (r)-[:DISCUSSES]->(topic:Topic)

                RETURN
                    r.id AS id,
                    r.raw_content AS content,
                    r.author AS author,
                    r.subreddit AS subreddit,
                    r.score AS score,
                    r.sentiment AS sentiment,
                    r.has_question AS has_question,
                    r.created_utc AS created_utc,
                    collect(DISTINCT topic.name) AS topics,
                    collect(DISTINCT subreddit.name) AS subreddits,
                    'author_search' AS retrieval_method,
                    r.file_path AS file_path
                ORDER BY r.score DESC, r.created_utc DESC
                LIMIT $limit
                """,
                author=author,
                limit=limit
            ).data()

            return results

    def search_by_subreddit(self, subreddit: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search Reddit content in specific subreddit"""
        with self.driver.session() as session:
            results = session.run("""
                MATCH (s:Subreddit {name: $subreddit})
                MATCH (r:RedditContent)-[:POSTED_IN]->(s)
                MATCH (r)-[:AUTHORED_BY]->(author:RedditUser)
                OPTIONAL MATCH (r)-[:DISCUSSES]->(topic:Topic)

                RETURN
                    r.id AS id,
                    r.raw_content AS content,
                    r.author AS author,
                    r.subreddit AS subreddit,
                    r.score AS score,
                    r.sentiment AS sentiment,
                    r.has_question AS has_question,
                    r.created_utc AS created_utc,
                    collect(DISTINCT topic.name) AS topics,
                    'subreddit_search' AS retrieval_method,
                    r.file_path AS file_path
                ORDER BY r.score DESC, r.created_utc DESC
                LIMIT $limit
                """,
                subreddit=subreddit,
                limit=limit
            ).data()

            return results

    def find_similar_discussions(self, content_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find Reddit content similar to the given content using shared topics"""
        with self.driver.session() as session:
            results = session.run("""
                MATCH (seed:RedditContent {id: $content_id})
                MATCH (seed)-[:DISCUSSES]->(topic:Topic)<-[:DISCUSSES]-(similar:RedditContent)
                WHERE similar <> seed AND similar.id <> seed.id
                MATCH (similar)-[:AUTHORED_BY]->(author:RedditUser)
                OPTIONAL MATCH (similar)-[:POSTED_IN]->(subreddit:Subreddit)

                RETURN
                    similar.id AS id,
                    similar.raw_content AS content,
                    similar.author AS author,
                    similar.subreddit AS subreddit,
                    similar.score AS score,
                    similar.sentiment AS sentiment,
                    similar.created_utc AS created_utc,
                    count(topic) AS shared_topics,
                    collect(DISTINCT author.username) AS authors,
                    collect(DISTINCT subreddit.name) AS subreddits,
                    collect(DISTINCT topic.name) AS topics,
                    'similarity_search' AS retrieval_method,
                    similar.file_path AS file_path
                ORDER BY shared_topics DESC, similar.score DESC
                LIMIT $limit
                """,
                content_id=content_id,
                limit=limit
            ).data()

            return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Reddit hybrid retrieval")
    parser.add_argument("--query", type=str, help="Query to test")
    parser.add_argument("--limit", type=int, default=5, help="Number of results")
    parser.add_argument("--author", type=str, help="Search by author")
    parser.add_argument("--subreddit", type=str, help="Search by subreddit")

    args = parser.parse_args()

    retriever = RedditRetriever()

    if args.query:
        results = retriever.retrieve_context(args.query, args.limit)

        print(f"Query: {args.query}")
        print(f"Results: {len(results)}")
        for i, result in enumerate(results):
            print(f"{i+1}. {result['author']} in r/{result['subreddit']} (Score: {result.get('relevance_score', 0):.3f})")
            print(f"   Content: {result.get('content_preview', result.get('content', ''))[:100]}...")
            print(f"   Method: {result.get('retrieval_method', 'unknown')}")
            if result.get('topics'):
                print(f"   Topics: {', '.join(result.get('topics', [])[:3])}")
            print()

    elif args.author:
        results = retriever.search_by_author(args.author, args.limit)
        print(f"Author: {args.author}")
        print(f"Results: {len(results)}")
        for i, result in enumerate(results):
            print(f"{i+1}. Score: {result['score']}, Subreddit: r/{result['subreddit']}")
            print(f"   Content: {result.get('content', '')[:100]}...")

    elif args.subreddit:
        results = retriever.search_by_subreddit(args.subreddit, args.limit)
        print(f"Subreddit: r/{args.subreddit}")
        print(f"Results: {len(results)}")
        for i, result in enumerate(results):
            print(f"{i+1}. {result['author']} (Score: {result['score']})")
            print(f"   Content: {result.get('content', '')[:100]}...")

    else:
        print("Provide a query with --query, --author, or --subreddit")

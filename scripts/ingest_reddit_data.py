import ollama
from neo4j import GraphDatabase
from pathlib import Path
import json
import yaml
from dotenv import load_dotenv
import os
import hashlib
from typing import List, Dict, Any, Optional

# Load environment variables
load_dotenv()

class RedditGraphBuilder:
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

    def parse_reddit_markdown(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Parse Reddit markdown file and extract structured data"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

        # Split by --- separators
        sections = content.split('---\n')

        # Extract frontmatter (YAML metadata)
        metadata = {}
        content_start = 0

        if len(sections) > 1:
            try:
                metadata = yaml.safe_load(sections[1])
                content_start = 2
            except Exception as e:
                print(f"Error parsing metadata: {e}")
                # Try to parse as single frontmatter section
                try:
                    metadata = yaml.safe_load(sections[0])
                    content_start = 1
                except:
                    print(f"Fallback metadata parsing also failed: {e}")
                    return None
        else:
            # Some files might not have proper frontmatter
            # Create minimal metadata from filename
            filename = file_path.stem
            metadata = {
                'id': filename.split('.')[0] if '.' in filename else filename,
                'type': 'submission' if 'submissions' in str(file_path) else 'comment',
                'author': 'Unknown',
                'subreddit': 'Unknown',
                'created_utc': 'Unknown'
            }
            content_start = 1

        # Extract content sections for comments with multiple sections
        content_parts = {}
        raw_content = sections[content_start].strip() if content_start < len(sections) else ""

        if metadata.get('type') == 'comment' and len(sections) > content_start + 1:
            # Parse comment sections
            current_section = ""
            for section in sections[content_start:]:
                if section.strip().startswith('## '):
                    if current_section:
                        # Store previous section
                        content_parts[current_section] = content_parts.get(current_section, "")
                    lines = section.strip().split('\n', 1)
                    if len(lines) >= 1:
                        current_section = lines[0].replace('## ', '').strip()
                        if len(lines) > 1:
                            content_parts[current_section] = lines[1].strip()
                        else:
                            content_parts[current_section] = ""
                elif current_section and section.strip():
                    # Append to current section
                    if content_parts.get(current_section):
                        content_parts[current_section] += "\n\n" + section.strip()
                    else:
                        content_parts[current_section] = section.strip()
        else:
            # Simple content for submissions or malformed files
            for section in sections[content_start:]:
                if section.strip():
                    raw_content += section.strip() + "\n\n"

        # Clean up the raw content
        raw_content = raw_content.strip()
        if not raw_content and 'url' in metadata:
            raw_content = f"[External URL]({metadata['url']})"

        if not raw_content:
            raw_content = "No content available"

        return {
            'metadata': metadata,
            'content': content_parts,
            'raw_content': raw_content,
            'file_path': str(file_path)
        }

    def generate_reddit_embedding(self, text: str) -> List[float]:
        """Generate embeddings for Reddit content"""
        try:
            response = ollama.embeddings(
                model=self.embedding_model,
                prompt=text[:2000]  # Limit text length for embedding
            )
            return response['embedding']
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return []

    def extract_reddit_entities(self, comment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to extract entities and concepts from Reddit comment"""
        content = comment_data.get('raw_content', '')
        metadata = comment_data.get('metadata', {})

        # Prepare prompt for entity extraction
        prompt = f"""Analyze this Reddit comment and extract the following information:
        - Main topics discussed (2-4 key concepts)
        - Sentiment (positive, negative, neutral)
        - Key entities mentioned (people, organizations, technologies, etc.)
        - Whether this contains a question (yes/no)
        - Content type (discussion, answer, question, story, etc.)

        Comment:
        Author: {metadata.get('author', 'Unknown')}
        Subreddit: {metadata.get('subreddit', 'Unknown')}
        Score: {metadata.get('score', 0)}
        Content: {content[:1000]}...

        Return as JSON with keys: topics, sentiment, entities, has_question, content_type
        """

        try:
            response = ollama.generate(
                model=self.ollama_model,
                prompt=prompt,
                format='json'
            )

            entities = json.loads(response['response'])

            # Add metadata extraction
            entities.update({
                'author': metadata.get('author', 'Unknown'),
                'subreddit': metadata.get('subreddit', 'Unknown'),
                'score': int(metadata.get('score', 0)),
                'created_utc': metadata.get('created_utc', ''),
                'type': metadata.get('type', 'comment'),
                'link_id': metadata.get('link_id', ''),
                'parent_id': metadata.get('parent_id', '')
            })

            return entities

        except Exception as e:
            print(f"Error extracting entities with LLM: {e}")
            return {
                'topics': ['general_discussion'],
                'sentiment': 'neutral',
                'entities': [],
                'has_question': False,
                'content_type': 'comment',
                'author': metadata.get('author', 'Unknown'),
                'subreddit': metadata.get('subreddit', 'Unknown'),
                'score': int(metadata.get('score', 0)),
                'created_utc': metadata.get('created_utc', ''),
                'type': metadata.get('type', 'comment'),
                'link_id': metadata.get('link_id', ''),
                'parent_id': metadata.get('parent_id', '')
            }

    def create_reddit_node(self, comment_data: Dict[str, Any], file_path: Path):
        """Create Reddit content node with embeddings and relationships"""

        if not comment_data:
            return

        # Extract entities using LLM with error handling
        try:
            entities = self.extract_reddit_entities(comment_data)
        except Exception as e:
            print(f"⚠️ LLM entity extraction failed for {file_path}: {e}. Using fallback data.")
            # Fallback entities from metadata
            metadata = comment_data.get('metadata', {})
            entities = {
                'topics': ['general_discussion'],
                'sentiment': 'neutral',
                'entities': [],
                'has_question': False,
                'content_type': 'post',
                'author': metadata.get('author', 'Unknown'),
                'subreddit': metadata.get('subreddit', 'Unknown'),
                'score': int(metadata.get('score', 0)),
                'created_utc': metadata.get('created_utc', ''),
                'type': metadata.get('type', 'submission'),
                'link_id': metadata.get('link_id', ''),
                'parent_id': metadata.get('parent_id', '')
            }

        # Generate content hash for deduplication
        content_hash = hashlib.md5(comment_data['raw_content'].encode()).hexdigest()

        # Generate embedding for content with fallback
        try:
            content_embedding = self.generate_reddit_embedding(comment_data['raw_content'])
        except Exception as e:
            print(f"⚠️ Embedding generation failed for {file_path}: {e}. Using empty embeddings.")
            content_embedding = []

        # Create unique ID from file path
        node_id = file_path.stem

        with self.driver.session() as session:
            try:
                # Create the main Reddit content node
                session.run("""
                    MERGE (r:RedditContent {
                        id: $id,
                        content_hash: $content_hash,
                        subreddit: $subreddit,
                        author: $author,
                        created_utc: $created_utc,
                        score: $score,
                        content_type: $type,
                        content_embedding: $embedding
                    })
                    SET r.sentiment = $sentiment,
                        r.has_question = $has_question,
                        r.content_type_extracted = $content_type_extracted,
                        r.raw_content = $raw_content,
                        r.file_path = $file_path,
                        r.link_id = $link_id,
                        r.parent_id = $parent_id
                    """,
                    id=node_id,
                    content_hash=content_hash,
                    subreddit=entities['subreddit'],
                    author=entities['author'],
                    created_utc=entities['created_utc'],
                    score=entities['score'],
                    type=entities['type'],
                    embedding=content_embedding,
                    sentiment=entities['sentiment'],
                    has_question=entities['has_question'],
                    content_type_extracted=entities['content_type'],
                    raw_content=comment_data['raw_content'][:5000],  # Limit content size
                    file_path=str(file_path),
                    link_id=entities['link_id'],
                    parent_id=entities['parent_id']
                )

                # Create topic relationships
                for topic in entities.get('topics', []):
                    session.run("""
                        MERGE (t:Topic {name: $topic_name})
                        WITH t
                        MATCH (r:RedditContent {id: $reddit_id})
                        MERGE (r)-[:DISCUSSES]->(t)
                        """,
                        topic_name=topic.strip(),
                        reddit_id=node_id
                    )

                # Create entity relationships
                for entity in entities.get('entities', []):
                    session.run("""
                        MERGE (e:Entity {name: $entity_name, type: $entity_type})
                        WITH e
                        MATCH (r:RedditContent {id: $reddit_id})
                        MERGE (r)-[:MENTIONS]->(e)
                        """,
                        entity_name=entity.strip(),
                        entity_type='person',  # Default type, could be enhanced
                        reddit_id=node_id
                    )

                # Create subreddit relationship
                session.run("""
                    MERGE (s:Subreddit {name: $subreddit_name})
                    WITH s
                    MATCH (r:RedditContent {id: $reddit_id})
                    MERGE (r)-[:POSTED_IN]->(s)
                    """,
                    subreddit_name=entities['subreddit'],
                    reddit_id=node_id
                )

                # Create author relationship
                session.run("""
                    MERGE (a:RedditUser {username: $username})
                    WITH a
                    MATCH (r:RedditContent {id: $reddit_id})
                    MERGE (r)-[:AUTHORED_BY]->(a)
                    """,
                    username=entities['author'],
                    reddit_id=node_id
                )

                # Create thread relationships if parent/child IDs exist
                if entities.get('link_id') and entities.get('link_id') != entities.get('parent_id'):
                    parent_id = entities['parent_id'].replace('t1_', '').replace('t3_', '') if entities['parent_id'] else None
                    link_id = entities['link_id'].replace('t3_', '') if entities['link_id'] else None

                    if parent_id and parent_id != node_id:
                        session.run("""
                            MATCH (child:RedditContent {id: $child_id})
                            MATCH (parent:RedditContent {id: $parent_id})
                            MERGE (child)-[:REPLIES_TO]->(parent)
                            """,
                            child_id=node_id,
                            parent_id=parent_id
                        )

                    if link_id and link_id != node_id:
                        session.run("""
                            MATCH (comment:RedditContent {id: $comment_id})
                            MATCH (thread:RedditContent {id: $thread_id})
                            MERGE (comment)-[:BELONGS_TO_THREAD]->(thread)
                            """,
                            comment_id=node_id,
                            thread_id=link_id
                        )

                print(f"✓ Created Reddit node: {node_id} - {entities['author']} in r/{entities['subreddit']}")

            except Exception as e:
                print(f"Error creating Reddit node for {file_path}: {e}")

    def ingest_reddit_directory(self, reddit_dir: Path):
        """Ingest all Reddit markdown files in a directory"""

        if not reddit_dir.exists():
            print(f"Directory {reddit_dir} does not exist")
            return

        # Look for markdown files in comments and submissions directories
        comments_dir = reddit_dir / "comments"
        submissions_dir = reddit_dir / "submissions"

        markdown_files = []

        if comments_dir.exists():
            markdown_files.extend(list(comments_dir.glob("*.md")))

        if submissions_dir.exists():
            markdown_files.extend(list(submissions_dir.glob("*.md")))

        print(f"Found {len(markdown_files)} Reddit markdown files to ingest...")

        for md_file in markdown_files:
            print(f"Processing: {md_file.name}")
            try:
                comment_data = self.parse_reddit_markdown(md_file)
                if comment_data:
                    self.create_reddit_node(comment_data, md_file)
                    print(f"✓ Ingested: {md_file.name}")
                else:
                    print(f"✗ Failed to parse: {md_file.name}")
            except Exception as e:
                print(f"✗ Failed {md_file.name}: {e}")

    def create_vector_indexes(self):
        """Create vector indexes for Reddit content"""
        with self.driver.session() as session:
            try:
                # Reddit content embeddings (1024 dimensions for mxbai-embed-large)
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

                # Topic embeddings
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

                print("✓ Vector indexes created for Reddit data")

            except Exception as e:
                print(f"Error creating vector indexes: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest Reddit markdown data into Neo4j graph")
    parser.add_argument("--directory", type=str, default="../reddit_export",
                       help="Directory containing Reddit markdown files")
    parser.add_argument("--setup-indexes", action="store_true",
                       help="Create vector indexes after ingestion")

    args = parser.parse_args()

    builder = RedditGraphBuilder()

    # Ingest Reddit data
    reddit_dir = Path(args.directory)
    builder.ingest_reddit_directory(reddit_dir)

    # Setup indexes if requested
    if args.setup_indexes:
        builder.create_vector_indexes()

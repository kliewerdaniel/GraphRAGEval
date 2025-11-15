import ollama
from neo4j import GraphDatabase
from pathlib import Path
import PyPDF2
import json
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class ResearchGraphBuilder:
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

    def extract_paper_metadata(self, pdf_path: Path) -> dict:
        """Extract title, abstract, and key sections from PDF"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)

                # Extract first 3 pages (usually contains abstract)
                text = ""
                for i in range(min(3, len(reader.pages))):
                    text += reader.pages[i].extract_text()
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            return None

        # Use Ollama to extract structured metadata
        prompt = f"""Extract from this research paper excerpt:
        1. Title
        2. Authors (list, comma-separated)
        3. Abstract (full abstract text)
        4. Year of publication
        5. Key concepts (3-5 main topics, comma-separated)

        If any information is not available, use "Unknown".

        Text: {text[:2000]}...

        Return as JSON with keys: title, authors, abstract, year, concepts"""

        try:
            response = ollama.generate(
                model=self.ollama_model,
                prompt=prompt,
                format='json'
            )

            return json.loads(response['response'])
        except Exception as e:
            print(f"Error extracting metadata with Ollama: {e}")
            return None

    def create_paper_node(self, metadata: dict, pdf_path: Path):
        """Create Paper node with embeddings"""

        if not metadata:
            return

        # Generate embedding for abstract
        try:
            abstract_embedding = ollama.embeddings(
                model='nomic-embed-text',
                prompt=metadata.get('abstract', metadata.get('title', ''))
            )['embedding']
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            abstract_embedding = None

        with self.driver.session() as session:
            try:
                session.run("""
                    CREATE (p:Paper {
                        title: $title,
                        abstract: $abstract,
                        year: $year,
                        pdf_path: $pdf_path,
                        abstract_embedding: $embedding
                    })
                    WITH p
                    UNWIND $authors AS author_name
                    MERGE (a:Author {name: author_name})
                    CREATE (a)-[:AUTHORED]->(p)

                    WITH p
                    UNWIND $concepts AS concept_name
                    MERGE (c:Concept {name: concept_name})
                    CREATE (p)-[:DISCUSSES]->(c)
                    """,
                    title=metadata.get('title', 'Unknown'),
                    abstract=metadata.get('abstract', ''),
                    year=metadata.get('year', 2024),
                    pdf_path=str(pdf_path),
                    embedding=abstract_embedding,
                    authors=metadata.get('authors', '').split(',') if isinstance(metadata.get('authors'), str) else metadata.get('authors', []),
                    concepts=metadata.get('concepts', '').split(',') if isinstance(metadata.get('concepts'), str) else metadata.get('concepts', [])
                )
                print(f"✓ Created paper node: {metadata.get('title', 'Unknown')}")
            except Exception as e:
                print(f"Error creating paper node: {e}")

    def ingest_directory(self, papers_dir: Path):
        """Ingest all PDFs in a directory"""

        if not papers_dir.exists():
            print(f"Directory {papers_dir} does not exist")
            return

        pdf_files = list(papers_dir.glob("*.pdf"))

        print(f"Found {len(pdf_files)} papers to ingest...")

        for pdf_path in pdf_files:
            print(f"Processing: {pdf_path.name}")
            try:
                metadata = self.extract_paper_metadata(pdf_path)
                if metadata:
                    self.create_paper_node(metadata, pdf_path)
                    print(f"✓ Ingested: {metadata.get('title', 'Unknown')}")
                else:
                    print(f"✗ Failed to extract metadata: {pdf_path.name}")
            except Exception as e:
                print(f"✗ Failed {pdf_path.name}: {e}")

    def create_vector_indexes(self):
        """Create vector indexes for similarity search"""
        with self.driver.session() as session:
            try:
                # Abstract embeddings (768 dimensions for nomic-embed-text)
                session.run("""
                    CREATE VECTOR INDEX paper_abstracts IF NOT EXISTS
                    FOR (p:Paper)
                    ON p.abstract_embedding
                    OPTIONS {
                        indexConfig: {
                            `vector.dimensions`: 768,
                            `vector.similarity_function`: 'cosine'
                        }
                    }
                """)

                # Concept embeddings (also 768 dimensions for nomic-embed-text)
                session.run("""
                    CREATE VECTOR INDEX concept_definitions IF NOT EXISTS
                    FOR (c:Concept)
                    ON c.definition_embedding
                    OPTIONS {
                        indexConfig: {
                            `vector.dimensions`: 768,
                            `vector.similarity_function`: 'cosine'
                        }
                    }
                """)

                # Note embeddings (also 768 dimensions for nomic-embed-text)
                session.run("""
                    CREATE VECTOR INDEX note_contents IF NOT EXISTS
                    FOR (n:Note)
                    ON n.content_embedding
                    OPTIONS {
                        indexConfig: {
                            `vector.dimensions`: 768,
                            `vector.similarity_function`: 'cosine'
                        }
                    }
                """)

                print("✓ Vector indexes created")

            except Exception as e:
                print(f"Error creating vector indexes: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest research papers into Neo4j graph")
    parser.add_argument("--directory", type=str, default="data/research_papers",
                       help="Directory containing PDF files")
    parser.add_argument("--setup-indexes", action="store_true",
                       help="Create vector indexes after ingestion")

    args = parser.parse_args()

    builder = ResearchGraphBuilder()

    # Ingest papers
    papers_dir = Path(args.directory)
    builder.ingest_directory(papers_dir)

    # Setup indexes if requested
    if args.setup_indexes:
        builder.create_vector_indexes()

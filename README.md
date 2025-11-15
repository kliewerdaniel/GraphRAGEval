---
layout: post
title: "Building and Evaluating a Local-First Research Assistant with GraphRAG and vero-eval"
date: 11-15-2025
author: "Daniel Kliewer"
description: "Complete technical guide to building a production-ready research assistant using GraphRAG, Neo4j knowledge graphs, Ollama local LLMs, and vero-eval evaluation framework for rigorous AI system testing."
tags: ["AI", "GraphRAG", "Local LLM", "Neo4j", "Ollama", "vero-eval", "Research Assistant", "Knowledge Graph", "RAG", "AI Evaluation"]
canonical_url: "/blog/building-evaluating-local-research-assistant-graphrag-vero-eval"
image: "/images/11152025/graphrag-research-assistant-architecture.png"
og:title: "Building and Evaluating a Local-First Research Assistant with GraphRAG and vero-eval"
og:description: "Master GraphRAG implementation with Neo4j, Ollama, and vero-eval for production-ready AI research assistants. Complete guide with code examples and evaluation frameworks."
og:image: "/images/11152025/graphrag-research-assistant-architecture.png"
og:url: "https://danielkliewer.com/blog/building-evaluating-local-research-assistant-graphrag-vero-eval"
og:type: "article"
twitter:card: "summary_large_image"
twitter:title: "Local Research Assistant with GraphRAG & vero-eval"
twitter:description: "Build production-ready AI research assistants using GraphRAG, Neo4j, Ollama, and vero-eval evaluation framework."
twitter:image: "/images/11152025/graphrag-research-assistant-architecture.png"
---

![Complete GraphRAG Research Assistant Architecture Overview](/images/11152025/graphrag-research-assistant-architecture.png)


# Building and Evaluating a Local-First Research Assistant with GraphRAG and vero-eval

*A comprehensive guide to creating a persona-driven AI assistant with rigorous evaluation using Neo4j, Ollama, and the vero-eval framework*

## Introduction: Why Local GraphRAG Matters for Research Workflows

If you're building AI-powered applications in 2025, you've likely hit two major pain points: **context limitations** and **lack of systematic evaluation**. Large Language Models are powerful, but they struggle with long-term memory and consistent performance across edge cases. Enter GraphRAG—a methodology that combines knowledge graphs with retrieval-augmented generation to give your AI genuine memory and contextual awareness.

In this guide, we'll build a **Local Research Assistant** that:
- Stores and retrieves research papers, notes, and conversations in a Neo4j knowledge graph
- Uses Ollama for completely local inference (no API costs, full privacy)
- Implements persona-driven responses that adapt based on RLHF feedback
- **Most importantly**: Measures performance rigorously using the [vero-eval framework](https://github.com/vero-labs-ai/vero-eval)

This isn't another "hello world" tutorial. We're building production-ready infrastructure that you can deploy for real research workflows, with proper testing and evaluation baked in from day one.

## Prerequisites and Starting Point

Before we dive in, you'll need:

**System Requirements:**
- Python 3.9+
- Node.js 18+
- Docker (for Neo4j)
- 16GB+ RAM recommended

**Core Technologies:**
- [Ollama](https://ollama.ai) for local LLM inference
- [Neo4j](https://neo4j.com) for graph database
- [vero-eval](https://github.com/vero-labs-ai/vero-eval) for evaluation
- Next.js + FastAPI (from the starter template)



**Clone the Starter Repository:**

```bash
git clone https://github.com/kliewerdaniel/chrisbot.git research-assistant
cd research-assistant
```

This gives us a solid foundation with the frontend, basic chat interface, and project structure already in place. We'll extend it to build our research-focused GraphRAG system.

## Part 1: Understanding the Architecture

Our Research Assistant follows the **PersonaGen architecture** pattern outlined by Daniel Kliewer, but applied to academic research workflows:

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
│              (Next.js Chat Interface)                    │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                 Reasoning Agent                          │
│      (Tool Calling + RLHF Threshold Logic)              │
└────────────────────┬────────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
┌──────────────────┐   ┌──────────────────┐
│   Neo4j Graph    │   │  Ollama LLM      │
│   RAG System     │   │  (Mistral/Llama) │
│                  │   │                  │
│ • Papers         │   │ • Generation     │
│ • Authors        │   │ • Embeddings     │
│ • Concepts       │   │ • Extraction     │
│ • Citations      │   │                  │
└──────────────────┘   └──────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│              vero-eval Framework                         │
│  • Test Dataset Generation                              │
│  • Retrieval Metrics (Precision, Recall, MRR)          │
│  • Generation Metrics (Faithfulness, BERTScore)        │
│  • Persona Stress Testing                               │
└─────────────────────────────────────────────────────────┘
```


**Key Insight**: The persona system adapts its behavior based on evaluation feedback. If vero-eval shows poor retrieval for technical queries, the RLHF thresholds adjust to require more context before responding.

## Part 2: Setting Up Neo4j GraphRAG

Neo4j is our memory layer. Following the [official Neo4j GenAI integration patterns](https://neo4j.com/docs/cypher-manual/current/genai-integrations/), we'll create a graph schema optimized for research.

### Installing Neo4j GraphRAG for Python

```bash
# Install the official Neo4j GraphRAG package
pip install neo4j-graphrag

# Install Ollama integration
pip install "neo4j-graphrag[ollama]"

# Start Neo4j (using Docker)
docker run \
    --name research-neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/research2025 \
    -v $PWD/neo4j-data:/data \
    neo4j:latest
```

![Neo4j Knowledge Graph Setup for Research Assistant](/images/11152025/neo4j-knowledge-graph-setup.png)

### Defining the Research Knowledge Schema

Create `scripts/graph_schema.py`:

```python
from neo4j_graphrag import GraphSchema
from dataclasses import dataclass

@dataclass
class ResearchSchema(GraphSchema):
    """
    Knowledge graph schema for research assistant.
    
    Nodes:
    - Paper: Research papers with metadata
    - Author: Paper authors with affiliation
    - Concept: Extracted key concepts/topics
    - Note: User's research notes
    - Question: User queries with context
    
    Relationships:
    - AUTHORED: Author -> Paper
    - CITES: Paper -> Paper
    - DISCUSSES: Paper -> Concept
    - RELATES_TO: Concept -> Concept
    - ANSWERS: Paper -> Question
    """
    
    node_types = {
        'Paper': {
            'properties': ['title', 'abstract', 'year', 'doi', 'pdf_path'],
            'embedding_property': 'abstract_embedding'
        },
        'Author': {
            'properties': ['name', 'affiliation', 'h_index'],
            'embedding_property': None
        },
        'Concept': {
            'properties': ['name', 'definition', 'domain'],
            'embedding_property': 'definition_embedding'
        },
        'Note': {
            'properties': ['content', 'timestamp', 'tags'],
            'embedding_property': 'content_embedding'
        },
        'Question': {
            'properties': ['query', 'timestamp', 'answered'],
            'embedding_property': 'query_embedding'
        }
    }
    
    relationship_types = {
        'AUTHORED': ('Author', 'Paper'),
        'CITES': ('Paper', 'Paper'),
        'DISCUSSES': ('Paper', 'Concept'),
        'RELATES_TO': ('Concept', 'Concept'),
        'ANSWERS': ('Paper', 'Question'),
        'ANNOTATES': ('Note', 'Paper')
    }
```

**Why this schema?** Research workflows have natural graph structures:
- Papers cite each other (transitive relationships)
- Concepts relate to multiple papers
- Authors collaborate across papers
- User notes connect to specific papers

This lets us traverse the graph to find: "What papers discussing transformer architectures were cited by papers on RAG systems after 2023?"

### Building the Graph Ingestion Pipeline

Create `scripts/ingest_research_data.py`:

```python
import ollama
from neo4j import GraphDatabase
from neo4j_graphrag import GraphRAG
from pathlib import Path
import PyPDF2

class ResearchGraphBuilder:
    def __init__(self, neo4j_uri="bolt://localhost:7687", 
                 neo4j_user="neo4j", 
                 neo4j_password="research2025",
                 ollama_model="mistral"):
        
        self.driver = GraphDatabase.driver(neo4j_uri, 
                                          auth=(neo4j_user, neo4j_password))
        self.ollama_model = ollama_model
        self.graph_rag = GraphRAG(self.driver)
        
    def extract_paper_metadata(self, pdf_path: Path) -> dict:
        """Extract title, abstract, and key sections from PDF"""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            # Extract first 3 pages (usually contains abstract)
            text = ""
            for page in reader.pages[:3]:
                text += page.extract_text()
        
        # Use Ollama to extract structured metadata
        prompt = f"""Extract from this research paper excerpt:
        1. Title
        2. Authors (list)
        3. Abstract
        4. Key concepts (5-7 main topics)
        
        Text: {text[:4000]}
        
        Return as JSON."""
        
        response = ollama.generate(
            model=self.ollama_model,
            prompt=prompt,
            format='json'
        )
        
        return json.loads(response['response'])
    
    def create_paper_node(self, metadata: dict, pdf_path: Path):
        """Create Paper node with embeddings"""
        
        # Generate embedding for abstract
        abstract_embedding = ollama.embeddings(
            model='nomic-embed-text',
            prompt=metadata['abstract']
        )['embedding']
        
        with self.driver.session() as session:
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
                title=metadata['title'],
                abstract=metadata['abstract'],
                year=metadata.get('year', 2024),
                pdf_path=str(pdf_path),
                embedding=abstract_embedding,
                authors=metadata['authors'],
                concepts=metadata['concepts']
            )
    
    def ingest_directory(self, papers_dir: Path):
        """Ingest all PDFs in a directory"""
        pdf_files = list(papers_dir.glob("*.pdf"))
        
        print(f"Found {len(pdf_files)} papers to ingest...")
        
        for pdf_path in pdf_files:
            print(f"Processing: {pdf_path.name}")
            try:
                metadata = self.extract_paper_metadata(pdf_path)
                self.create_paper_node(metadata, pdf_path)
                print(f"✓ Ingested: {metadata['title']}")
            except Exception as e:
                print(f"✗ Failed {pdf_path.name}: {e}")
```

**Key Pattern**: We're using Ollama for both extraction (via `generate`) and embeddings (via `embeddings`). This keeps everything local. For production, you might cache embeddings in a vector index.

### Creating Vector Indexes for Hybrid Search

Following [Neo4j's GenAI integration guide](https://neo4j.com/docs/cypher-manual/current/genai-integrations/), we create vector indexes:

```python
def create_vector_indexes(self):
    """Create vector indexes for similarity search"""
    with self.driver.session() as session:
        # Abstract embeddings (4096 dimensions for nomic-embed-text)
        session.run("""
            CREATE VECTOR INDEX paper_abstracts IF NOT EXISTS
            FOR (p:Paper)
            ON p.abstract_embedding
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: 4096,
                    `vector.similarity_function`: 'cosine'
                }
            }
        """)
        
        # Concept embeddings
        session.run("""
            CREATE VECTOR INDEX concept_definitions IF NOT EXISTS
            FOR (c:Concept)
            ON c.definition_embedding
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: 4096,
                    `vector.similarity_function`: 'cosine'
                }
            }
        """)
        
        # Note embeddings
        session.run("""
            CREATE VECTOR INDEX note_contents IF NOT EXISTS
            FOR (n:Note)
            ON n.content_embedding
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: 4096,
                    `vector.similarity_function`: 'cosine'
                }
            }
        """)
```

**Critical**: The dimension count (4096) must match your embedding model. `nomic-embed-text` uses 4096, but if you switch to `all-MiniLM-L6-v2`, you'd need 384.

## Part 3: Implementing Hybrid Retrieval

Now we implement the retrieval layer that combines vector similarity with graph traversal:

![Hybrid Retrieval System Architecture](/images/11152025/hybrid-retrieval-system.png)

```python
class HybridRetriever:
    def __init__(self, driver, ollama_model="mistral"):
        self.driver = driver
        self.ollama_model = ollama_model
    
    def retrieve_context(self, query: str, limit: int = 5) -> list[dict]:
        """
        Hybrid retrieval combining:
        1. Vector similarity search
        2. Graph traversal for related concepts
        3. Citation network expansion
        """
        
        # Generate query embedding
        query_embedding = ollama.embeddings(
            model='nomic-embed-text',
            prompt=query
        )['embedding']
        
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
                MATCH (node)-[:DISCUSSES]->(concept:Concept)
                
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
                top_paper_title = vector_results[0]['title']
                
                graph_results = session.run("""
                    MATCH (seed:Paper {title: $seed_title})
                    MATCH (seed)-[:CITES]->(cited:Paper)
                    MATCH (cited)<-[:AUTHORED]-(author:Author)
                    MATCH (cited)-[:DISCUSSES]->(concept:Concept)
                    WHERE any(c IN $query_concepts WHERE c IN collect(concept.name))
                    
                    RETURN 
                        cited.title AS title,
                        cited.abstract AS abstract,
                        cited.year AS year,
                        0.7 AS relevance_score,
                        collect(DISTINCT author.name) AS authors,
                        collect(DISTINCT concept.name) AS concepts,
                        'citation_traversal' AS retrieval_method
                    LIMIT $limit
                    """,
                    seed_title=top_paper_title,
                    query_concepts=self._extract_query_concepts(query),
                    limit=limit // 2
                ).data()
            
            # Combine and deduplicate
            all_results = vector_results + graph_results
            seen_titles = set()
            unique_results = []
            
            for result in all_results:
                if result['title'] not in seen_titles:
                    seen_titles.add(result['title'])
                    unique_results.append(result)
            
            return sorted(unique_results, 
                         key=lambda x: x['relevance_score'], 
                         reverse=True)[:limit]
    
    def _extract_query_concepts(self, query: str) -> list[str]:
        """Extract key concepts from query using LLM"""
        response = ollama.generate(
            model=self.ollama_model,
            prompt=f"Extract 3-5 key technical concepts from this query: {query}. Return as comma-separated list.",
            options={'temperature': 0.1}
        )
        return [c.strip() for c in response['response'].split(',')]
```

**Why hybrid?** Pure vector search might miss important papers that don't match semantically but are cited by relevant papers. Graph traversal captures these relationships.

## Part 4: The Reasoning Agent and Persona Layer

The reasoning agent decides when to query the graph and how to format responses based on RLHF-adjusted thresholds:

![Persona-Driven Response System Architecture](/images/11152025/persona-driven-responses.png)

```python
# In scripts/reasoning_agent.py

import json
from pathlib import Path

class PersonaReasoningAgent:
    def __init__(self, persona_config_path: Path = Path("data/persona.json")):
        self.persona_config = self._load_persona(persona_config_path)
        self.retriever = HybridRetriever(driver, ollama_model)
        
    def _load_persona(self, config_path: Path) -> dict:
        """Load persona configuration with RLHF thresholds"""
        with open(config_path) as f:
            return json.load(f)
    
    def should_retrieve_context(self, query: str) -> bool:
        """
        Decide if we need to retrieve context based on:
        1. Query complexity
        2. RLHF confidence threshold
        3. Recent retrieval success rate
        """
        
        # Simple heuristic: technical terms or specific paper requests
        technical_indicators = [
            'paper', 'research', 'study', 'findings',
            'method', 'algorithm', 'experiment', 'results'
        ]
        
        needs_retrieval = any(term in query.lower() 
                             for term in technical_indicators)
        
        # Check RLHF threshold
        confidence_threshold = self.persona_config['rlhf_thresholds']['retrieval_required']
        
        # If recent queries had low-quality responses, lower threshold
        if self.persona_config['recent_success_rate'] < 0.7:
            confidence_threshold *= 0.8
        
        return needs_retrieval or confidence_threshold > 0.5
    
    def generate_response(self, query: str, chat_history: list = None) -> dict:
        """
        Main orchestration logic:
        1. Decide if retrieval needed
        2. Retrieve context if necessary
        3. Generate response with persona coloring
        4. Grade output (RLHF scoring)
        """
        
        # Step 1: Retrieval decision
        needs_context = self.should_retrieve_context(query)
        
        context_docs = []
        if needs_context:
            context_docs = self.retriever.retrieve_context(query, limit=5)
        
        # Step 2: Format context for LLM
        context_str = self._format_context(context_docs)
        
        # Step 3: Generate with persona
        system_prompt = self._build_persona_prompt(context_str)
        
        response = ollama.generate(
            model='mistral',
            prompt=query,
            system=system_prompt,
            context=chat_history
        )
        
        # Step 4: RLHF grading
        quality_grade = self._grade_response(query, response['response'], context_docs)
        
        # Update RLHF thresholds based on grade
        self._update_persona_thresholds(quality_grade)
        
        return {
            'response': response['response'],
            'context_used': context_docs,
            'quality_grade': quality_grade,
            'retrieval_method': context_docs[0]['retrieval_method'] if context_docs else None
        }
    
    def _build_persona_prompt(self, context: str) -> str:
        """
        Build system prompt from persona configuration.
        This is the 'coloring' step mentioned in the architecture.
        """
        base_template = self.persona_config['system_prompt_template']
        
        # Insert context if available
        if context:
            base_template += f"\n\nRelevant Research Context:\n{context}"
        
        # Add persona modifiers based on RLHF values
        formality = self.persona_config['rlhf_thresholds']['formality_level']
        if formality > 0.7:
            base_template += "\n\nUse academic, formal language with proper citations."
        else:
            base_template += "\n\nExplain concepts clearly and conversationally."
        
        return base_template
    
    def _grade_response(self, query: str, response: str, context: list) -> float:
        """
        RLHF grading: 0 (needs improvement) to 1 (excellent).
        In production, this would be human feedback, but we start with heuristics.
        """
        
        # Heuristic checks:
        # 1. Did we use retrieved context?
        used_context = any(
            doc['title'].lower() in response.lower() 
            for doc in context
        ) if context else True
        
        # 2. Is response substantive (not too short)?
        is_substantive = len(response.split()) > 50
        
        # 3. Does response directly address query?
        query_terms = set(query.lower().split())
        response_terms = set(response.lower().split())
        overlap = len(query_terms & response_terms) / len(query_terms)
        
        # Weighted score
        score = (
            0.4 * float(used_context) +
            0.3 * float(is_substantive) +
            0.3 * overlap
        )
        
        return min(1.0, score)
    
    def _update_persona_thresholds(self, quality_grade: float):
        """
        Update RLHF thresholds based on response quality.
        This is the adaptive learning mechanism.
        """
        
        # If grade < 0.5, we need more context
        if quality_grade < 0.5:
            self.persona_config['rlhf_thresholds']['retrieval_required'] += 0.05
        else:
            # Successful response, can relax threshold slightly
            self.persona_config['rlhf_thresholds']['retrieval_required'] -= 0.02
        
        # Clamp values
        self.persona_config['rlhf_thresholds']['retrieval_required'] = max(
            0.0, 
            min(1.0, self.persona_config['rlhf_thresholds']['retrieval_required'])
        )
        
        # Save updated config
        with open("data/persona.json", 'w') as f:
            json.dump(self.persona_config, f, indent=2)
```

**Key Insight**: The persona adapts over time. If vero-eval (which we'll integrate next) shows poor performance, these thresholds shift to require more evidence before responding.

## Part 5: Integrating vero-eval for Rigorous Testing

<iframe width="560" height="315" src="https://www.youtube.com/embed/6aoKDIrM3_E?si=ZqCuXXV9YgcfqL1a" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

This is where the magic happens. **vero-eval** provides production-grade evaluation that goes far beyond simple accuracy metrics. It tests edge cases, persona stress scenarios, and real-world failure modes.

![vero-eval Testing Framework for AI Research Assistant](/images/11152025/vero-eval-testing-framework.png)

### Installing and Configuring vero-eval

```bash
# Install vero-eval
pip install vero-eval

# Initialize evaluation directory
mkdir -p evaluation/datasets evaluation/results
```

### Generating a Research-Specific Test Dataset

vero-eval can generate test datasets tailored to your domain:

```python
# evaluation/generate_test_dataset.py

from vero.test_dataset_generator import generate_and_save
from pathlib import Path

def generate_research_test_dataset():
    """
    Generate challenging test queries for research assistant.
    vero-eval creates persona-based edge cases automatically.
    """
    
    # Point to your research papers directory
    data_path = Path('data/research_papers')
    
    # Define the use case
    use_case = """
    This is a research assistant that helps academics:
    - Find relevant papers on specific topics
    - Understand connections between research areas
    - Get summaries of complex papers
    - Discover citation networks
    - Answer technical questions about methodologies
    
    Edge cases to test:
    - Queries about very recent papers (after knowledge cutoff)
    - Multi-hop reasoning (papers that cite papers that discuss X)
    - Ambiguous author names
    - Requests for specific experimental results
    - Cross-domain queries (e.g., physics papers relevant to biology)
    """
    
    # Generate dataset with persona variations
    generate_and_save(
        data_path=str(data_path),
        usecase=use_case,
        save_path_dir='evaluation/datasets/research_assistant_v1',
        n_queries=150,  # Generate 150 test queries
        
        # Persona variations
        personas=[
            {
                'name': 'PhD Student',
                'characteristics': 'Detail-oriented, asks follow-up questions, wants methodology details'
            },
            {
                'name': 'Senior Researcher',
                'characteristics': 'Broad queries, interested in connections, asks about citations'
            },
            {
                'name': 'Industry Practitioner',
                'characteristics': 'Practical focus, wants applicable results, less theory'
            }
        ],
        
        # vero-eval will use Ollama for generation
        llm_provider='ollama',
        model_name='mistral'
    )
    
    print("✓ Generated test dataset with persona variations")
    print("  Check: evaluation/datasets/research_assistant_v1/")

if __name__ == "__main__":
    generate_research_test_dataset()
```

**Run this:**
```bash
python evaluation/generate_test_dataset.py
```

This creates a JSON file with queries like:
```json
{
  "query": "What papers discuss attention mechanisms in the context of graph neural networks published after 2022?",
  "persona": "Senior Researcher",
  "expected_characteristics": ["multi-hop", "temporal_constraint", "domain_crossing"],
  "ground_truth_chunk_ids": ["paper_47", "paper_89", "paper_102"],
  "complexity_score": 0.85
}
```

### Running the Evaluation Suite

Now we test our system against this dataset:

```python
# evaluation/run_evaluation.py

from vero.evaluator import Evaluator
from vero.metrics import (
    PrecisionMetric, RecallMetric, SufficiencyMetric,
    FaithfulnessMetric, BERTScoreMetric, RougeMetric,
    MRRMetric, MAPMetric, NDCGMetric
)
from reasoning_agent import PersonaReasoningAgent
import json

def run_full_evaluation():
    """
    Run comprehensive evaluation using vero-eval framework.
    Tests both retrieval and generation quality.
    """
    
    # Initialize our system
    agent = PersonaReasoningAgent()
    
    # Load test dataset
    with open('evaluation/datasets/research_assistant_v1/queries.json') as f:
        test_queries = json.load(f)
    
    # Initialize vero-eval
    evaluator = Evaluator(
        test_dataset=test_queries,
        trace_db_path='evaluation/trace.db'  # Logs all queries
    )
    
    # Define evaluation metrics
    retrieval_metrics = [
        PrecisionMetric(k=5),
        RecallMetric(k=5),
        SufficiencyMetric(),  # Are retrieved docs sufficient to answer?
    ]
    
    generation_metrics = [
        FaithfulnessMetric(),  # Is response faithful to retrieved docs?
        BERTScoreMetric(),     # Semantic similarity to reference answers
        RougeMetric()          # Token overlap with references
    ]
    
    ranking_metrics = [
        MRRMetric(),  # Mean Reciprocal Rank
        MAPMetric(),  # Mean Average Precision
        NDCGMetric()  # Normalized Discounted Cumulative Gain
    ]
    
    results = {
        'retrieval': {},
        'generation': {},
        'ranking': {},
        'per_persona': {}
    }
    
    # Run evaluation for each query
    for query_data in test_queries:
        query = query_data['query']
        persona = query_data['persona']
        ground_truth = query_data['ground_truth_chunk_ids']
        
        # Generate response using our system
        response_data = agent.generate_response(query)
        
        # Extract retrieved document IDs
        retrieved_ids = [
            doc.get('paper_id', doc['title']) 
            for doc in response_data['context_used']
        ]
        
        # Log to vero-eval's trace database
        evaluator.log_query(
            query=query,
            retrieved_docs=retrieved_ids,
            generated_response=response_data['response'],
            metadata={'persona': persona}
        )
        
        # Evaluate retrieval
        for metric in retrieval_metrics:
            score = metric.compute(
                retrieved=retrieved_ids,
                relevant=ground_truth
            )
            
            metric_name = metric.__class__.__name__
            if metric_name not in results['retrieval']:
                results['retrieval'][metric_name] = []
            results['retrieval'][metric_name].append(score)
        
        # Evaluate generation
        for metric in generation_metrics:
            score = metric.compute(
                generated=response_data['response'],
                reference=query_data.get('reference_answer', ''),
                context=response_data['context_used']
            )
            
            metric_name = metric.__class__.__name__
            if metric_name not in results['generation']:
                results['generation'][metric_name] = []
            results['generation'][metric_name].append(score)
        
        # Track per-persona performance
        if persona not in results['per_persona']:
            results['per_persona'][persona] = {
                'precision': [],
                'faithfulness': []
            }
        
        results['per_persona'][persona]['precision'].append(
            results['retrieval']['PrecisionMetric'][-1]
        )
        results['per_persona'][persona]['faithfulness'].append(
            results['generation']['FaithfulnessMetric'][-1]
        )
    
    # Aggregate results
    for category in ['retrieval', 'generation']:
        for metric_name, scores in results[category].items():
            results[category][metric_name] = {
                'mean': sum(scores) / len(scores),
                'min': min(scores),
                'max': max(scores),
                'std': np.std(scores)
            }
    
    # Save results
    with open('evaluation/results/full_evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✓ Evaluation complete!")
    print(f"  Retrieval Precision@5: {results['retrieval']['PrecisionMetric']['mean']:.3f}")
    print(f"  Retrieval Recall@5: {results['retrieval']['RecallMetric']['mean']:.3f}")
    print(f"  Generation Faithfulness: {results['generation']['FaithfulnessMetric']['mean']:.3f}")
    
    return results

if __name__ == "__main__":
    results = run_full_evaluation()
```

**Run the evaluation:**
```bash
python evaluation/run_evaluation.py
```

### Generating Performance Reports

vero-eval includes a report generator:

```python
from vero.report import ReportGenerator

# Generate comprehensive HTML report
generator = ReportGenerator(
    trace_db_path='evaluation/trace.db',
    results_path='evaluation/results/full_evaluation.json'
)

generator.generate_report(
    output_path='evaluation/results/performance_report.html',
    include_sections=[
        'executive_summary',
        'retrieval_analysis',
        'generation_analysis',
        'persona_breakdown',
        'failure_cases',
        'recommendations'
    ]
)

print("✓ Report generated: evaluation/results/performance_report.html")
```

This creates an interactive HTML report showing:
- Overall metrics with confidence intervals
- Per-persona performance breakdown
- Failure case analysis (queries where system performed poorly)
- Recommendations for improvement

## Part 6: The RLHF Feedback Loop

Now we close the loop: use vero-eval results to update the persona's RLHF thresholds:

```python
# evaluation/update_persona_from_results.py

import json

def update_persona_thresholds(evaluation_results: dict):
    """
    Analyze vero-eval results and adjust persona thresholds.
    This is the core RLHF mechanism.
    """
    
    # Load current persona config
    with open('data/persona.json') as f:
        persona_config = json.load(f)
    
    # Analyze retrieval performance
    retrieval_recall = evaluation_results['retrieval']['RecallMetric']['mean']
    
    if retrieval_recall < 0.6:
        # Low recall → need to retrieve more documents
        persona_config['rlhf_thresholds']['retrieval_limit'] += 2
        persona_config['rlhf_thresholds']['retrieval_required'] += 0.1
        
        print("⚠️  Low recall detected. Increasing retrieval aggressiveness.")
    
    # Analyze generation faithfulness
    faithfulness = evaluation_results['generation']['FaithfulnessMetric']['mean']
    
    if faithfulness < 0.7:
        # Responses not faithful to sources → need stronger grounding
        persona_config['rlhf_thresholds']['minimum_context_overlap'] = 0.4
        persona_config['system_prompt_template'] += (
            "\n\nIMPORTANT: Always cite specific papers when making claims. "
            "Do not speculate beyond what the retrieved papers state."
        )
        
        print("⚠️  Low faithfulness detected. Strengthening citation requirements.")
    
    # Per-persona adjustments
    for persona_name, metrics in evaluation_results['per_persona'].items():
        avg_precision = sum(metrics['precision']) / len(metrics['precision'])
        
        if avg_precision < 0.5:
            print(f"⚠️  {persona_name} persona underperforming (Precision: {avg_precision:.2f})")
            
            # Could adjust persona-specific prompts here
            # For now, log for manual review
    
    # Save updated config
    with open('data/persona.json', 'w') as f:
        json.dump(persona_config, f, indent=2)
    
    print("✓ Persona thresholds updated based on evaluation results")

# Usage after evaluation
with open('evaluation/results/full_evaluation.json') as f:
    results = json.load(f)

update_persona_thresholds(results)
```

**The workflow becomes:**
1. Run system on test queries
2. vero-eval measures performance
3. Script analyzes metrics
4. Persona thresholds adjust automatically
5. Re-evaluate to confirm improvement

This is **reinforcement learning through human feedback** (RLHF) in action, but guided by rigorous automated evaluation rather than ad-hoc human ratings.

## Part 7: Integrating with the Frontend

Now we wire this into the Next.js chat interface. Update `src/app/api/chat/route.ts`:

```typescript
import { NextRequest } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'

export async function POST(request: NextRequest) {
  const { message, messages, graphRAG = true } = await request.json()
  
  if (!graphRAG) {
    // Regular chat without RAG
    return handleRegularChat(message, messages)
  }
  
  // Call our Python reasoning agent
  const agentPath = path.join(process.cwd(), 'scripts', 'reasoning_agent.py')
  
  const result = await new Promise<{response: string, context: any[]}>((resolve, reject) => {
    const pythonProcess = spawn('python3', [
      agentPath,
      'generate',
      JSON.stringify({ query: message, chat_history: messages })
    ])
    
    let stdout = ''
    let stderr = ''
    
    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString()
    })
    
    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString()
    })
    
    pythonProcess.on('close', (code) => {
      if (code === 0) {
        try {
          const result = JSON.parse(stdout)
          resolve(result)
        } catch (e) {
          reject(new Error(`Failed to parse response: ${e}`))
        }
      } else {
        reject(new Error(`Agent failed: ${stderr}`))
      }
    })
  })
  
  // Stream response back to client
  const stream = new ReadableStream({
    start(controller) {
      // Send response with context metadata
      const formatted = `${result.response}\n\n---\n**Sources:**\n${
        result.context.map((doc, i) => 
          `[${i+1}] ${doc.title} (${doc.year})`
        ).join('\n')
      }`
      
      controller.enqueue(new TextEncoder().encode(formatted))
      controller.close()
    }
  })
  
  return new Response(stream, {
    headers: {
      'Content-Type': 'text/plain; charset=utf-8',
    },
  })
}
```

Update the chat UI to show retrieval metadata:

```typescript
// In src/components/Chat.tsx

{message.role === 'assistant' && message.context && (
  <div className="mt-2 text-xs text-muted-foreground">
    <details>
      <summary className="cursor-pointer hover:text-foreground">
        📚 {message.context.length} sources retrieved
      </summary>
      <ul className="mt-2 space-y-1">
        {message.context.map((doc, i) => (
          <li key={i} className="flex items-center gap-2">
            <span className="font-mono">
              {doc.retrieval_method === 'vector_search' ? '🔍' : '🔗'}
            </span>
            <span>{doc.title}</span>
            <span className="text-muted-foreground">
              (relevance: {(doc.relevance_score * 100).toFixed(0)}%)
            </span>
          </li>
        ))}
      </ul>
    </details>
  </div>
)}
```

Now users can see which papers were retrieved and how (vector search vs. citation traversal).

## Part 8: Running the Complete System

### Setup Script

Create `setup.sh`:

```bash
#!/bin/bash

echo "🔬 Setting up Research Assistant GraphRAG System"

# 1. Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# 2. Start Neo4j
echo "🗄️  Starting Neo4j..."
docker-compose up -d neo4j

# Wait for Neo4j to be ready
echo "⏳ Waiting for Neo4j..."
until curl -s http://localhost:7474 > /dev/null; do
  sleep 2
done
echo "✓ Neo4j ready"

# 3. Start Ollama
echo "🤖 Checking Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "Please install Ollama from https://ollama.ai"
    exit 1
fi

ollama serve &
sleep 5

# Pull required models
ollama pull mistral
ollama pull nomic-embed-text

# 4. Initialize Neo4j graph schema
echo "📊 Initializing graph schema..."
python scripts/init_graph_schema.py

# 5. Ingest sample research papers
echo "📚 Ingesting sample papers..."
python scripts/ingest_research_data.py --directory data/sample_papers

# 6. Generate test dataset
echo "🧪 Generating evaluation dataset..."
python evaluation/generate_test_dataset.py

# 7. Run initial evaluation
echo "📈 Running initial evaluation..."
python evaluation/run_evaluation.py

# 8. Start Next.js frontend
echo "🌐 Starting frontend..."
npm install
npm run dev &

echo ""
echo "✅ Setup complete!"
echo ""
echo "🔗 Access points:"
echo "   Frontend: http://localhost:3000"
echo "   Neo4j Browser: http://localhost:7474"
echo "   Evaluation Reports: evaluation/results/"
echo ""
echo "📖 Next steps:"
echo "   1. Add your research papers to data/research_papers/"
echo "   2. Run: python scripts/ingest_research_data.py"
echo "   3. Chat with your research assistant at localhost:3000"
echo "   4. Check evaluation results in evaluation/results/"
```

Run it:
```bash
chmod +x setup.sh
./setup.sh
```

## Part 9: Practical Use Cases and Patterns

### Use Case 1: Literature Review Assistant

```python
# Example query patterns for literature reviews

queries = [
    "What are the main approaches to attention mechanisms in transformers since 2020?",
    "Find papers that cite Vaswani et al. 2017 and discuss efficiency improvements",
    "What experimental setups are common in graph neural network papers?",
    "Compare the methodologies used in top-cited RAG papers"
]

for query in queries:
    response = agent.generate_response(query)
    
    # System automatically:
    # 1. Retrieves relevant papers using hybrid search
    # 2. Traverses citation network
    # 3. Formats response with proper attributions
    # 4. Logs everything to vero-eval trace DB
```

### Use Case 2: Cross-Domain Research Discovery

```python
# Finding connections between domains

query = """
Are there any techniques from computer vision that have been 
successfully applied to natural language processing in the last 3 years?
"""

# The graph traversal will:
# 1. Find CV papers discussing specific techniques
# 2. Find NLP papers citing those CV papers
# 3. Identify the bridging concepts
# 4. Present a coherent narrative

response = agent.generate_response(query)
```

### Use Case 3: Methodology Extraction

```python
# Extracting specific methodological details

query = """
What evaluation metrics are most commonly used in papers about 
few-shot learning for NLP tasks?
"""

# Behind the scenes:
# 1. Retrieve few-shot NLP papers
# 2. Extract methodology sections (using LLM)
# 3. Aggregate metrics across papers
# 4. Present frequency analysis
```

## Part 10: Measuring Success with vero-eval

After running the system for a while, check the vero-eval dashboard:

```python
# evaluation/generate_dashboard.py

from vero.dashboard import create_dashboard
from vero.trace_db import TraceDB

# Load trace database
trace_db = TraceDB('evaluation/trace.db')

# Create interactive dashboard
create_dashboard(
    trace_db=trace_db,
    output_path='evaluation/dashboard.html',
    metrics=[
        'retrieval_precision',
        'retrieval_recall',
        'generation_faithfulness',
        'response_time',
        'context_sufficiency'
    ],
    groupby=['persona', 'query_complexity']
)
```

This generates an interactive Plotly dashboard showing:
- **Metric trends over time** (is the system improving?)
- **Persona performance comparison** (which user types are we serving well?)
- **Query complexity vs. accuracy** (where do we struggle?)
- **Retrieval method effectiveness** (vector vs. graph traversal success rates)

## Advanced Patterns and Optimizations

### Pattern 1: Caching Embeddings

For production, cache embeddings to avoid recomputation:

```python
import pickle
from pathlib import Path

class EmbeddingCache:
    def __init__(self, cache_dir: Path = Path('cache/embeddings')):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_embedding(self, text: str, model: str = 'nomic-embed-text') -> list[float]:
        # Create hash of text for cache key
        cache_key = hashlib.md5(text.encode()).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}_{model}.pkl"
        
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        # Generate new embedding
        embedding = ollama.embeddings(model=model, prompt=text)['embedding']
        
        # Cache it
        with open(cache_path, 'wb') as f:
            pickle.dump(embedding, f)
        
        return embedding
```

### Pattern 2: Batch Processing for Large Collections

When ingesting 1000+ papers:

```python
def ingest_batch(papers: list[Path], batch_size: int = 10):
    """Process papers in batches to manage memory"""
    
    for i in range(0, len(papers), batch_size):
        batch = papers[i:i+batch_size]
        
        # Extract metadata in parallel
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            metadata_list = executor.map(extract_paper_metadata, batch)
        
        # Insert into Neo4j in single transaction
        with driver.session() as session:
            with session.begin_transaction() as tx:
                for metadata, pdf_path in zip(metadata_list, batch):
                    create_paper_node(tx, metadata, pdf_path)
                
                tx.commit()
        
        print(f"✓ Processed {i+batch_size}/{len(papers)} papers")
```

### Pattern 3: Incremental Evaluation

Don't wait to run full evaluation. Track metrics continuously:

```python
class ContinuousEvaluator:
    def __init__(self, alert_threshold: float = 0.6):
        self.alert_threshold = alert_threshold
        self.recent_scores = []
        
    def evaluate_response(self, query: str, response: dict):
        # Quick evaluation on the fly
        score = self._quick_score(response)
        self.recent_scores.append(score)
        
        # Keep only last 50 queries
        if len(self.recent_scores) > 50:
            self.recent_scores.pop(0)
        
        # Alert if average drops
        if len(self.recent_scores) >= 10:
            avg = sum(self.recent_scores) / len(self.recent_scores)
            if avg < self.alert_threshold:
                self._send_alert(avg)
    
    def _quick_score(self, response: dict) -> float:
        # Lightweight scoring
        has_context = len(response['context_used']) > 0
        response_length = len(response['response'].split())
        
        return 0.7 * has_context + 0.3 * min(1.0, response_length / 100)
```

## Troubleshooting Common Issues

### Issue 1: Neo4j Connection Errors

```python
# Test Neo4j connection
from neo4j import GraphDatabase

def test_connection():
    try:
        driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "research2025")
        )
        
        with driver.session() as session:
            result = session.run("RETURN 1 AS num")
            print("✓ Neo4j connection successful")
            
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        print("  Make sure Neo4j is running: docker ps")
```

### Issue 2: Ollama Model Not Found

```bash
# Check available models
ollama list

# Pull missing models
ollama pull mistral
ollama pull nomic-embed-text

# Verify they work
ollama run mistral "Test query"
```

### Issue 3: Low Retrieval Scores

Check your embeddings:

```python
# Verify embeddings are being generated correctly
from ingest_research_data import ResearchGraphBuilder

builder = ResearchGraphBuilder()

# Test on a sample paper
test_text = "Transformers are a type of neural network architecture..."
embedding = builder.graph_rag.generate_embedding(test_text)

print(f"Embedding dimension: {len(embedding)}")  # Should be 4096
print(f"Sample values: {embedding[:5]}")
```

## Conclusion and Next Steps

You now have a production-ready Research Assistant with:

✅ **Local-first architecture** (no API costs, full privacy)  
✅ **Neo4j knowledge graph** (papers, authors, concepts, citations)  
✅ **Hybrid retrieval** (vector similarity + graph traversal)  
✅ **Persona-driven responses** with RLHF adaptation  
✅ **Comprehensive evaluation** via vero-eval framework  
✅ **Automated improvement** through feedback loops

### Recommended Next Steps:

1. **Expand the Dataset**: Ingest your actual research papers
   ```bash
   python scripts/ingest_research_data.py --directory ~/Documents/Research
   ```

2. **Run Weekly Evaluations**: Set up a cron job
   ```bash
   0 2 * * 0 cd /path/to/research-assistant && python evaluation/run_evaluation.py
   ```

3. **Fine-tune Personas**: Create persona configs for different user types:
   - PhD Student persona (detail-oriented, wants methodology)
   - Senior Researcher persona (big picture, cross-domain)
   - Industry persona (practical applications)

4. **Integrate Additional Sources**:
   - arXiv API for latest papers
   - Connected Papers for visualization
   - Semantic Scholar for citation data

5. **Scale Up**: 
   - Use a vector database (Pinecone, Weaviate) for 10K+ papers
   - Implement query result caching
   - Add paper summarization pipeline

### Resources for Going Deeper

- **Neo4j GenAI Integration**: [Official Documentation](https://neo4j.com/docs/cypher-manual/current/genai-integrations/)
- **llama.cpp**: [Mastering Local LLM Integration](https://danielkliewer.com/blog/2025-11-12-mastering-llama-cpp-local-llm-integration-guide)
- **vero-eval Framework**: [GitHub Repository](https://github.com/vero-labs-ai/vero-eval)

### Production Deployment Checklist

Before deploying to production, ensure you've addressed:

```python
# deployment/production_checklist.py

PRODUCTION_CHECKLIST = {
    'Infrastructure': [
        '☐ Neo4j running with persistent volumes',
        '☐ Ollama configured with appropriate model cache',
        '☐ Redis/Memcached for query result caching',
        '☐ Load balancer for API endpoints',
        '☐ CDN for static assets'
    ],
    'Security': [
        '☐ API authentication implemented',
        '☐ Rate limiting configured (per user/IP)',
        '☐ Input sanitization for all user queries',
        '☐ Neo4j credentials rotated and secured',
        '☐ HTTPS enabled with valid certificates'
    ],
    'Monitoring': [
        '☐ Prometheus metrics exported',
        '☐ Grafana dashboards for system health',
        '☐ vero-eval continuous evaluation running',
        '☐ Error tracking (Sentry/Rollbar)',
        '☐ Query latency monitoring'
    ],
    'Data Management': [
        '☐ Automated backups of Neo4j database',
        '☐ Embedding cache backup strategy',
        '☐ Data retention policies defined',
        '☐ GDPR compliance for user queries',
        '☐ Paper metadata update pipeline'
    ],
    'Performance': [
        '☐ Embedding generation batched/cached',
        '☐ Neo4j indexes optimized',
        '☐ Query result caching implemented',
        '☐ Connection pooling configured',
        '☐ Async processing for long-running queries'
    ]
}
```

## Part 11: Advanced vero-eval Techniques

Now let's dive deeper into what makes vero-eval exceptional for production AI systems.

### Stress Testing with Adversarial Queries

vero-eval can generate adversarial test cases that expose edge cases:

```python
# evaluation/adversarial_testing.py

from vero.adversarial import AdversarialGenerator
from reasoning_agent import PersonaReasoningAgent

def run_adversarial_tests():
    """
    Generate adversarial queries designed to break the system.
    This reveals weaknesses before users find them.
    """
    
    agent = PersonaReasoningAgent()
    
    # Initialize adversarial generator
    adv_gen = AdversarialGenerator(
        base_queries=load_valid_queries(),
        attack_types=[
            'jailbreak',          # Try to bypass safety guardrails
            'context_overflow',   # Queries requiring huge context
            'ambiguous_reference', # "the paper mentioned earlier" without context
            'temporal_confusion', # Mixing past/future tenses
            'multi_hop_complex',  # Require 3+ reasoning steps
            'contradictory',      # Ask for contradicting information
            'out_of_domain'       # Queries completely outside research
        ]
    )
    
    adversarial_queries = adv_gen.generate(n=50)
    
    failures = []
    
    for query_data in adversarial_queries:
        query = query_data['query']
        attack_type = query_data['attack_type']
        
        print(f"Testing: {attack_type} - {query[:60]}...")
        
        try:
            response = agent.generate_response(query)
            
            # Check for failure modes
            if len(response['response']) < 10:
                failures.append({
                    'query': query,
                    'attack_type': attack_type,
                    'failure_mode': 'empty_response'
                })
            
            elif 'hallucination' in detect_hallucinations(
                response['response'], 
                response['context_used']
            ):
                failures.append({
                    'query': query,
                    'attack_type': attack_type,
                    'failure_mode': 'hallucination'
                })
            
        except Exception as e:
            failures.append({
                'query': query,
                'attack_type': attack_type,
                'failure_mode': 'exception',
                'error': str(e)
            })
    
    # Generate failure report
    with open('evaluation/results/adversarial_failures.json', 'w') as f:
        json.dump(failures, f, indent=2)
    
    print(f"\n⚠️  Found {len(failures)} failure cases out of 50 adversarial queries")
    print(f"   Failure rate: {len(failures)/50*100:.1f}%")
    
    # Categorize failures
    failure_by_type = {}
    for failure in failures:
        attack_type = failure['attack_type']
        failure_by_type[attack_type] = failure_by_type.get(attack_type, 0) + 1
    
    print("\n📊 Failures by attack type:")
    for attack_type, count in sorted(failure_by_type.items(), 
                                     key=lambda x: x[1], 
                                     reverse=True):
        print(f"   {attack_type}: {count}")
    
    return failures

def detect_hallucinations(response: str, context_docs: list) -> list:
    """
    Detect potential hallucinations by checking if claims in response
    are supported by retrieved context.
    """
    
    hallucinations = []
    
    # Extract claims from response (sentences making factual statements)
    claims = extract_claims(response)
    
    # Create context text corpus
    context_text = "\n".join([doc['abstract'] for doc in context_docs])
    
    for claim in claims:
        # Check if claim is substantiated by context
        # Use simple token overlap for now (could use entailment model)
        claim_tokens = set(claim.lower().split())
        context_tokens = set(context_text.lower().split())
        
        overlap = len(claim_tokens & context_tokens) / len(claim_tokens)
        
        if overlap < 0.3:  # Less than 30% overlap suggests hallucination
            hallucinations.append({
                'claim': claim,
                'overlap_score': overlap,
                'severity': 'high' if overlap < 0.1 else 'medium'
            })
    
    return hallucinations

def extract_claims(response: str) -> list[str]:
    """Extract factual claims from response."""
    # Simple heuristic: sentences with "is", "are", "shows", "demonstrates"
    sentences = response.split('.')
    
    claim_indicators = ['is', 'are', 'shows', 'demonstrates', 'found', 'reports']
    
    claims = [
        sent.strip() for sent in sentences
        if any(indicator in sent.lower() for indicator in claim_indicators)
        and len(sent.split()) > 5  # Substantial claim
    ]
    
    return claims

if __name__ == "__main__":
    failures = run_adversarial_tests()
```

**Run this regularly:**
```bash
# Weekly adversarial testing
0 3 * * 1 cd /path/to/research-assistant && python evaluation/adversarial_testing.py
```

### Continuous Monitoring with vero-eval

Set up real-time quality monitoring:

```python
# evaluation/continuous_monitor.py

from vero.monitor import QualityMonitor
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText

class ProductionMonitor:
    def __init__(self, trace_db_path: str):
        self.monitor = QualityMonitor(trace_db_path)
        self.alert_thresholds = {
            'precision_drop': 0.15,      # Alert if precision drops by 15%
            'latency_spike': 2.0,        # Alert if latency > 2 seconds
            'error_rate': 0.05,          # Alert if error rate > 5%
            'faithfulness_drop': 0.20    # Alert if faithfulness drops by 20%
        }
        
    def check_system_health(self):
        """
        Run every hour to check if system performance is degrading.
        """
        
        # Get metrics for last 24 hours
        recent_metrics = self.monitor.get_metrics(
            start_time=datetime.now() - timedelta(hours=24),
            end_time=datetime.now()
        )
        
        # Get baseline metrics (last week average)
        baseline_metrics = self.monitor.get_metrics(
            start_time=datetime.now() - timedelta(days=7),
            end_time=datetime.now() - timedelta(days=1)
        )
        
        alerts = []
        
        # Check for precision drop
        precision_drop = (
            baseline_metrics['precision'] - recent_metrics['precision']
        )
        if precision_drop > self.alert_thresholds['precision_drop']:
            alerts.append({
                'severity': 'high',
                'metric': 'precision',
                'message': f"Precision dropped by {precision_drop:.2%}",
                'baseline': baseline_metrics['precision'],
                'current': recent_metrics['precision']
            })
        
        # Check for latency spikes
        if recent_metrics['avg_latency'] > self.alert_thresholds['latency_spike']:
            alerts.append({
                'severity': 'medium',
                'metric': 'latency',
                'message': f"Average latency: {recent_metrics['avg_latency']:.2f}s",
                'baseline': baseline_metrics['avg_latency'],
                'current': recent_metrics['avg_latency']
            })
        
        # Check error rate
        if recent_metrics['error_rate'] > self.alert_thresholds['error_rate']:
            alerts.append({
                'severity': 'critical',
                'metric': 'error_rate',
                'message': f"Error rate: {recent_metrics['error_rate']:.2%}",
                'baseline': baseline_metrics['error_rate'],
                'current': recent_metrics['error_rate']
            })
        
        # Check faithfulness
        faithfulness_drop = (
            baseline_metrics['faithfulness'] - recent_metrics['faithfulness']
        )
        if faithfulness_drop > self.alert_thresholds['faithfulness_drop']:
            alerts.append({
                'severity': 'high',
                'metric': 'faithfulness',
                'message': f"Faithfulness dropped by {faithfulness_drop:.2%}",
                'baseline': baseline_metrics['faithfulness'],
                'current': recent_metrics['faithfulness']
            })
        
        # Send alerts if any
        if alerts:
            self.send_alerts(alerts)
        
        # Log to monitoring system
        self.log_health_check(recent_metrics, alerts)
        
        return alerts
    
    def send_alerts(self, alerts: list):
        """Send alerts via email/Slack/PagerDuty"""
        
        critical_alerts = [a for a in alerts if a['severity'] == 'critical']
        
        if critical_alerts:
            # Page on-call engineer
            self.page_oncall(critical_alerts)
        
        # Email summary
        email_body = self.format_alert_email(alerts)
        self.send_email(
            to='team@example.com',
            subject=f"🚨 Research Assistant Quality Alert - {len(alerts)} issues",
            body=email_body
        )
    
    def format_alert_email(self, alerts: list) -> str:
        """Format alerts as HTML email"""
        
        html = """
        <h2>Research Assistant Quality Alerts</h2>
        <p>The following performance degradations were detected:</p>
        <table border="1" cellpadding="10">
            <tr>
                <th>Severity</th>
                <th>Metric</th>
                <th>Baseline</th>
                <th>Current</th>
                <th>Message</th>
            </tr>
        """
        
        for alert in alerts:
            severity_color = {
                'critical': '#ff0000',
                'high': '#ff6600',
                'medium': '#ffaa00'
            }[alert['severity']]
            
            html += f"""
            <tr>
                <td style="background-color: {severity_color}; color: white;">
                    {alert['severity'].upper()}
                </td>
                <td>{alert['metric']}</td>
                <td>{alert['baseline']:.3f}</td>
                <td>{alert['current']:.3f}</td>
                <td>{alert['message']}</td>
            </tr>
            """
        
        html += """
        </table>
        <p>
        <a href="http://your-monitoring-url/dashboard">View Full Dashboard</a>
        </p>
        """
        
        return html
    
    def log_health_check(self, metrics: dict, alerts: list):
        """Log to your monitoring system (Prometheus/Datadog/etc)"""
        
        # Example: Push to Prometheus Pushgateway
        # In production, you'd use actual client library
        
        print(f"[{datetime.now()}] Health Check:")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  Faithfulness: {metrics['faithfulness']:.3f}")
        print(f"  Avg Latency: {metrics['avg_latency']:.2f}s")
        print(f"  Error Rate: {metrics['error_rate']:.2%}")
        
        if alerts:
            print(f"  ⚠️  {len(alerts)} alerts triggered")
        else:
            print(f"  ✓ All metrics within normal range")

# Run as scheduled job
if __name__ == "__main__":
    monitor = ProductionMonitor('evaluation/trace.db')
    alerts = monitor.check_system_health()
    
    if alerts:
        exit(1)  # Non-zero exit code for alerting systems
```

**Set up as cron job:**
```bash
# Check every hour
0 * * * * cd /path/to/research-assistant && python evaluation/continuous_monitor.py
```

## Part 12: Scaling Beyond 10K Papers

As your research collection grows, you'll need to optimize:

### 1. Migrate to a Dedicated Vector Database

For 10K+ papers, Neo4j's vector indexes can become slow. Use a specialized vector DB:

```python
# scripts/migrate_to_pinecone.py

import pinecone
from neo4j import GraphDatabase
import os

def migrate_embeddings_to_pinecone():
    """
    Migrate embeddings from Neo4j to Pinecone for faster retrieval.
    Keep Neo4j for graph relationships, Pinecone for vector search.
    """
    
    # Initialize Pinecone
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment="us-west1-gcp"
    )
    
    # Create index if doesn't exist
    if "research-papers" not in pinecone.list_indexes():
        pinecone.create_index(
            name="research-papers",
            dimension=4096,  # nomic-embed-text
            metric="cosine",
            pods=2,
            replicas=1,
            pod_type="p1.x1"
        )
    
    index = pinecone.Index("research-papers")
    
    # Extract embeddings from Neo4j
    driver = GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "research2025")
    )
    
    with driver.session() as session:
        # Get papers in batches
        batch_size = 100
        offset = 0
        
        while True:
            papers = session.run("""
                MATCH (p:Paper)
                RETURN p.title AS title,
                       p.abstract AS abstract,
                       p.abstract_embedding AS embedding,
                       p.year AS year,
                       ID(p) AS neo4j_id
                ORDER BY p.year DESC
                SKIP $offset
                LIMIT $batch_size
                """,
                offset=offset,
                batch_size=batch_size
            ).data()
            
            if not papers:
                break
            
            # Prepare vectors for Pinecone
            vectors = []
            for paper in papers:
                vectors.append({
                    'id': str(paper['neo4j_id']),
                    'values': paper['embedding'],
                    'metadata': {
                        'title': paper['title'],
                        'abstract': paper['abstract'][:500],  # Truncate
                        'year': paper['year'],
                        'neo4j_id': paper['neo4j_id']
                    }
                })
            
            # Upsert to Pinecone
            index.upsert(vectors=vectors, namespace="papers")
            
            print(f"✓ Migrated {offset + len(papers)} papers")
            offset += batch_size
    
    print(f"\n✅ Migration complete! {offset} papers in Pinecone")

# Update retriever to use Pinecone
class HybridRetrieverWithPinecone:
    def __init__(self, neo4j_driver, pinecone_index_name="research-papers"):
        self.neo4j_driver = neo4j_driver
        self.pinecone_index = pinecone.Index(pinecone_index_name)
    
    def retrieve_context(self, query: str, limit: int = 5) -> list[dict]:
        """Hybrid retrieval using Pinecone + Neo4j graph"""
        
        # 1. Vector search with Pinecone (fast!)
        query_embedding = ollama.embeddings(
            model='nomic-embed-text',
            prompt=query
        )['embedding']
        
        pinecone_results = self.pinecone_index.query(
            vector=query_embedding,
            top_k=limit * 2,
            include_metadata=True,
            namespace="papers"
        )
        
        # 2. Get Neo4j IDs from Pinecone results
        neo4j_ids = [
            int(match['metadata']['neo4j_id']) 
            for match in pinecone_results['matches']
        ]
        
        # 3. Enrich with graph relationships from Neo4j
        with self.neo4j_driver.session() as session:
            enriched = session.run("""
                UNWIND $neo4j_ids AS paper_id
                MATCH (p:Paper) WHERE ID(p) = paper_id
                OPTIONAL MATCH (p)<-[:AUTHORED]-(a:Author)
                OPTIONAL MATCH (p)-[:DISCUSSES]->(c:Concept)
                OPTIONAL MATCH (p)-[:CITES]->(cited:Paper)
                
                RETURN 
                    p.title AS title,
                    p.abstract AS abstract,
                    p.year AS year,
                    collect(DISTINCT a.name) AS authors,
                    collect(DISTINCT c.name) AS concepts,
                    collect(DISTINCT cited.title) AS citations
                """,
                neo4j_ids=neo4j_ids
            ).data()
        
        # 4. Combine Pinecone scores with Neo4j metadata
        results = []
        for i, match in enumerate(pinecone_results['matches']):
            neo4j_data = enriched[i] if i < len(enriched) else {}
            
            results.append({
                'title': neo4j_data.get('title', match['metadata']['title']),
                'abstract': neo4j_data.get('abstract', match['metadata']['abstract']),
                'year': neo4j_data.get('year', match['metadata']['year']),
                'authors': neo4j_data.get('authors', []),
                'concepts': neo4j_data.get('concepts', []),
                'citations': neo4j_data.get('citations', []),
                'relevance_score': match['score'],
                'retrieval_method': 'pinecone_vector_search'
            })
        
        return results[:limit]
```

**Benefits of this architecture:**
- Pinecone handles 10M+ vectors easily
- Neo4j focuses on graph relationships (citations, authorship)
- Best of both worlds: fast vector search + rich graph traversal

### 2. Implement Query Result Caching

```python
# lib/query_cache.py

import redis
import hashlib
import json
from datetime import timedelta

class QueryCache:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)
        self.ttl = timedelta(hours=24)  # Cache for 24 hours
    
    def get_cached_response(self, query: str, persona_config: dict) -> dict | None:
        """
        Check if we have a cached response for this query+persona combination.
        """
        
        # Create cache key from query + persona config
        cache_key = self._create_cache_key(query, persona_config)
        
        cached = self.redis.get(cache_key)
        if cached:
            print(f"✓ Cache hit for query: {query[:50]}...")
            return json.loads(cached)
        
        return None
    
    def cache_response(self, query: str, persona_config: dict, response: dict):
        """Store response in cache"""
        
        cache_key = self._create_cache_key(query, persona_config)
        
        self.redis.setex(
            cache_key,
            self.ttl,
            json.dumps(response)
        )
    
    def _create_cache_key(self, query: str, persona_config: dict) -> str:
        """Create deterministic cache key"""
        
        # Include relevant persona config aspects
        persona_hash = hashlib.md5(
            json.dumps(persona_config, sort_keys=True).encode()
        ).hexdigest()
        
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        return f"query_cache:{query_hash}:{persona_hash}"
    
    def invalidate_cache(self):
        """Invalidate all cached queries (e.g., after persona update)"""
        
        keys = self.redis.keys("query_cache:*")
        if keys:
            self.redis.delete(*keys)
            print(f"✓ Invalidated {len(keys)} cached queries")

# Integrate into reasoning agent
class CachedReasoningAgent(PersonaReasoningAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = QueryCache()
    
    def generate_response(self, query: str, chat_history: list = None) -> dict:
        """Generate response with caching"""
        
        # Check cache first
        cached = self.cache.get_cached_response(query, self.persona_config)
        if cached:
            return cached
        
        # Generate fresh response
        response = super().generate_response(query, chat_history)
        
        # Cache if quality is good
        if response['quality_grade'] > 0.7:
            self.cache.cache_response(query, self.persona_config, response)
        
        return response
```

### 3. Batch Embedding Generation

When ingesting large collections:

```python
# scripts/batch_embedding_generator.py

from concurrent.futures import ThreadPoolExecutor
import ollama
import time

class BatchEmbeddingGenerator:
    def __init__(self, model: str = 'nomic-embed-text', max_workers: int = 4):
        self.model = model
        self.max_workers = max_workers
        self.rate_limit_delay = 0.1  # 100ms between requests
    
    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in parallel with rate limiting.
        """
        
        embeddings = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = []
            for i, text in enumerate(texts):
                future = executor.submit(self._generate_single, text, i)
                futures.append(future)
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
            
            # Collect results in order
            for future in futures:
                embedding, index = future.result()
                embeddings.append((index, embedding))
        
        # Sort by original index
        embeddings.sort(key=lambda x: x[0])
        
        return [emb for _, emb in embeddings]
    
    def _generate_single(self, text: str, index: int) -> tuple[list[float], int]:
        """Generate single embedding with retry logic"""
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = ollama.embeddings(
                    model=self.model,
                    prompt=text[:8192]  # Truncate to model limit
                )
                return response['embedding'], index
            
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                
                print(f"⚠️  Retry {attempt+1}/{max_retries} for text {index}: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff

# Use in ingestion pipeline
def ingest_large_collection(papers: list[Path]):
    """Efficiently ingest 1000+ papers"""
    
    generator = BatchEmbeddingGenerator(max_workers=8)
    
    # Process in batches of 50
    batch_size = 50
    
    for i in range(0, len(papers), batch_size):
        batch = papers[i:i+batch_size]
        
        print(f"Processing batch {i//batch_size + 1}/{len(papers)//batch_size + 1}")
        
        # Extract abstracts
        abstracts = []
        metadata_list = []
        for paper_path in batch:
            metadata = extract_paper_metadata(paper_path)
            abstracts.append(metadata['abstract'])
            metadata_list.append(metadata)
        
        # Generate embeddings in parallel
        embeddings = generator.generate_embeddings_batch(abstracts)
        
        # Insert into database
        with neo4j_driver.session() as session:
            for metadata, embedding in zip(metadata_list, embeddings):
                metadata['abstract_embedding'] = embedding
                create_paper_node(session, metadata)
        
        print(f"✓ Ingested batch {i//batch_size + 1}")
```

## Part 13: Real-World Production Case Study

Let's walk through a complete example from a hypothetical research lab:

### Scenario: Computational Biology Research Lab

**Requirements:**
- 5,000 existing papers in their collection
- Weekly updates with new publications
- 15 active researchers with different expertise levels
- Need to find cross-domain connections (CS ↔ Biology)
- High precision required (wrong papers waste researcher time)

**Implementation:**

```python
# config/bio_lab_config.py

RESEARCH_LAB_CONFIG = {
    'name': 'Computational Biology Lab',
    'paper_sources': [
        'local_collection',  # Existing 5K papers
        'pubmed_api',        # Weekly updates
        'biorxiv_api',      # Preprints
        'arxiv_bio'         # CS bio papers
    ],
    'personas': {
        'wet_lab_biologist': {
            'description': 'Bench scientists with limited CS background',
            'rlhf_thresholds': {
                'technical_detail': 0.3,  # Less technical jargon
                'methodology_depth': 0.8,  # High experimental detail
                'formality': 0.5
            },
            'preferred_sources': ['Nature', 'Cell', 'Science']
        },
        'computational_biologist': {
            'description': 'Hybrid CS/Bio expertise',
            'rlhf_thresholds': {
                'technical_detail': 0.8,  # Can handle complexity
                'methodology_depth': 0.9,  # Wants algorithm details
                'formality': 0.7
            },
            'preferred_sources': ['Nature Methods', 'Bioinformatics', 'PLOS Comp Bio']
        },
        'pi_researcher': {
            'description': 'Principal investigator, needs big picture',
            'rlhf_thresholds': {
                'technical_detail': 0.5,  # Balanced
                'methodology_depth': 0.4,  # Focus on conclusions
                'formality': 0.9           # Very formal
            },
            'preferred_sources': ['High-impact journals', 'Review articles']
        }
    },
    'quality_requirements': {
        'min_precision': 0.85,  # Must retrieve >85% relevant papers
        'min_faithfulness': 0.90,  # Responses must be 90% faithful to sources
        'max_latency': 3.0  # 3 second max response time
    }
}
```

**Setup Script:**

```bash
#!/bin/bash
# setup_bio_lab.sh

echo "🧬 Setting up Computational Biology Research Assistant"

# 1. Ingest existing collection
echo "📚 Ingesting 5,000 existing papers..."
python scripts/ingest_research_data.py \
    --directory /data/lab_papers \
    --batch-size 50 \
    --parallel-workers 8

# 2. Set up automated paper updates
echo "📰 Configuring automated updates..."
python scripts/setup_paper_updates.py \
    --sources pubmed,biorxiv,arxiv \
    --schedule daily \
    --filter "computational biology OR bioinformatics"

# 3. Generate persona-specific test datasets
echo "🧪 Generating evaluation datasets..."
python evaluation/generate_test_dataset.py \
    --personas wet_lab,computational,pi \
    --queries-per-persona 50

# 4. Run initial evaluation
echo "📊 Running baseline evaluation..."
python evaluation/run_evaluation.py \
    --config config/bio_lab_config.py

# 5. Deploy to production
echo "🚀 Deploying to production..."
docker-compose -f docker-compose.bio-lab.yml up -d

echo "✅ Setup complete!"
echo "   Dashboard: http://lab-research-assistant.local"
echo "   Monitoring: http://lab-research-assistant.local/metrics"
```

**Weekly Evaluation Report Email:**

```python
# scripts/weekly_report.py

from vero.report import ReportGenerator
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import matplotlib.pyplot as plt

def generate_weekly_report():
    """
    Automated weekly report sent to PI and lab members.
    """
    
    # Generate vero-eval report
    generator = ReportGenerator(
        trace_db_path='evaluation/trace.db',
        results_path='evaluation/results/weekly.json'
    )
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Precision trends by persona
    axes[0, 0].plot(
        weekly_data['wet_lab_precision'],
        label='Wet Lab',
        marker='o'
    )
    axes[0, 0].plot(
        weekly_data['computational_precision'],
        label='Computational',
        marker='s'
    )
    axes[0, 0].plot(
        weekly_data['pi_precision'],
        label='PI',
        marker='^'
    )
    axes[0, 0].set_title('Retrieval Precision by Persona')
    axes[0, 0].set_xlabel('Week')
    axes[0, 0].set_ylabel('Precision@5')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Faithfulness over time
    axes[0, 1].plot(
        weekly_data['faithfulness'],
        color='green',
        marker='o'
    )
    axes[0, 1].axhline(y=0.90, color='r', linestyle='--', 
                       label='Target (90%)')
    axes[0, 1].set_title('Response Faithfulness')
    axes[0, 1].set_xlabel('Week')
    axes[0, 1].set_ylabel('Faithfulness Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Query latency distribution
    axes[1, 0].hist(
        weekly_data['latencies'],
        bins=30,
        edgecolor='black'
    )
    axes[1, 0].axvline(x=3.0, color='r', linestyle='--',
                       label='Max Latency (3s)')
    axes[1, 0].set_title('Query Latency Distribution')
    axes[1, 0].set_xlabel('Latency (seconds)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # 4. Top failure categories
    failure_categories = weekly_data['failure_categories']
    axes[1, 1].barh(
        list(failure_categories.keys()),
        list(failure_categories.values())
    )
    axes[1, 1].set_title('Top Failure Categories')
    axes[1, 1].set_xlabel('Count')
    
    plt.tight_layout()
    plt.savefig('evaluation/results/weekly_report.png', dpi=150)
    
    # Create email
    msg = MIMEMultipart()
    msg['Subject'] = f'Research Assistant Weekly Report - Week {week_number}'
    msg['From'] = 'research-assistant@lab.edu'
    msg['To'] = 'pi@lab.edu, lab-members@lab.edu'
    
    # Email body
    html_body = f"""
    <html>
    <body>
    <h2>Research Assistant Performance Report</h2>
    <h3>Week {week_number} - {date_range}</h3>
    
    <h4>📊 Key Metrics</h4>
    <table border="1" cellpadding="10">
        <tr>
            <th>Metric</th>
            <th>This Week</th>
            <th>Last Week</th>
            <th>Change</th>
        </tr>
        <tr>
            <td>Avg Precision@5</td>
            <td>{current_precision:.2%}</td>
            <td>{last_precision:.2%}</td>
            <td style="color: {'green' if change > 0 else 'red'};">
                {change:+.2%}
            </td>
        </tr>
        <tr>
            <td>Faithfulness</td>
            <td>{current_faithfulness:.2%}</td>
            <td>{last_faithfulness:.2%}</td>
            <td style="color: {'green' if faith_change > 0 else 'red'};">
                {faith_change:+.2%}
            </td>
        </tr>
        <tr>
            <td>Avg Latency</td>
            <td>{current_latency:.2f}s</td>
            <td>{last_latency:.2f}s</td>
            <td style="color: {'green' if latency_change < 0 else 'red'};">
                {latency_change:+.2f}s
            </td>
        </tr>
        <tr>
            <td>Queries Served</td>
            <td>{current_queries}</td>
            <td>{last_queries}</td>
            <td>{queries_change:+d}</td>
        </tr>
    </table>
    
    <h4>🎯 Performance by Persona</h4>
    <ul>
        <li><strong>Wet Lab Biologists:</strong> 
            Precision: {wet_lab_precision:.2%} 
            (Target: >85% ✓)
        </li>
        <li><strong>Computational Biologists:</strong> 
            Precision: {comp_bio_precision:.2%}
            (Target: >85% ✓)
        </li>
        <li><strong>PI Queries:</strong> 
            Precision: {pi_precision:.2%}
            (Target: >85% ⚠️ Below target)
        </li>
    </ul>
    
    <h4>⚠️ Issues & Recommendations</h4>
    <ul>
        <li>{issue_1}</li>
        <li>{issue_2}</li>
    </ul>
    
    <p>See attached visualization for detailed trends.</p>
    
    <p>
    <a href="http://lab-research-assistant.local/dashboard">
        View Interactive Dashboard
    </a>
    </p>
    
    </body>
    </html>
    """
    
    msg.attach(MIMEText(html_body, 'html'))
    
    # Attach visualization
    with open('evaluation/results/weekly_report.png', 'rb') as f:
        img = MIMEImage(f.read())
        img.add_header('Content-Disposition', 'attachment', 
                      filename='weekly_trends.png')
        msg.attach(img)
    
    # Send email
    with smtplib.SMTP('smtp.lab.edu', 587) as smtp:
        smtp.starttls()
        smtp.login('research-assistant@lab.edu', os.getenv('EMAIL_PASSWORD'))
        smtp.send_message(msg)
    
    print("✓ Weekly report sent to lab members")

if __name__ == "__main__":
    generate_weekly_report()
```

![Ollama Local LLM Integration Setup](/images/11152025/ollama-local-llm-integration.png)


## Conclusion: The Complete Picture

You now have everything needed to build, evaluate, and deploy a production-ready Research Assistant:

**Core Architecture:**
✅ Neo4j knowledge graph for research papers  
✅ Ollama for local LLM inference  
✅ Hybrid retrieval (vector + graph)  
✅ Persona-driven responses with RLHF  

**Evaluation & Quality:**
✅ vero-eval for rigorous testing  
✅ Automated adversarial testing  
✅ Continuous monitoring with alerts  
✅ Weekly performance reports  

**Production Features:**
✅ Caching for performance  
✅ Batch processing for scale  
✅ Automated paper updates  
✅ Multi-persona support  

**The vero-eval Advantage:**

What makes this system production-ready is the evaluation framework. Unlike traditional RAG systems that rely on gut feeling and spot-checking, we have:

1. **Systematic edge case testing** - adversarial queries expose weaknesses
2. **Persona stress testing** - ensures all user types are served well
3. **Automated regression detection** - alerts when quality degrades
4. **Actionable metrics** - precision/recall/faithfulness directly inform improvements
5. **Continuous learning** - RLHF loop closes based on real performance data

This is the difference between a demo and a system you'd trust with real research workflows.

**Next Steps:**
1. Clone the starter repo and follow the setup script
2. Ingest your first 100 papers to test the pipeline
3. Run vero-eval to establish your baseline
4. Iterate on retrieval and persona prompts
5. Deploy to staging and gather feedback
6. Use weekly reports to drive improvements

**Remember:** The goal isn't perfect accuracy on day one. It's building a system that measurably improves over time through evaluation-driven iteration.

Now go build something that makes research more efficient! 🚀

---

**Resources:**
- [Complete Code Repository](https://github.com/kliewerdaniel/chrisbot)
- [vero-eval Documentation](https://github.com/vero-labs-ai/vero-eval)
- [Neo4j GenAI Integration Guide](https://neo4j.com/docs/cypher-manual/current/genai-integrations/)
- [llama.cpp Guide](https://danielkliewer.com/blog/2025-11-12-mastering-llama-cpp-local-llm-integration-guide)

Questions? Open an issue in the repo or reach out to the community.

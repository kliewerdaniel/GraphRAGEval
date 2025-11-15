"""
Generate vero-eval test datasets for research assistant evaluation
Creates persona-based test cases that stress-test the system
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import random
from datetime import datetime

def generate_research_test_dataset(
    sample_papers: List[Dict] = None,
    output_path: Path = Path("evaluation/datasets/research_assistant_v1.json"),
    n_queries: int = 50,
    personas: List[Dict] = None
) -> Dict[str, Any]:
    """
    Generate a comprehensive test dataset for the research assistant.

    Args:
        sample_papers: List of paper metadata dicts (title, authors, abstract, etc.)
        output_path: Where to save the generated dataset
        n_queries: Number of test queries to generate
        personas: Custom persona definitions

    Returns:
        Generated dataset as dictionary
    """

    # Default personas based on README
    if not personas:
        personas = [
            {
                'name': 'PhD Student',
                'characteristics': 'Detail-oriented, asks follow-up questions, wants methodology details, budget constraints'
            },
            {
                'name': 'Senior Researcher',
                'characteristics': 'Broad queries, interested in connections, asks about citations, publication-focused'
            },
            {
                'name': 'Industry Practitioner',
                'characteristics': 'Practical focus, wants applicable results, less theory, implementation details'
            },
            {
                'name': 'Professor',
                'characteristics': 'Teaching-focused, wants comprehensive overviews, student-friendly explanations'
            },
            {
                'name': 'Postdoc',
                'characteristics': 'Fast-paced, wants quick insights, methodological rigor, novelty-focused'
            }
        ]

    # Use sample papers or create synthetic ones
    if not sample_papers:
        sample_papers = get_or_create_sample_papers()

    # Generate query templates covering different research scenarios
    query_templates = [
        # Basic understanding
        "What is {concept} and how does it work?",
        "Explain the key principles behind {method}.",
        "What are the main approaches to {topic}?",

        # Comparative analysis
        "Compare and contrast {method1} with {method2}.",
        "What are the advantages and disadvantages of {approach}?",
        "How does {technique} differ from traditional methods?",

        # Application-focused
        "How can {method} be applied to real-world {domain} problems?",
        "What are the practical implications of {finding}?",
        "How would you implement {algorithm} in practice?",

        # Research questions
        "What evidence supports the effectiveness of {method}?",
        "What are the current limitations of {approach}?",
        "What future directions are proposed for {field}?",

        # Complex reasoning
        "How might {concept1} and {concept2} complement each other?",
        "What connections exist between {field1} and {field2}?",
        "How has research in {topic} evolved over the past decade?",

        # Specific to papers
        "According to {paper_title}, what are the key findings?",
        "What methodology was used in the study about {topic}?",
        "How do the results from {paper_title} compare to previous work?",

        # Edge cases
        "What are some non-obvious applications of {technique}?",
        "What happens if we combine {method1} and {method2} in unexpected ways?",
        "What are the most controversial findings in {field}?",

        # Follow-up questions
        "Can you explain that in simpler terms?",
        "What would happen if we increased {parameter}?",
        "How does this relate to what we discussed earlier?"
    ]

    # Technical concepts to draw from
    concepts = [
        # Machine Learning
        "attention mechanisms", "transformer architectures", "neural networks",
        "graph neural networks", "reinforcement learning", "transfer learning",

        # Research-specific
        "quantum encryption", "cryptographic protocols", "military security",
        "defense applications", "quantum computing", "quantum sensors",

        # General research
        "empirical validation", "statistical significance", "experimental design",
        "hypothesis testing", "peer review process", "reproducibility",

        # Domain-specific
        "computational biology", "bioinformatics", "machine learning in healthcare",
        "natural language processing", "computer vision", "robotics"
    ]

    domains = [
        "healthcare", "finance", "education", "defense", "science", "technology",
        "business", "government", "academia", "industry"
    ]

    methods = [
        "deep learning", "supervised learning", "unsupervised learning",
        "federated learning", "meta-learning", "few-shot learning",
        "quantum algorithms", "optimization techniques", "statistical modeling"
    ]

    # Generate queries
    generated_queries = []

    for i in range(n_queries):
        # Select random persona
        persona = random.choice(personas)

        # Generate query based on template
        template = random.choice(query_templates)

        # Fill template with appropriate content
        query_text = template

        # Replace placeholders
        if "{concept}" in template:
            query_text = query_text.replace("{concept}", random.choice(concepts))
        if "{topic}" in template:
            query_text = query_text.replace("{topic}", random.choice(concepts))
        if "{method}" in template:
            query_text = query_text.replace("{method}", random.choice(methods))
        if "{method1}" in template and "{method2}" in template:
            method1 = random.choice(methods)
            method2 = random.choice([m for m in methods if m != method1])
            query_text = query_text.replace("{method1}", method1)
            query_text = query_text.replace("{method2}", method2)
        if "{approach}" in template:
            query_text = query_text.replace("{approach}", random.choice(methods))
        if "{technique}" in template:
            query_text = query_text.replace("{technique}", random.choice(methods))
        if "{algorithm}" in template:
            query_text = query_text.replace("{algorithm}", random.choice(methods))
        if "{domain}" in template:
            query_text = query_text.replace("{domain}", random.choice(domains))
        if "{finding}" in template:
            query_text = query_text.replace("{finding}", random.choice(concepts))
        if "{concept1}" in template and "{concept2}" in template:
            concept1 = random.choice(concepts)
            concept2 = random.choice([c for c in concepts if c != concept1])
            query_text = query_text.replace("{concept1}", concept1)
            query_text = query_text.replace("{concept2}", concept2)
        if "{field1}" in template and "{field2}" in template:
            field1 = random.choice(concepts)
            field2 = random.choice([c for c in concepts if c != field1])
            query_text = query_text.replace("{field1}", field1)
            query_text = query_text.replace("{field2}", field2)
        if "{field}" in template:
            query_text = query_text.replace("{field}", random.choice(concepts))
        if "{paper_title}" in template:
            paper = random.choice(sample_papers)
            query_text = query_text.replace("{paper_title}", paper['title'])
        if "{parameter}" in template:
            query_text = query_text.replace("{parameter}", random.choice([
                "learning rate", "batch size", "model size", "training data",
                "temperature", "threshold", "number of layers"
            ]))

        # Determine expected ground truth papers
        relevant_papers = []
        for paper in sample_papers:
            # Simple relevance scoring based on keyword matching
            paper_text = f"{paper['title']} {paper.get('abstract', '')}".lower()
            query_lower = query_text.lower()
            overlap = len(set(query_lower.split()) & set(paper_text.split()))
            if overlap > 1 or any(concept in paper_text for concept in concepts):
                relevant_papers.append(paper['title'])

        # Limit to top matches
        ground_truth = relevant_papers[:3] if relevant_papers else ["unknown"]

        # Generate expected reference answer (synthetic, for evaluation)
        reference_answer = generate_reference_answer(query_text, relevant_papers, persona)

        # Calculate complexity score
        complexity_score = calculate_query_complexity(query_text, persona)

        # Add some edge case queries for stress testing
        expected_characteristics = []
        if "compare" in query_text.lower() or "contrast" in query_text.lower():
            expected_characteristics.append("comparative")
        if "how" in query_text.lower() or "why" in query_text.lower():
            expected_characteristics.append("explanatory")
        if len(query_text.split()) > 20:
            expected_characteristics.append("complex")

        query_data = {
            'query': query_text,
            'persona': persona['name'],
            'persona_description': persona['characteristics'],
            'expected_characteristics': expected_characteristics,
            'ground_truth_chunk_ids': ground_truth,
            'reference_answer': reference_answer,
            'complexity_score': complexity_score,
            'generated_at': datetime.now().isoformat(),
            'query_id': f"query_{i+1:03d}"
        }

        generated_queries.append(query_data)

    # Create dataset
    dataset = {
        'metadata': {
            'name': 'Research Assistant Test Dataset v1',
            'description': 'Comprehensive test dataset for GraphRAG research assistant evaluation',
            'created_at': datetime.now().isoformat(),
            'generated_queries': n_queries,
            'personas': len(personas),
            'evaluation_framework': 'vero-eval'
        },
        'personas': personas,
        'queries': generated_queries,
        'sample_papers': sample_papers[:10] if sample_papers else []  # Include some sample papers
    }

    # Save dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"✓ Generated test dataset with {n_queries} queries")
    print(f"  Saved to: {output_path}")
    print(f"  Personas: {len(personas)}")
    print(f"  Sample papers: {len(dataset['sample_papers'])}")

    return dataset

def get_or_create_sample_papers() -> List[Dict]:
    """Get sample papers or create synthetic ones for testing"""
    sample_papers = []

    # Check if we have real papers
    papers_dir = Path("data/research_papers")
    if papers_dir.exists():
        pdf_files = list(papers_dir.glob("*.pdf"))
        if pdf_files:
            # Use real paper data if available
            try:
                from scripts.ingest_research_data import ResearchGraphBuilder
                builder = ResearchGraphBuilder()

                for pdf_file in pdf_files[:5]:  # Limit to 5 for dataset
                    metadata = builder.extract_paper_metadata(pdf_file)
                    if metadata:
                        sample_papers.append({
                            'title': metadata.get('title', 'Unknown'),
                            'authors': metadata.get('authors', []),
                            'abstract': metadata.get('abstract', ''),
                            'year': metadata.get('year', 2024),
                            'concepts': metadata.get('concepts', [])
                        })
            except Exception as e:
                print(f"Could not extract real paper data: {e}")

    # If no real papers or extraction failed, create synthetic ones
    if not sample_papers:
        synthetic_papers = [
            {
                'title': 'Military and Security Dimensions of Quantum Technologies',
                'authors': ['Michal Krelina'],
                'abstract': 'This paper explores how quantum technologies are shaping military security, including quantum encryption, sensors, and computing applications.',
                'year': 2025,
                'concepts': ['quantum encryption', 'military security', 'defense applications']
            },
            {
                'title': 'Attention Mechanisms in Deep Learning',
                'authors': ['Vaswani et al.', 'Bahdanau et al.'],
                'abstract': 'Comprehensive overview of attention mechanisms and their role in modern neural architectures, particularly transformers.',
                'year': 2023,
                'concepts': ['attention mechanisms', 'transformers', 'deep learning']
            },
            {
                'title': 'Graph Neural Networks: A Survey',
                'authors': ['Wu et al.', 'Zhou et al.'],
                'abstract': 'Survey of graph neural network architectures, applications, and recent advances in the field.',
                'year': 2024,
                'concepts': ['graph neural networks', 'graph theory', 'representation learning']
            },
            {
                'title': 'Federated Learning for Privacy-Preserving AI',
                'authors': ['Yang et al.', 'Li et al.'],
                'abstract': 'Analysis of federated learning approaches for training machine learning models while preserving data privacy.',
                'year': 2023,
                'concepts': ['federated learning', 'privacy preservation', 'distributed computing']
            },
            {
                'title': 'Large Language Models in Scientific Discovery',
                'authors': ['Wang et al.', 'Jumper et al.'],
                'abstract': 'Applications of large language models in accelerating scientific research and hypothesis generation.',
                'year': 2024,
                'concepts': ['large language models', 'scientific discovery', 'AI in research']
            }
        ]
        sample_papers.extend(synthetic_papers)

    return sample_papers

def generate_reference_answer(query: str, relevant_papers: List[str], persona: Dict) -> str:
    """Generate a synthetic reference answer for evaluation"""
    # This would ideally use an LLM, but we'll create a template-based version
    base_answers = {
        'attention': "Attention mechanisms allow models to focus on relevant parts of input data. The key innovation in transformers was self-attention, which computes relationships between all input elements simultaneously.",
        'quantum': "Quantum technologies leverage quantum mechanics principles like superposition and entanglement. In security applications, quantum cryptography provides unconditional security.",
        'neural_networks': "Neural networks are computational models inspired by biological neural systems. They consist of layers of interconnected nodes that learn through backpropagation.",
        'federated': "Federated learning enables training models on decentralized data without sharing raw information, preserving privacy while building global models.",
        'graphs': "Graph neural networks operate on graph-structured data, learning representations of nodes and edges through message passing between neighboring nodes."
    }

    # Find relevant answer
    answer = "Based on current research, this topic involves complex interactions between multiple factors. Key findings include improved performance through advanced techniques, though practical implementations remain challenging."

    for key, ans in base_answers.items():
        if key in query.lower():
            answer = ans
            break

    # Add persona-specific aspects
    if persona['name'] == 'PhD Student':
        answer += " For further reading, consider exploring the original papers and implementing simple versions of these techniques."
    elif persona['name'] == 'Senior Researcher':
        answer += " Recent citations show increasing adoption in applied settings across various domains."
    elif persona['name'] == 'Industry Practitioner':
        answer += " Production implementations typically require careful optimization for scalability and robustness."

    return answer

def calculate_query_complexity(query: str, persona: Dict) -> float:
    """Calculate query complexity score (0-1)"""
    complexity = 0.5  # Base complexity

    # Length-based complexity
    if len(query.split()) > 15:
        complexity += 0.2
    elif len(query.split()) < 5:
        complexity -= 0.2

    # Technical terms
    technical_terms = ['algorithm', 'neural', 'quantum', 'cryptography', 'federated', 'transformer']
    technical_count = sum(1 for term in technical_terms if term in query.lower())
    complexity += min(technical_count * 0.1, 0.3)

    # Persona-based adjustments
    if persona['name'] == 'Professor':
        complexity -= 0.1  # Wants simpler explanations
    elif persona['name'] == 'Postdoc':
        complexity += 0.1  # Handles complex topics

    return max(0.0, min(1.0, complexity))

def generate_stress_test_queries(output_path: Path = Path("evaluation/datasets/stress_tests.json")):
    """Generate adversarial stress test queries"""
    stress_queries = [
        {
            'query': 'What are the implications of using quantum monkeys with banana-powered flux capacitors in zero-gravity environments?',
            'persona': 'Senior Researcher',
            'attack_type': 'nonsensical',
            'expected_response': 'non-sensical queries should be identified and handled gracefully'
        },
        {
            'query': 'Write a comprehensive review covering all papers published between 1900 and 2025 on machine learning, including every single algorithm, application, and implementation detail.',
            'persona': 'Professor',
            'attack_type': 'overly_broad',
            'expected_response': 'overly broad queries should be scoped appropriately'
        },
        {
            'query': 'According to the 1973 paper by Jones that never actually existed and the 1999 conference nobody attended, what quantum effects on cheese aging were observed?',
            'persona': 'PhD Student',
            'attack_type': 'hallucinated_sources',
            'expected_response': 'non-existent sources should be clearly identified'
        },
        {
            'query': 'How does the current work on finite-state automata from the 1950s relate to modern blockchain technology in cryptocurrency systems?',
            'persona': 'Industry Practitioner',
            'attack_type': 'temporal_connection',
            'expected_response': 'should identify valid historical connections'
        }
    ]

    with open(output_path, 'w') as f:
        json.dump({
            'metadata': {
                'name': 'Adversarial Stress Tests',
                'description': 'Queries designed to test system robustness and error handling'
            },
            'queries': stress_queries
        }, f, indent=2)

    print(f"✓ Generated {len(stress_queries)} stress test queries")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate vero-eval test datasets")
    parser.add_argument("--queries", type=int, default=50, help="Number of queries to generate")
    parser.add_argument("--output", type=str, default="evaluation/datasets/research_assistant_v1.json",
                       help="Output path")
    parser.add_argument("--include-stress-tests", action="store_true",
                       help="Also generate stress test queries")

    args = parser.parse_args()

    # Generate main dataset
    dataset = generate_research_test_dataset(
        output_path=Path(args.output),
        n_queries=args.queries
    )

    if args.include_stress_tests:
        generate_stress_test_queries()

    print("✓ Dataset generation complete!")
    print(f"  Main dataset: {len(dataset['queries'])} queries")
    print(f"  Output: {args.output}")

"""
vero-eval evaluation runner for research assistant
"""
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import pandas as pd

# Add parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.reasoning_agent import PersonaReasoningAgent
from evaluation.metrics import (
    PrecisionMetric, RecallMetric, SufficiencyMetric,
    FaithfulnessMetric, BERTScoreMetric, RougeMetric,
    MRRMetric, MAPMetric, NDCGMetric, HallucinationDetectionMetric
)

class Evaluator:
    def __init__(self, test_dataset_path: Path, trace_db_path: Path):
        self.test_dataset_path = test_dataset_path
        self.trace_db_path = trace_db_path
        self.trace_data = []

        # Load or create trace database
        self._load_trace_db()

    def _load_trace_db(self):
        """Load trace database or create if it doesn't exist"""
        if self.trace_db_path.exists():
            try:
                with open(self.trace_db_path, 'r') as f:
                    self.trace_data = json.load(f)
            except:
                self.trace_data = []
        else:
            self.trace_data = []

    def _save_trace_db(self):
        """Save trace database"""
        self.trace_db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.trace_db_path, 'w') as f:
            json.dump(self.trace_data, f, indent=2)

    def log_query(self, query: str, retrieved_docs: List[str],
                  generated_response: str, metadata: Dict[str, Any] = None):
        """Log a query execution for later analysis"""
        trace_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'retrieved_docs': retrieved_docs,
            'generated_response': generated_response,
            'metadata': metadata or {}
        }

        self.trace_data.append(trace_entry)
        self._save_trace_db()

    def run_evaluation(self, queries: List[Dict[str, Any]], output_path: Path) -> Dict[str, Any]:
        """
        Run comprehensive evaluation using vero-eval metrics
        """

        agent = PersonaReasoningAgent()

        results = {
            'retrieval': {},
            'generation': {},
            'ranking': {},
            'per_persona': {},
            'query_analyses': []
        }

        # Initialize metrics
        retrieval_metrics = [
            PrecisionMetric("precision", k=5),
            RecallMetric("recall", k=5),
            SufficiencyMetric(),
        ]

        generation_metrics = [
            FaithfulnessMetric(),
            BERTScoreMetric(),
            RougeMetric(),
            HallucinationDetectionMetric()
        ]

        ranking_metrics = [
            MRRMetric("mrr"),
            MAPMetric("map"),
            NDCGMetric()
        ]

        # Evaluate each query
        for i, query_data in enumerate(queries):
            query = query_data['query']
            persona = query_data.get('persona', 'default')
            ground_truth = query_data.get('ground_truth_chunk_ids', [])
            reference_answer = query_data.get('reference_answer', '')

            print(f"Evaluating query {i+1}/{len(queries)}: {query[:50]}...")

            # Generate response using agent
            try:
                response_data = agent.generate_response(query)
                retrieved_ids = [
                    doc.get('title', 'Unknown') for doc in response_data['context_used']
                ]
                generated_response = response_data['response']
            except Exception as e:
                print(f"Error generating response: {e}")
                retrieved_ids = []
                generated_response = "Error occurred during response generation."

            # Evaluate retrieval
            for metric in retrieval_metrics:
                score = metric.compute(
                    retrieved=retrieved_ids,
                    relevant=ground_truth,
                    query=query,
                    generated=generated_response
                )

                metric_name = metric.__class__.__name__
                if metric_name not in results['retrieval']:
                    results['retrieval'][metric_name] = []
                results['retrieval'][metric_name].append(score)

            # Evaluate generation
            for metric in generation_metrics:
                score = metric.compute(
                    retrieved=[doc.get('abstract', '') for doc in response_data['context_used']],
                    generated=generated_response,
                    reference=reference_answer,
                    context=response_data['context_used']
                )

                metric_name = metric.__class__.__name__
                if metric_name not in results['generation']:
                    results['generation'][metric_name] = []
                results['generation'][metric_name].append(score)

            # Evaluate ranking
            for metric in ranking_metrics:
                score = metric.compute(
                    retrieved=retrieved_ids,
                    relevant=ground_truth
                )

                metric_name = metric.__class__.__name__
                if metric_name not in results['ranking']:
                    results['ranking'][metric_name] = []
                results['ranking'][metric_name].append(score)

            # Track per-persona performance
            if persona not in results['per_persona']:
                results['per_persona'][persona] = {
                    'precision': [],
                    'faithfulness': [],
                    'count': 0
                }

            results['per_persona'][persona]['precision'].append(
                results['retrieval']['PrecisionMetric'][-1]
            )
            results['per_persona'][persona]['faithfulness'].append(
                results['generation']['FaithfulnessMetric'][-1]
            )
            results['per_persona'][persona]['count'] += 1

            # Log query analysis
            query_analysis = {
                'query': query,
                'persona': persona,
                'retrieved_count': len(retrieved_ids),
                'response_length': len(generated_response.split()),
                'quality_grade': response_data.get('quality_grade', 0.0)
            }
            results['query_analyses'].append(query_analysis)

            # Log to trace database
            self.log_query(
                query=query,
                retrieved_docs=retrieved_ids,
                generated_response=generated_response,
                metadata={
                    'persona': persona,
                    'ground_truth': ground_truth,
                    'quality_grade': response_data.get('quality_grade', 0.0)
                }
            )

        # Aggregate results
        self._aggregate_results(results)

        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"✓ Evaluation complete! Results saved to {output_path}")
        return results

    def _aggregate_results(self, results: Dict[str, Any]):
        """Compute aggregate statistics"""
        from statistics import mean, median, stdev

        for category in ['retrieval', 'generation', 'ranking']:
            if category in results:
                # Create a copy of metric names to avoid dictionary modification during iteration
                metric_names = list(results[category].keys())
                for metric_name in metric_names:
                    scores = results[category][metric_name]
                    if isinstance(scores, list) and scores:
                        try:
                            results[category][f"{metric_name}_summary"] = {
                                'mean': mean(scores),
                                'median': median(scores),
                                'std': stdev(scores) if len(scores) > 1 else 0.0,
                                'min': min(scores),
                                'max': max(scores),
                                'count': len(scores)
                            }
                        except:
                            results[category][f"{metric_name}_summary"] = {
                                'error': 'Could not compute statistics'
                            }

# Quick evaluation script
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run vero-eval evaluation")
    parser.add_argument("--dataset", type=str, default="evaluation/datasets/research_assistant_v1.json",
                       help="Path to test dataset")
    parser.add_argument("--output", type=str, default="evaluation/results/evaluation_output.json",
                       help="Output path for results")

    args = parser.parse_args()

    # Sample queries if no dataset exists
    if not Path(args.dataset).exists():
        print(f"Dataset {args.dataset} not found. Using sample queries...")
        queries = [
            {
                'query': 'What are the main approaches to attention mechanisms in deep learning?',
                'persona': 'researcher',
                'ground_truth_chunk_ids': ['attention paper 1', 'attention paper 2'],
                'reference_answer': 'The main approaches include...',
                'complexity_score': 0.7
            },
            {
                'query': 'How do transformer models handle long-range dependencies?',
                'persona': 'student',
                'ground_truth_chunk_ids': ['transformer paper 1'],
                'reference_answer': 'Transformer models use...',
                'complexity_score': 0.6
            }
        ]
    else:
        with open(args.dataset, 'r') as f:
            queries = json.load(f)

    evaluator = Evaluator(
        test_dataset_path=Path(args.dataset),
        trace_db_path=Path("evaluation/trace.db")
    )

    results = evaluator.run_evaluation(
        queries=queries,
        output_path=Path(args.output)
    )

    # Print summary
    if 'retrieval' in results and 'PrecisionMetric_summary' in results['retrieval']:
        precision = results['retrieval']['PrecisionMetric_summary']['mean']
        print(".3f")
    if 'generation' in results and 'FaithfulnessMetric_summary' in results['generation']:
        faithfulness = results['generation']['FaithfulnessMetric_summary']['mean']
        print(".3f")

    print(f"✓ Results saved to {args.output}")

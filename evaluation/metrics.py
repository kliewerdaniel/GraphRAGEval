"""
vero-eval metrics for RAG evaluation
Following the patterns from the guide for comprehensive evaluation
"""
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from rouge_score import rouge_scorer
from collections import Counter

class BaseMetric:
    """Base class for evaluation metrics"""
    def __init__(self, name: str, k: int = None):
        self.name = name
        self.k = k

    def compute(self, retrieved: List[str], relevant: List[str], **kwargs) -> float:
        """Compute metric score"""
        raise NotImplementedError

class PrecisionMetric(BaseMetric):
    """Precision@K: Fraction of retrieved documents that are relevant"""
    def compute(self, retrieved: List[str], relevant: List[str], **kwargs) -> float:
        if not retrieved:
            return 0.0

        k = self.k or len(retrieved)
        retrieved_k = retrieved[:k]

        relevant_set = set(relevant)
        relevant_retrieved = sum(1 for doc in retrieved_k if doc in relevant_set)

        return relevant_retrieved / len(retrieved_k) if retrieved_k else 0.0

class RecallMetric(BaseMetric):
    """Recall@K: Fraction of relevant documents that are retrieved"""
    def compute(self, retrieved: List[str], relevant: List[str], **kwargs) -> float:
        if not relevant:
            return 1.0 if not retrieved else 0.0

        k = self.k or len(retrieved)
        retrieved_k = retrieved[:k]

        retrieved_set = set(retrieved_k)
        relevant_retrieved = sum(1 for doc in relevant if doc in retrieved_set)

        return relevant_retrieved / len(relevant)

class SufficiencyMetric(BaseMetric):
    """Assess if retrieved documents are sufficient to answer the query"""
    def __init__(self, name: str = "sufficiency", model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(name)
        self.encoder = SentenceTransformer(model_name)

    def compute(self, retrieved: List[str], relevant: List[str],
                query: str = "", generated: str = "", **kwargs) -> float:
        if not retrieved:
            return 0.0

        # Semantic similarity between query and retrieved content
        query_embedding = self.encoder.encode([query])[0]
        doc_texts = retrieved
        doc_embeddings = self.encoder.encode(doc_texts)

        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

        # Check if any document has high enough similarity to query
        max_similarity = np.max(similarities)

        # Simple heuristic: if max similarity > 0.3, consider it sufficient
        return 1.0 if max_similarity > 0.3 else 0.0

class FaithfulnessMetric(BaseMetric):
    """Faithfulness: Generated answer grounded in retrieved documents"""
    def __init__(self, name: str = "faithfulness", model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(name)
        self.encoder = SentenceTransformer(model_name)

    def compute(self, retrieved: List[str], relevant: List[str] = None,
                generated: str = "", context: List[Dict] = None, **kwargs) -> float:
        if not retrieved or not generated:
            return 0.0

        # Check if generated claims are supported by retrieved documents
        generated_embedding = self.encoder.encode([generated])[0]

        # Combine all retrieved document text
        combined_docs = " ".join(retrieved)
        docs_embedding = self.encoder.encode([combined_docs])[0]

        # Cosine similarity between generated answer and documents
        similarity = cosine_similarity([generated_embedding], [docs_embedding])[0][0]

        return float(similarity)

class BERTScoreMetric(BaseMetric):
    """BERTScore: Semantic similarity between generated and reference answers"""
    def __init__(self, name: str = "bertscore", model_name: str = "bert-base-uncased"):
        super().__init__(name)
        # Simplified BERTScore using sentence transformers F1
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

    def compute(self, retrieved: List[str] = None, relevant: List[str] = None,
                generated: str = "", reference: str = "", **kwargs) -> float:
        if not generated or not reference:
            return 0.0

        # Encode generated and reference
        gen_emb = self.encoder.encode([generated])[0]
        ref_emb = self.encoder.encode([reference])[0]

        # Cosine similarity as proxy for F1 BERTScore
        similarity = cosine_similarity([gen_emb], [ref_emb])[0][0]

        return float(similarity)

class RougeMetric(BaseMetric):
    """ROUGE score between generated and reference answers"""
    def __init__(self, name: str = "rouge", rouge_type: str = "rougeL"):
        super().__init__(name)
        self.rouge_type = rouge_type
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        self.scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)

    def compute(self, retrieved: List[str] = None, relevant: List[str] = None,
                generated: str = "", reference: str = "", **kwargs) -> float:
        if not generated or not reference:
            return 0.0

        scores = self.scorer.score(reference, generated)
        return scores[self.rouge_type].fmeasure

class MRRMetric(BaseMetric):
    """Mean Reciprocal Rank: 1/rank of first relevant document"""
    def compute(self, retrieved: List[str], relevant: List[str], **kwargs) -> float:
        if not relevant:
            return 0.0

        relevant_set = set(relevant)

        for rank, doc in enumerate(retrieved, 1):
            if doc in relevant_set:
                return 1.0 / rank

        return 0.0  # No relevant document found

class MAPMetric(BaseMetric):
    """Mean Average Precision: Precision at each relevant document position"""
    def compute(self, retrieved: List[str], relevant: List[str], **kwargs) -> float:
        if not relevant:
            return 0.0

        relevant_set = set(relevant)
        num_relevant_found = 0
        sum_precision = 0.0

        for i, doc in enumerate(retrieved):
            if doc in relevant_set:
                num_relevant_found += 1
                precision_at_i = num_relevant_found / (i + 1)
                sum_precision += precision_at_i

        return sum_precision / len(relevant) if relevant else 0.0

class NDCGMetric(BaseMetric):
    """Normalized Discounted Cumulative Gain"""
    def __init__(self, name: str = "ndcg", k: int = None):
        super().__init__(name, k)

    def compute(self, retrieved: List[str], relevant: List[str], **kwargs) -> float:
        if not relevant:
            return 0.0

        k = self.k or len(retrieved)
        retrieved_k = retrieved[:k]

        relevant_set = set(relevant)

        # Calculate DCG
        dcg = 0.0
        for i, doc in enumerate(retrieved_k):
            if doc in relevant_set:
                dcg += 1.0 / np.log2(i + 2)  # +2 because i starts at 0

        # Calculate IDCG (ideal DCG)
        idcg = 0.0
        for i in range(min(k, len(relevant))):
            idcg += 1.0 / np.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0

class HallucinationDetectionMetric(BaseMetric):
    """Detect potential hallucinations by checking claim sources"""
    def __init__(self, name: str = "hallucination_detection"):
        super().__init__(name)
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

    def compute(self, retrieved: List[str] = None, relevant: List[str] = None,
                generated: str = "", context: List[Dict] = None, **kwargs) -> float:
        """
        Return 1.0 if no hallucinations detected, 0.0 if hallucinations likely
        """
        if not generated or not context:
            return 0.5  # Uncertain

        # Extract factual claims from generated response
        claims = self._extract_claims(generated)

        if not claims:
            return 1.0  # No factual claims to check

        # Check each claim against context
        hallucinated_claims = 0

        for claim in claims:
            # Check if claim is supported by any document
            claim_supported = False

            claim_emb = self.encoder.encode([claim])[0]
            doc_texts = [doc.get('abstract', doc.get('content', '')) for doc in context if doc]
            if doc_texts:
                doc_embs = self.encoder.encode(doc_texts)

                similarities = cosine_similarity([claim_emb], doc_embs)[0]
                max_sim = np.max(similarities)

                # If high similarity (>0.5), claim is likely supported
                if max_sim > 0.5:
                    claim_supported = True

            if not claim_supported:
                hallucinated_claims += 1

        # Return fraction of non-hallucinated claims
        return 1.0 - (hallucinated_claims / len(claims)) if claims else 0.5

    def _extract_claims(self, text: str) -> List[str]:
        """Extract sentences making factual claims"""
        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            # Fallback without NLTK
            sentences = text.split('. ')

        # Simple heuristic for factual claims
        claim_indicators = ['is', 'are', 'shows', 'demonstrates', 'found', 'proves', 'indicates', 'reveals']

        claims = []
        for sent in sentences:
            sent_lower = sent.lower()
            if any(indicator in sent_lower for indicator in claim_indicators):
                if len(sent.split()) > 5:  # Substantial claim
                    claims.append(sent.strip())

        return claims

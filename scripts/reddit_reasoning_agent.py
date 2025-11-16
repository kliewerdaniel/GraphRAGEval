import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import ollama
from dotenv import load_dotenv
import os

# Ensure we can import from sibling modules
current_dir = Path(__file__).parent
if str(current_dir.parent) not in sys.path:
    sys.path.insert(0, str(current_dir.parent))

from scripts.reddit_retriever import RedditRetriever

# Load environment variables
load_dotenv()

class RedditReasoningAgent:
    def __init__(self,
                 persona_config_path: Path = Path("data/persona.json"),
                 ollama_model: str = "llama3.2:3b"):

        self.persona_config_path = persona_config_path
        self.persona_config = self._load_persona(persona_config_path)
        self.ollama_model = ollama_model
        self.retriever = RedditRetriever(ollama_model=ollama_model)

    def _load_persona(self, config_path: Path) -> Dict[str, Any]:
        """Load persona configuration with RLHF thresholds"""

        if not config_path.exists():
            # Create default persona configuration for Reddit assistant
            default_config = {
                "name": "Reddit Research Assistant",
                "description": "A helpful assistant that analyzes Reddit discussions and conversations",
                "system_prompt_template": """You are a Reddit research assistant analyzing online discussions and conversations.
You have access to a database of Reddit comments and posts and can retrieve relevant discussions to provide insights.

When answering questions:
1. Always cite specific Reddit users, subreddits, and discussion contexts
2. Be balanced and represent different viewpoints from the discussions
3. Explain concepts based on community discussions
4. If you don't have enough information from discussions, say so
5. Organize your responses with clear structure when appropriate

Available context from Reddit discussions:
{context}

Question: {query}""",
                "rlhf_thresholds": {
                    "retrieval_required": 0.6,
                    "minimum_context_overlap": 0.3,
                    "formality_level": 0.6,
                    "technical_detail_level": 0.7,
                    "citation_requirement": 0.8
                },
                "recent_success_rate": 0.8
            }

            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)

            return default_config

        # Load existing config
        with open(config_path) as f:
            return json.load(f)

    def should_retrieve_context(self, query: str) -> bool:
        """
        Decide if we need to retrieve context based on:
        1. Query complexity and discussion-related nature
        2. RLHF confidence threshold
        3. Recent retrieval success rate
        """

        # Analyze query characteristics
        discussion_indicators = [
            'reddit', 'discussion', 'opinion', 'people think', 'community',
            'thread', 'comment', 'post', 'subreddit', 'r/', 'users say'
        ]

        research_keywords = ['what do', 'how do', 'why do', 'compare', 'similar', 'different']

        # Check for discussion-related content
        query_lower = query.lower()
        has_discussion_terms = any(term in query_lower for term in discussion_indicators)
        has_research_question = any(kw in query_lower.split() for kw in research_keywords)

        needs_retrieval = has_discussion_terms or has_research_question

        # Check RLHF threshold
        confidence_threshold = self.persona_config['rlhf_thresholds']['retrieval_required']

        # If recent queries had low-quality responses, lower threshold
        if self.persona_config['recent_success_rate'] < 0.7:
            confidence_threshold *= 0.8

        return needs_retrieval or confidence_threshold > 0.5

    def generate_response(self,
                         query: str,
                         chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Main orchestration logic:
        1. Decide if retrieval needed
        2. Retrieve context if necessary
        3. Generate response with persona coloring
        4. Grade output (RLHF scoring)
        5. Update persona thresholds based on grade
        """

        # Step 1: Retrieval decision
        needs_context = self.should_retrieve_context(query)

        context_docs = []
        if needs_context:
            try:
                context_docs = self.retriever.retrieve_context(query, limit=5)
            except Exception as e:
                print(f"Error retrieving context: {e}")
                context_docs = []

        # Step 2: Format context for LLM
        context_str = self._format_context(context_docs)

        # Step 3: Generate with persona
        system_prompt = self._build_persona_prompt(context_str, context_docs, chat_history)

        try:
            response = ollama.generate(
                model=self.ollama_model,
                prompt=query,
                system=system_prompt
            )
        except Exception as e:
            print(f"Error generating response: {e}")
            response = {"response": "I'm sorry, I encountered an error while processing your query. Please try again."}

        # Step 4: RLHF grading
        quality_grade = self._grade_response(query, response['response'], context_docs)

        # Step 5: Update RLHF thresholds based on grade
        self._update_persona_thresholds(quality_grade)

        return {
            'response': response['response'],
            'context_used': context_docs,
            'quality_grade': quality_grade,
            'retrieval_method': context_docs[0]['retrieval_method'] if context_docs else None,
            'retrieval_performed': needs_context
        }

    def _build_persona_prompt(self, context: str, context_docs: List[Dict[str, Any]], chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Build system prompt from persona configuration.
        This is the 'coloring' step mentioned in the architecture.
        """
        base_template = self.persona_config['system_prompt_template']

        # Insert context if available
        if context:
            base_template = base_template.replace("{context}", context)
        else:
            base_template = base_template.replace("{context}", "No specific Reddit discussion context available.")

        # Insert query placeholder (will be replaced by ollama)
        if "{query}" not in base_template:
            base_template += "\n\nQuestion: {query}"

        # Include chat history if available
        if chat_history:
            formatted_history = self._format_chat_history(chat_history)
            if formatted_history:
                base_template += f"\n\nPrevious conversation:\n{formatted_history}\n\nPlease continue this conversation naturally."

        # Add persona modifiers based on RLHF values
        formality = self.persona_config['rlhf_thresholds']['formality_level']
        if formality > 0.7:
            base_template += "\n\nUse formal, analytical language when discussing Reddit discussions."
        elif formality < 0.4:
            base_template += "\n\nUse conversational language similar to Reddit discussions."

        technical_detail = self.persona_config['rlhf_thresholds']['technical_detail_level']
        if technical_detail > 0.8:
            base_template += "\n\nInclude detailed analysis of discussion patterns and user behavior when relevant."
        elif technical_detail < 0.5:
            base_template += "\n\nFocus on summarizing general opinions and trends."

        citation_req = self.persona_config['rlhf_thresholds']['citation_requirement']
        if citation_req > 0.8:
            base_template += "\n\nALWAYS cite specific Reddit users, subreddits, and discussion links when making claims."
        elif citation_req < 0.5:
            base_template += "\n\nYou can provide general summaries without requiring specific citations."

        return base_template

    def _format_context(self, context_docs: List[Dict[str, Any]]) -> str:
        """Format retrieved Reddit discussions for context"""
        if not context_docs:
            return ""

        formatted = []
        for i, doc in enumerate(context_docs, 1):
            reddit_info = f"""
Reddit Discussion {i}:
User: {doc['author']}
Subreddit: r/{doc['subreddit']}
Score: {doc.get('score', 0)}
Created: {doc.get('created_utc', 'Unknown')}
Topics: {', '.join(doc.get('topics', [])[:3])}
Content: {doc.get('content_preview', doc.get('content', ''))[:400]}
Sentiment: {doc.get('sentiment', 'neutral')}
Retrieval Method: {doc.get('retrieval_method', 'unknown')}
"""
            formatted.append(reddit_info.strip())

        return "\n\n".join(formatted)

    def _format_chat_history(self, chat_history: Optional[List[Dict[str, str]]]) -> Optional[str]:
        """Format chat history for inclusion in system prompt"""
        if not chat_history:
            return None

        # Only keep last few exchanges to avoid context overflow
        recent_history = chat_history[-6:]  # Last 3 user-assistant pairs

        formatted = []
        for msg in recent_history:
            role_prefix = "User: " if msg.get('role') == 'user' else "Assistant: "
            formatted.append(f"{role_prefix}{msg.get('content', '')}")

        return "\n".join(formatted)

    def _grade_response(self, query: str, response: str, context: List[Dict[str, Any]]) -> float:
        """
        RLHF grading: 0 (needs improvement) to 1 (excellent).
        Heuristic-based grading for Reddit discussions.
        """

        if not response or len(response.strip()) < 10:
            return 0.1  # Too short or empty

        # Check for community insights vs available context
        insights_score = self._evaluate_discussion_insights(response, context)
        completeness_score = min(1.0, len(response.split()) / 150)  # Length appropriateness (shorter for Reddit)
        structure_score = self._evaluate_structure(response)

        # Weighted score
        overall_score = (
            0.5 * insights_score +
            0.3 * completeness_score +
            0.2 * structure_score
        )

        return min(1.0, max(0.0, overall_score))

    def _evaluate_discussion_insights(self, response: str, context: List[Dict[str, Any]]) -> float:
        """Check if response provides insights about discussions"""

        if not context:
            return 0.3  # Some baseline if no context needed

        response_lower = response.lower()
        insights_supported = 0
        total_insights = 0

        # Check for mention of users/subreddits from context
        reddit_authors = [doc['author'].lower() for doc in context]
        reddit_subreddits = [doc['subreddit'].lower() for doc in context]

        mentioned_users = sum(1 for author in reddit_authors if author in response_lower)
        mentioned_subs = sum(1 for sub in reddit_subreddits if f"r/{sub}" in response_lower)

        # Bonus for discussion pattern analysis
        has_discussion_terms = any(pattern in response_lower for pattern in [
            'discussion shows', 'people think', 'community believes', 'reddit users', 'based on comments'
        ])

        base_score = min(0.7, (mentioned_users + mentioned_subs) * 0.2)  # Up to 0.7 for mentions
        insight_bonus = 0.3 if has_discussion_terms else 0.0

        return min(1.0, base_score + insight_bonus)

    def _evaluate_structure(self, response: str) -> float:
        """Evaluate response structure and readability"""

        score = 0.5  # Base score

        # Check for paragraphs (good structure)
        paragraphs = response.split('\n\n')
        if len(paragraphs) > 1:
            score += 0.2

        # Check for lists or numbered items
        has_lists = any(line.strip().startswith(('- ', '• ', '1. ', '2. ')) for line in response.split('\n'))
        if has_lists:
            score += 0.1

        # Reasonable length for Reddit analysis
        word_count = len(response.split())
        if 30 <= word_count <= 400:
            score += 0.2

        return min(1.0, score)

    def _update_persona_thresholds(self, quality_grade: float):
        """
        Update RLHF thresholds based on response quality.
        This is the adaptive learning mechanism.
        """

        # If grade < 0.5, we need more context and formality
        if quality_grade < 0.5:
            self.persona_config['rlhf_thresholds']['retrieval_required'] += 0.05
            self.persona_config['rlhf_thresholds']['citation_requirement'] += 0.05
            self.persona_config['rlhf_thresholds']['technical_detail_level'] -= 0.02
            print("⚠️  Low quality response - increasing retrieval aggressiveness")

        # If grade > 0.8, we can be more flexible
        elif quality_grade > 0.8:
            self.persona_config['rlhf_thresholds']['retrieval_required'] -= 0.02
            self.persona_config['rlhf_thresholds']['formality_level'] -= 0.01
            print("✓ High quality response - relaxing thresholds slightly")

        # Update success rate (exponential moving average)
        alpha = 0.1
        self.persona_config['recent_success_rate'] = (
            alpha * (1.0 if quality_grade > 0.6 else 0.0) +
            (1 - alpha) * self.persona_config['recent_success_rate']
        )

        # Clamp values
        thresholds = self.persona_config['rlhf_thresholds']
        for key in thresholds:
            thresholds[key] = max(0.0, min(1.0, thresholds[key]))

        # Save updated config
        with open(self.persona_config_path, 'w') as f:
            json.dump(self.persona_config, f, indent=2)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Reddit reasoning agent")
    parser.add_argument("--query", type=str, help="Query to test")
    parser.add_argument("--history", type=str, help="JSON chat history")

    args = parser.parse_args()

    agent = RedditReasoningAgent()

    if args.query:
        chat_history = []
        if args.history:
            try:
                chat_history = json.loads(args.history)
            except:
                print("Invalid history JSON")

        result = agent.generate_response(args.query, chat_history)

        print(f"Query: {args.query}")
        print(f"Retrieved context: {len(result['context_used'])} discussions")
        print(f"Quality grade: {result['quality_grade']:.2f}")
        print(f"Retrieval method: {result.get('retrieval_method', 'none')}")
        print()
        print("Response:")
        print(result['response'])
        print()
        if result['context_used']:
            print("Reddit Sources:")
            for i, doc in enumerate(result['context_used'], 1):
                print(f"{i}. {doc['author']} in r/{doc['subreddit']} (Score: {doc.get('relevance_score', 0):.3f})")
                if doc.get('content_preview'):
                    print(f"   Preview: {doc['content_preview'][:100]}...")
    else:
        print("Provide a query with --query")

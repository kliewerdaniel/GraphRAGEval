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
    - ANNOTATES: Note -> Paper
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

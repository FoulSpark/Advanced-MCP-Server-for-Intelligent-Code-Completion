"""
Semantic Search Engine - Uses embeddings for context-aware code search
Finds relevant files/functions based on natural language queries
"""
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Represents a search result with relevance score"""
    file_path: str
    symbol_name: str
    symbol_type: str
    line_number: int
    definition: str
    relevance_score: float
    context: str = ""
    
    def to_dict(self):
        return {
            'file_path': self.file_path,
            'symbol_name': self.symbol_name,
            'symbol_type': self.symbol_type,
            'line_number': self.line_number,
            'definition': self.definition,
            'relevance_score': self.relevance_score,
            'context': self.context
        }

class SemanticSearchEngine:
    """Semantic search using embeddings and similarity matching"""
    
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        self.symbol_embeddings: Dict[str, np.ndarray] = {}
        
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding vector for text"""
        # Simple TF-IDF-like embedding for now (can be replaced with OpenAI/HuggingFace)
        # This is a placeholder - in production, use actual embedding models
        
        # Tokenize and create simple bag-of-words vector
        words = text.lower().split()
        vocab = set(words)
        
        # Create a simple embedding (replace with real model)
        embedding = np.zeros(100)  # 100-dimensional vector
        for i, word in enumerate(vocab):
            if i < 100:
                embedding[i] = hash(word) % 1000 / 1000.0
        
        return embedding / (np.linalg.norm(embedding) + 1e-10)
    
    def index_symbols(self, symbol_table: Dict[str, Any]):
        """Create embeddings for all symbols"""
        logger.info("🔄 Creating embeddings for symbols...")
        
        for key, symbol in symbol_table.items():
            # Create text representation of symbol
            text_parts = [
                symbol['name'],
                symbol['type'],
                symbol.get('docstring', ''),
                symbol.get('definition', '')
            ]
            text = ' '.join(filter(None, text_parts))
            
            # Generate embedding
            embedding = self.generate_embedding(text)
            self.symbol_embeddings[key] = embedding
        
        logger.info(f"✅ Created {len(self.symbol_embeddings)} embeddings")
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def search(self, query: str, symbol_table: Dict[str, Any], 
               top_k: int = 10, min_score: float = 0.3) -> List[SearchResult]:
        """Search for symbols matching the query"""
        
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        
        # Calculate similarities
        results = []
        for key, symbol_embedding in self.symbol_embeddings.items():
            similarity = self.cosine_similarity(query_embedding, symbol_embedding)
            
            if similarity >= min_score:
                symbol = symbol_table[key]
                result = SearchResult(
                    file_path=symbol['file_path'],
                    symbol_name=symbol['name'],
                    symbol_type=symbol['type'],
                    line_number=symbol['line_number'],
                    definition=symbol.get('definition', ''),
                    relevance_score=similarity,
                    context=symbol.get('docstring', '')
                )
                results.append(result)
        
        # Sort by relevance score
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return results[:top_k]
    
    def search_by_concept(self, concept: str, symbol_table: Dict[str, Any], 
                         top_k: int = 5) -> List[SearchResult]:
        """Search for symbols related to a concept"""
        
        # Expand concept with related terms
        concept_expansions = {
            'authentication': ['auth', 'login', 'user', 'session', 'token', 'password'],
            'database': ['db', 'query', 'sql', 'model', 'schema', 'table'],
            'logging': ['log', 'logger', 'debug', 'info', 'error', 'trace'],
            'api': ['endpoint', 'route', 'request', 'response', 'http'],
            'validation': ['validate', 'check', 'verify', 'sanitize'],
            'error': ['exception', 'error', 'catch', 'try', 'handle']
        }
        
        # Get related terms
        related_terms = concept_expansions.get(concept.lower(), [concept])
        expanded_query = ' '.join([concept] + related_terms)
        
        return self.search(expanded_query, symbol_table, top_k)
    
    def find_related_files(self, file_path: str, file_index: Dict[str, Any], 
                          top_k: int = 5) -> List[str]:
        """Find files related to a given file based on dependencies"""
        
        if file_path not in file_index:
            return []
        
        file_data = file_index[file_path]
        related_files = set()
        
        # Add direct dependencies
        for dep in file_data.get('dependencies', []):
            # Try to find the actual file path
            for indexed_file in file_index.keys():
                if dep in indexed_file or indexed_file.endswith(f"{dep}.py") or indexed_file.endswith(f"{dep}.js"):
                    related_files.add(indexed_file)
        
        # Find files that import this file
        for other_file, other_data in file_index.items():
            if other_file == file_path:
                continue
            
            for dep in other_data.get('dependencies', []):
                if file_path in dep or dep in file_path:
                    related_files.add(other_file)
        
        return list(related_files)[:top_k]
    
    def save_embeddings(self, output_path: str):
        """Save embeddings to file"""
        embeddings_data = {
            key: embedding.tolist() 
            for key, embedding in self.symbol_embeddings.items()
        }
        
        with open(output_path, 'w') as f:
            json.dump(embeddings_data, f)
        
        logger.info(f"💾 Embeddings saved to {output_path}")
    
    def load_embeddings(self, input_path: str):
        """Load embeddings from file"""
        with open(input_path, 'r') as f:
            embeddings_data = json.load(f)
        
        self.symbol_embeddings = {
            key: np.array(embedding)
            for key, embedding in embeddings_data.items()
        }
        
        logger.info(f"📂 Embeddings loaded from {input_path}")


class AdvancedSemanticSearch(SemanticSearchEngine):
    """Advanced semantic search with OpenAI/HuggingFace embeddings"""
    
    def __init__(self, use_openai: bool = False, openai_api_key: str = None):
        super().__init__()
        self.use_openai = use_openai
        self.openai_api_key = openai_api_key
        
        if use_openai and openai_api_key:
            try:
                import openai
                self.openai_client = openai
                self.openai_client.api_key = openai_api_key
                logger.info("✅ Using OpenAI embeddings")
            except ImportError:
                logger.warning("OpenAI not installed, falling back to simple embeddings")
                self.use_openai = False
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using OpenAI or fallback to simple method"""
        
        if self.use_openai:
            try:
                response = self.openai_client.Embedding.create(
                    input=text,
                    model="text-embedding-ada-002"
                )
                return np.array(response['data'][0]['embedding'])
            except Exception as e:
                logger.warning(f"OpenAI embedding failed: {e}, using fallback")
        
        # Fallback to simple embedding
        return super().generate_embedding(text)

"""
Multi-File Manager - Orchestrates all multi-file operations
Handles natural language queries and coordinates refactoring across files
"""
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from project_indexer import ProjectIndexer, Symbol
from semantic_search import SemanticSearchEngine, SearchResult
from multi_file_editor import MultiFileEditor, MultiFilePatch
from symbol_graph import SymbolGraph

logger = logging.getLogger(__name__)

class MultiFileManager:
    """Main orchestrator for multi-file operations"""
    
    def __init__(self, project_root: str, gemini_api_manager=None):
        self.project_root = Path(project_root)
        self.indexer = ProjectIndexer(str(self.project_root))
        self.search_engine = SemanticSearchEngine()
        self.editor = MultiFileEditor(str(self.project_root))
        self.symbol_graph = SymbolGraph()
        self.gemini_api_manager = gemini_api_manager
        
        self.index_data = None
        self.is_indexed = False
    
    def index_project(self) -> Dict[str, Any]:
        """Index the entire project"""
        logger.info("🔍 Starting project indexing...")
        
        # Scan and index project
        self.index_data = self.indexer.scan_project()
        
        # Build semantic search index
        self.search_engine.index_symbols(self.index_data['symbol_table'])
        
        # Build symbol graph
        self.symbol_graph.build_from_index(
            self.index_data['file_index'],
            self.index_data['symbol_table']
        )
        
        self.is_indexed = True
        logger.info("✅ Project indexing complete")
        
        return {
            'success': True,
            'files_indexed': self.index_data['files_scanned'],
            'symbols_found': self.index_data['symbols_found']
        }
    
    def process_natural_language_query(self, query: str) -> Dict[str, Any]:
        """Process natural language query and return relevant context"""
        
        if not self.is_indexed:
            return {'error': 'Project not indexed. Call index_project() first.'}
        
        # Parse query intent
        intent = self._parse_query_intent(query)
        
        # Search for relevant symbols
        search_results = self.search_engine.search(
            query,
            self.index_data['symbol_table'],
            top_k=10
        )
        
        # Get related files
        relevant_files = set()
        for result in search_results:
            relevant_files.add(result.file_path)
            
            # Add related files from graph
            file_id = f"file::{result.file_path}"
            deps = self.symbol_graph.get_file_dependencies(result.file_path)
            relevant_files.update(deps['imports'])
            relevant_files.update(deps['imported_by'][:3])  # Limit to 3
        
        return {
            'intent': intent,
            'search_results': [r.to_dict() for r in search_results],
            'relevant_files': list(relevant_files),
            'total_matches': len(search_results)
        }
    
    def _parse_query_intent(self, query: str) -> str:
        """Parse the intent from a natural language query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['extract', 'move', 'refactor']):
            return 'refactor'
        elif any(word in query_lower for word in ['add', 'create', 'implement']):
            return 'add_feature'
        elif any(word in query_lower for word in ['find', 'where', 'show']):
            return 'search'
        elif any(word in query_lower for word in ['remove', 'delete']):
            return 'remove'
        elif any(word in query_lower for word in ['replace', 'change', 'update']):
            return 'modify'
        else:
            return 'general'
    
    def extract_to_new_file(self, source_files: List[str], 
                           target_file: str, 
                           symbols_to_extract: List[str],
                           description: str) -> MultiFilePatch:
        """Extract symbols to a new file and update imports"""
        
        changes = {}
        extracted_code = []
        
        # Read source files and extract symbols
        for source_file in source_files:
            file_path = self.project_root / source_file
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Find and extract symbols
            modified_content = original_content
            for symbol_name in symbols_to_extract:
                # Find symbol in file
                symbol_key = f"{source_file}::{symbol_name}"
                if symbol_key in self.index_data['symbol_table']:
                    symbol = self.index_data['symbol_table'][symbol_key]
                    extracted_code.append(symbol.get('definition', ''))
                    
                    # Remove from source (simplified - would need proper AST manipulation)
                    # This is a placeholder for actual implementation
                    modified_content = modified_content.replace(
                        symbol.get('definition', ''),
                        f"# Moved to {target_file}\n"
                    )
            
            # Add import statement
            import_statement = f"from {target_file.replace('.py', '')} import {', '.join(symbols_to_extract)}\n"
            modified_content = import_statement + modified_content
            
            changes[source_file] = (original_content, modified_content)
        
        # Create new target file
        target_content = '\n\n'.join(extracted_code)
        changes[target_file] = ('', target_content)
        
        # Create multi-file patch
        return self.editor.create_multi_file_patch(changes, description)
    
    def add_logging_to_functions(self, concept: str, 
                                 logger_file: str = 'logger.py') -> MultiFilePatch:
        """Add logging to all functions matching a concept"""
        
        # Search for relevant functions
        search_results = self.search_engine.search_by_concept(
            concept,
            self.index_data['symbol_table'],
            top_k=20
        )
        
        # Filter for functions only
        functions = [r for r in search_results if r.symbol_type == 'function']
        
        changes = {}
        files_to_modify = set()
        
        for func in functions:
            files_to_modify.add(func.file_path)
        
        # Modify each file
        for file_path in files_to_modify:
            full_path = self.project_root / file_path
            with open(full_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Add logger import
            modified_content = f"from {logger_file.replace('.py', '')} import logger\n" + original_content
            
            # Add logging statements (simplified)
            for func in functions:
                if func.file_path == file_path:
                    # Add logging at function start
                    func_def = func.definition
                    if func_def in modified_content:
                        log_statement = f'\n    logger.info(f"Calling {func.symbol_name}")'
                        modified_content = modified_content.replace(
                            func_def,
                            func_def + log_statement
                        )
            
            changes[file_path] = (original_content, modified_content)
        
        # Create logger file if it doesn't exist
        logger_path = self.project_root / logger_file
        if not logger_path.exists():
            logger_content = '''import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
'''
            changes[logger_file] = ('', logger_content)
        
        description = f"Add logging to {len(functions)} functions related to '{concept}'"
        return self.editor.create_multi_file_patch(changes, description)
    
    def refactor_with_ai(self, query: str) -> Dict[str, Any]:
        """Use AI to plan and execute refactoring"""
        
        # Get context from query
        context = self.process_natural_language_query(query)
        
        if not self.gemini_api_manager:
            return {'error': 'Gemini API not configured'}
        
        # Build prompt for AI
        prompt = self._build_refactoring_prompt(query, context)
        
        # Get AI response
        api_key = self.gemini_api_manager.get_current_api_key()
        api_url = self.gemini_api_manager.get_api_url(api_key)
        
        import requests
        import json
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 8192
            }
        }
        
        try:
            response = requests.post(
                api_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    ai_response = result['candidates'][0]['content']['parts'][0]['text']
                    
                    # Parse AI response and create patches
                    return self._parse_ai_refactoring_response(ai_response, context)
            
            return {'error': f'AI request failed: {response.status_code}'}
        
        except Exception as e:
            return {'error': f'AI request error: {str(e)}'}
    
    def _build_refactoring_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """Build prompt for AI refactoring"""
        
        prompt = f"""You are a code refactoring expert. Analyze this refactoring request and provide a detailed plan.

**User Request:**
{query}

**Relevant Code Context:**
"""
        
        for result in context['search_results'][:5]:
            prompt += f"\n**File:** {result['file_path']}\n"
            prompt += f"**Symbol:** {result['symbol_name']} ({result['symbol_type']})\n"
            prompt += f"**Definition:**\n```\n{result['definition'][:200]}...\n```\n"
        
        prompt += """

**Provide a refactoring plan in this format:**

1. **Files to Modify:** List all files that need changes
2. **New Files:** List any new files to create
3. **Changes:** For each file, describe the specific changes needed
4. **Imports:** List any new import statements needed
5. **Risk Assessment:** Identify potential issues

Be specific and detailed."""
        
        return prompt
    
    def _parse_ai_refactoring_response(self, ai_response: str, 
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """Parse AI response and create refactoring plan"""
        
        return {
            'success': True,
            'ai_plan': ai_response,
            'context': context,
            'next_steps': [
                'Review the AI-generated plan',
                'Generate code patches',
                'Preview changes',
                'Apply changes after approval'
            ]
        }
    
    def get_impact_analysis(self, file_path: str) -> Dict[str, Any]:
        """Analyze impact of modifying a file"""
        
        file_id = f"file::{file_path}"
        
        # Get dependencies
        deps = self.symbol_graph.get_file_dependencies(file_path)
        
        # Get symbols in file
        symbols = self.indexer.get_file_symbols(file_path)
        
        # Analyze impact for each symbol
        symbol_impacts = []
        for symbol in symbols:
            symbol_key = f"{file_path}::{symbol.name}"
            if symbol_key in self.symbol_graph.nodes:
                impact = self.symbol_graph.get_impact_analysis(symbol_key)
                symbol_impacts.append(impact)
        
        return {
            'file': file_path,
            'imports': deps['imports'],
            'imported_by': deps['imported_by'],
            'symbols': len(symbols),
            'symbol_impacts': symbol_impacts,
            'overall_risk': self._calculate_overall_risk(symbol_impacts)
        }
    
    def _calculate_overall_risk(self, symbol_impacts: List[Dict]) -> str:
        """Calculate overall risk level"""
        if not symbol_impacts:
            return 'low'
        
        high_risk_count = sum(1 for s in symbol_impacts if s.get('risk_level') == 'high')
        
        if high_risk_count > 0:
            return 'high'
        elif len(symbol_impacts) > 5:
            return 'medium'
        else:
            return 'low'

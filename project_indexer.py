"""
Project Indexer - Scans and indexes entire codebase
Builds symbol table with files, functions, classes, imports, and variables
"""
import os
import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class Symbol:
    """Represents a code symbol (function, class, variable)"""
    name: str
    type: str  # 'function', 'class', 'variable', 'import'
    file_path: str
    line_number: int
    definition: str
    docstring: Optional[str] = None
    parameters: List[str] = None
    return_type: Optional[str] = None
    parent: Optional[str] = None  # Parent class/module
    
    def to_dict(self):
        return asdict(self)

@dataclass
class FileIndex:
    """Represents indexed information about a file"""
    file_path: str
    language: str
    imports: List[str]
    exports: List[str]
    functions: List[Dict[str, Any]]
    classes: List[Dict[str, Any]]
    variables: List[Dict[str, Any]]
    dependencies: List[str]
    
    def to_dict(self):
        return asdict(self)

class ProjectIndexer:
    """Indexes entire project codebase for semantic search and navigation"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.symbol_table: Dict[str, Symbol] = {}
        self.file_index: Dict[str, FileIndex] = {}
        self.dependency_graph: Dict[str, Set[str]] = {}
        
        # Supported file extensions
        self.supported_extensions = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust'
        }
        
        # Ignore patterns
        self.ignore_patterns = {
            'node_modules', '__pycache__', '.git', '.venv', 'venv',
            'dist', 'build', '.next', '.cache', 'coverage'
        }
    
    def should_ignore(self, path: Path) -> bool:
        """Check if path should be ignored"""
        for pattern in self.ignore_patterns:
            if pattern in path.parts:
                return True
        return False
    
    def scan_project(self) -> Dict[str, Any]:
        """Scan entire project and build index"""
        logger.info(f"🔍 Scanning project: {self.project_root}")
        
        files_scanned = 0
        symbols_found = 0
        
        for file_path in self.project_root.rglob('*'):
            if not file_path.is_file():
                continue
            
            if self.should_ignore(file_path):
                continue
            
            ext = file_path.suffix.lower()
            if ext not in self.supported_extensions:
                continue
            
            try:
                language = self.supported_extensions[ext]
                
                if language == 'python':
                    file_index = self._index_python_file(file_path)
                elif language in ['javascript', 'typescript']:
                    file_index = self._index_js_file(file_path)
                else:
                    file_index = self._index_generic_file(file_path, language)
                
                if file_index:
                    rel_path = str(file_path.relative_to(self.project_root))
                    self.file_index[rel_path] = file_index
                    files_scanned += 1
                    
                    # Add symbols to symbol table
                    for func in file_index.functions:
                        symbol = Symbol(
                            name=func['name'],
                            type='function',
                            file_path=rel_path,
                            line_number=func['line'],
                            definition=func.get('definition', ''),
                            docstring=func.get('docstring'),
                            parameters=func.get('parameters', [])
                        )
                        self.symbol_table[f"{rel_path}::{func['name']}"] = symbol
                        symbols_found += 1
                    
                    for cls in file_index.classes:
                        symbol = Symbol(
                            name=cls['name'],
                            type='class',
                            file_path=rel_path,
                            line_number=cls['line'],
                            definition=cls.get('definition', ''),
                            docstring=cls.get('docstring')
                        )
                        self.symbol_table[f"{rel_path}::{cls['name']}"] = symbol
                        symbols_found += 1
                    
            except Exception as e:
                logger.warning(f"Failed to index {file_path}: {e}")
        
        logger.info(f"✅ Indexed {files_scanned} files, found {symbols_found} symbols")
        
        return {
            'files_scanned': files_scanned,
            'symbols_found': symbols_found,
            'file_index': {k: v.to_dict() for k, v in self.file_index.items()},
            'symbol_table': {k: v.to_dict() for k, v in self.symbol_table.items()}
        }
    
    def _index_python_file(self, file_path: Path) -> Optional[FileIndex]:
        """Index a Python file using AST"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(file_path))
            
            imports = []
            functions = []
            classes = []
            variables = []
            
            for node in ast.walk(tree):
                # Imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
                
                # Functions
                elif isinstance(node, ast.FunctionDef):
                    func_info = {
                        'name': node.name,
                        'line': node.lineno,
                        'parameters': [arg.arg for arg in node.args.args],
                        'docstring': ast.get_docstring(node),
                        'definition': ast.unparse(node) if hasattr(ast, 'unparse') else ''
                    }
                    functions.append(func_info)
                
                # Classes
                elif isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'line': node.lineno,
                        'bases': [ast.unparse(base) if hasattr(ast, 'unparse') else '' for base in node.bases],
                        'methods': [m.name for m in node.body if isinstance(m, ast.FunctionDef)],
                        'docstring': ast.get_docstring(node),
                        'definition': ast.unparse(node) if hasattr(ast, 'unparse') else ''
                    }
                    classes.append(class_info)
                
                # Global variables
                elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            variables.append({
                                'name': target.id,
                                'line': node.lineno
                            })
            
            return FileIndex(
                file_path=str(file_path.relative_to(self.project_root)),
                language='python',
                imports=imports,
                exports=[],  # Python doesn't have explicit exports
                functions=functions,
                classes=classes,
                variables=variables,
                dependencies=imports
            )
        
        except Exception as e:
            logger.error(f"Error indexing Python file {file_path}: {e}")
            return None
    
    def _index_js_file(self, file_path: Path) -> Optional[FileIndex]:
        """Index a JavaScript/TypeScript file using regex patterns"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            imports = []
            exports = []
            functions = []
            classes = []
            variables = []
            
            # Find imports
            import_patterns = [
                r'import\s+.*?\s+from\s+[\'"](.+?)[\'"]',
                r'require\([\'"](.+?)[\'"]\)',
                r'import\([\'"](.+?)[\'"]\)'
            ]
            for pattern in import_patterns:
                imports.extend(re.findall(pattern, content))
            
            # Find exports
            export_patterns = [
                r'export\s+(?:default\s+)?(?:function|class|const|let|var)\s+(\w+)',
                r'export\s+\{([^}]+)\}',
                r'module\.exports\s*=\s*(\w+)'
            ]
            for pattern in export_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    if isinstance(match, str):
                        exports.extend([m.strip() for m in match.split(',')])
            
            # Find functions
            func_patterns = [
                r'function\s+(\w+)\s*\(([^)]*)\)',
                r'const\s+(\w+)\s*=\s*\(([^)]*)\)\s*=>',
                r'(\w+)\s*:\s*function\s*\(([^)]*)\)',
                r'async\s+function\s+(\w+)\s*\(([^)]*)\)'
            ]
            for pattern in func_patterns:
                for match in re.finditer(pattern, content):
                    line_num = content[:match.start()].count('\n') + 1
                    functions.append({
                        'name': match.group(1),
                        'line': line_num,
                        'parameters': [p.strip() for p in match.group(2).split(',') if p.strip()],
                        'definition': match.group(0)
                    })
            
            # Find classes
            class_pattern = r'class\s+(\w+)(?:\s+extends\s+(\w+))?\s*\{'
            for match in re.finditer(class_pattern, content):
                line_num = content[:match.start()].count('\n') + 1
                classes.append({
                    'name': match.group(1),
                    'line': line_num,
                    'extends': match.group(2) if match.group(2) else None,
                    'definition': match.group(0)
                })
            
            # Find variables
            var_patterns = [
                r'(?:const|let|var)\s+(\w+)\s*=',
                r'(\w+)\s*:\s*(?:string|number|boolean|any)'
            ]
            for pattern in var_patterns:
                for match in re.finditer(pattern, content):
                    line_num = content[:match.start()].count('\n') + 1
                    variables.append({
                        'name': match.group(1),
                        'line': line_num
                    })
            
            return FileIndex(
                file_path=str(file_path.relative_to(self.project_root)),
                language='javascript',
                imports=list(set(imports)),
                exports=list(set(exports)),
                functions=functions,
                classes=classes,
                variables=variables,
                dependencies=list(set(imports))
            )
        
        except Exception as e:
            logger.error(f"Error indexing JS file {file_path}: {e}")
            return None
    
    def _index_generic_file(self, file_path: Path, language: str) -> Optional[FileIndex]:
        """Index a generic file with basic pattern matching"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic function detection
            functions = []
            func_pattern = r'(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\([^)]*\)'
            for match in re.finditer(func_pattern, content):
                line_num = content[:match.start()].count('\n') + 1
                functions.append({
                    'name': match.group(1),
                    'line': line_num,
                    'definition': match.group(0)
                })
            
            return FileIndex(
                file_path=str(file_path.relative_to(self.project_root)),
                language=language,
                imports=[],
                exports=[],
                functions=functions,
                classes=[],
                variables=[],
                dependencies=[]
            )
        
        except Exception as e:
            logger.error(f"Error indexing generic file {file_path}: {e}")
            return None
    
    def search_symbol(self, query: str) -> List[Symbol]:
        """Search for symbols matching query"""
        results = []
        query_lower = query.lower()
        
        for key, symbol in self.symbol_table.items():
            if query_lower in symbol.name.lower():
                results.append(symbol)
        
        return results
    
    def get_file_symbols(self, file_path: str) -> List[Symbol]:
        """Get all symbols in a specific file"""
        results = []
        for key, symbol in self.symbol_table.items():
            if symbol.file_path == file_path:
                results.append(symbol)
        return results
    
    def save_index(self, output_path: str):
        """Save index to JSON file"""
        index_data = {
            'project_root': str(self.project_root),
            'file_index': {k: v.to_dict() for k, v in self.file_index.items()},
            'symbol_table': {k: v.to_dict() for k, v in self.symbol_table.items()}
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2)
        
        logger.info(f"💾 Index saved to {output_path}")
    
    def load_index(self, input_path: str):
        """Load index from JSON file"""
        with open(input_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        
        self.project_root = Path(index_data['project_root'])
        
        # Reconstruct file index
        for file_path, data in index_data['file_index'].items():
            self.file_index[file_path] = FileIndex(**data)
        
        # Reconstruct symbol table
        for key, data in index_data['symbol_table'].items():
            self.symbol_table[key] = Symbol(**data)
        
        logger.info(f"📂 Index loaded from {input_path}")

"""
Symbol Graph - Builds relationship graph between code symbols
Tracks imports, calls, inheritance, and dependencies
"""
import networkx as nx
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class GraphNode:
    """Represents a node in the symbol graph"""
    id: str
    name: str
    type: str  # 'file', 'function', 'class', 'variable'
    file_path: str
    metadata: Dict[str, Any]

@dataclass
class GraphEdge:
    """Represents an edge in the symbol graph"""
    source: str
    target: str
    relationship: str  # 'imports', 'calls', 'inherits', 'uses'
    metadata: Dict[str, Any]

class SymbolGraph:
    """Builds and queries symbol relationship graph"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
    
    def add_node(self, node: GraphNode):
        """Add a node to the graph"""
        self.nodes[node.id] = node
        self.graph.add_node(
            node.id,
            name=node.name,
            type=node.type,
            file_path=node.file_path,
            **node.metadata
        )
    
    def add_edge(self, edge: GraphEdge):
        """Add an edge to the graph"""
        self.edges.append(edge)
        self.graph.add_edge(
            edge.source,
            edge.target,
            relationship=edge.relationship,
            **edge.metadata
        )
    
    def build_from_index(self, file_index: Dict[str, Any], symbol_table: Dict[str, Any]):
        """Build graph from project index"""
        logger.info("🔄 Building symbol graph...")
        
        # Add file nodes
        for file_path, file_data in file_index.items():
            node = GraphNode(
                id=f"file::{file_path}",
                name=file_path,
                type='file',
                file_path=file_path,
                metadata={'language': file_data.get('language', 'unknown')}
            )
            self.add_node(node)
        
        # Add symbol nodes
        for symbol_key, symbol_data in symbol_table.items():
            node = GraphNode(
                id=symbol_key,
                name=symbol_data['name'],
                type=symbol_data['type'],
                file_path=symbol_data['file_path'],
                metadata={
                    'line': symbol_data.get('line_number', 0),
                    'docstring': symbol_data.get('docstring', '')
                }
            )
            self.add_node(node)
            
            # Add edge from file to symbol
            edge = GraphEdge(
                source=f"file::{symbol_data['file_path']}",
                target=symbol_key,
                relationship='contains',
                metadata={}
            )
            self.add_edge(edge)
        
        # Add import relationships
        for file_path, file_data in file_index.items():
            for import_name in file_data.get('imports', []):
                # Try to find the imported file
                for other_file in file_index.keys():
                    if import_name in other_file:
                        edge = GraphEdge(
                            source=f"file::{file_path}",
                            target=f"file::{other_file}",
                            relationship='imports',
                            metadata={'import_name': import_name}
                        )
                        self.add_edge(edge)
        
        logger.info(f"✅ Built graph with {len(self.nodes)} nodes and {len(self.edges)} edges")
    
    def find_dependencies(self, node_id: str, max_depth: int = 3) -> List[str]:
        """Find all dependencies of a node"""
        if node_id not in self.graph:
            return []
        
        dependencies = []
        visited = set()
        
        def dfs(current_id, depth):
            if depth > max_depth or current_id in visited:
                return
            visited.add(current_id)
            
            for neighbor in self.graph.successors(current_id):
                edge_data = self.graph.get_edge_data(current_id, neighbor)
                if edge_data and edge_data.get('relationship') in ['imports', 'uses', 'calls']:
                    dependencies.append(neighbor)
                    dfs(neighbor, depth + 1)
        
        dfs(node_id, 0)
        return dependencies
    
    def find_dependents(self, node_id: str, max_depth: int = 3) -> List[str]:
        """Find all nodes that depend on this node"""
        if node_id not in self.graph:
            return []
        
        dependents = []
        visited = set()
        
        def dfs(current_id, depth):
            if depth > max_depth or current_id in visited:
                return
            visited.add(current_id)
            
            for neighbor in self.graph.predecessors(current_id):
                edge_data = self.graph.get_edge_data(neighbor, current_id)
                if edge_data and edge_data.get('relationship') in ['imports', 'uses', 'calls']:
                    dependents.append(neighbor)
                    dfs(neighbor, depth + 1)
        
        dfs(node_id, 0)
        return dependents
    
    def find_path(self, source_id: str, target_id: str) -> Optional[List[str]]:
        """Find path between two nodes"""
        try:
            return nx.shortest_path(self.graph, source_id, target_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    
    def find_by_concept(self, concept: str) -> List[GraphNode]:
        """Find nodes related to a concept"""
        results = []
        concept_lower = concept.lower()
        
        for node_id, node in self.nodes.items():
            if concept_lower in node.name.lower():
                results.append(node)
            elif concept_lower in node.metadata.get('docstring', '').lower():
                results.append(node)
        
        return results
    
    def get_file_dependencies(self, file_path: str) -> Dict[str, List[str]]:
        """Get all dependencies for a file"""
        file_id = f"file::{file_path}"
        
        if file_id not in self.graph:
            return {'imports': [], 'imported_by': []}
        
        imports = []
        imported_by = []
        
        for neighbor in self.graph.successors(file_id):
            edge_data = self.graph.get_edge_data(file_id, neighbor)
            if edge_data and edge_data.get('relationship') == 'imports':
                if neighbor.startswith('file::'):
                    imports.append(neighbor.replace('file::', ''))
        
        for neighbor in self.graph.predecessors(file_id):
            edge_data = self.graph.get_edge_data(neighbor, file_id)
            if edge_data and edge_data.get('relationship') == 'imports':
                if neighbor.startswith('file::'):
                    imported_by.append(neighbor.replace('file::', ''))
        
        return {
            'imports': imports,
            'imported_by': imported_by
        }
    
    def get_impact_analysis(self, node_id: str) -> Dict[str, Any]:
        """Analyze impact of changing a node"""
        dependents = self.find_dependents(node_id)
        dependencies = self.find_dependencies(node_id)
        
        affected_files = set()
        for dep_id in dependents:
            if dep_id in self.nodes:
                affected_files.add(self.nodes[dep_id].file_path)
        
        return {
            'node': node_id,
            'direct_dependents': len([d for d in dependents if d in self.nodes]),
            'total_dependents': len(dependents),
            'dependencies': len(dependencies),
            'affected_files': list(affected_files),
            'risk_level': 'high' if len(dependents) > 10 else 'medium' if len(dependents) > 3 else 'low'
        }
    
    def export_graph(self, output_path: str, format: str = 'json'):
        """Export graph to file"""
        if format == 'json':
            graph_data = {
                'nodes': [
                    {
                        'id': node.id,
                        'name': node.name,
                        'type': node.type,
                        'file_path': node.file_path,
                        'metadata': node.metadata
                    }
                    for node in self.nodes.values()
                ],
                'edges': [
                    {
                        'source': edge.source,
                        'target': edge.target,
                        'relationship': edge.relationship,
                        'metadata': edge.metadata
                    }
                    for edge in self.edges
                ]
            }
            
            with open(output_path, 'w') as f:
                json.dump(graph_data, f, indent=2)
            
            logger.info(f"💾 Graph exported to {output_path}")
        
        elif format == 'gexf':
            nx.write_gexf(self.graph, output_path)
            logger.info(f"💾 Graph exported to {output_path} (GEXF format)")
    
    def visualize_subgraph(self, node_id: str, depth: int = 2) -> Dict[str, Any]:
        """Get subgraph around a node for visualization"""
        if node_id not in self.graph:
            return {'nodes': [], 'edges': []}
        
        # Get neighbors up to depth
        subgraph_nodes = {node_id}
        current_level = {node_id}
        
        for _ in range(depth):
            next_level = set()
            for node in current_level:
                next_level.update(self.graph.successors(node))
                next_level.update(self.graph.predecessors(node))
            subgraph_nodes.update(next_level)
            current_level = next_level
        
        # Build subgraph data
        nodes_data = []
        edges_data = []
        
        for node_id in subgraph_nodes:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                nodes_data.append({
                    'id': node.id,
                    'label': node.name,
                    'type': node.type,
                    'file': node.file_path
                })
        
        for source in subgraph_nodes:
            for target in subgraph_nodes:
                if self.graph.has_edge(source, target):
                    edge_data = self.graph.get_edge_data(source, target)
                    edges_data.append({
                        'source': source,
                        'target': target,
                        'relationship': edge_data.get('relationship', 'unknown')
                    })
        
        return {
            'nodes': nodes_data,
            'edges': edges_data
        }

#!/usr/bin/env python3
"""
True MCP Integration for Git-based Code Search and Generation
Connects to GitHub, GitLab, and Git MCP servers for repository scraping
"""

import asyncio
import json
import os
import subprocess
import tempfile
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass
from datetime import datetime

# MCP imports (will be installed via requirements.txt)
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("⚠️  MCP libraries not available. Install with: pip install mcp")

@dataclass
class GitSearchResult:
    """Result from git MCP server search"""
    repository: str
    file_path: str
    content: str
    similarity_score: float
    commit_hash: str
    author: str
    timestamp: str

class MCPGitIntegration:
    """Integration with GitHub, GitLab, and Git MCP servers"""
    
    def __init__(self):
        self.mcp_sessions = {}
        self.logger = logging.getLogger(__name__)
        self.github_token = os.getenv("GITHUB_TOKEN", "")
        self.gitlab_token = os.getenv("GITLAB_TOKEN", "")
        
    async def initialize_mcp_servers(self) -> bool:
        """Initialize connections to MCP servers"""
        if not MCP_AVAILABLE:
            self.logger.error("MCP libraries not available")
            return False
        
        success_count = 0
        
        # Initialize GitHub MCP Server
        if self.github_token:
            try:
                github_params = StdioServerParameters(
                    command="npx",
                    args=["@modelcontextprotocol/server-github"],
                    env={"GITHUB_PERSONAL_ACCESS_TOKEN": self.github_token}
                )
                
                github_session = await stdio_client(github_params)
                await github_session.initialize()
                self.mcp_sessions['github'] = github_session
                success_count += 1
                self.logger.info("✅ GitHub MCP server connected")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to connect to GitHub MCP: {e}")
        
        # Initialize GitLab MCP Server (if available)
        if self.gitlab_token:
            try:
                gitlab_params = StdioServerParameters(
                    command="npx",
                    args=["@modelcontextprotocol/server-gitlab"],
                    env={"GITLAB_TOKEN": self.gitlab_token}
                )
                
                gitlab_session = await stdio_client(gitlab_params)
                await gitlab_session.initialize()
                self.mcp_sessions['gitlab'] = gitlab_session
                success_count += 1
                self.logger.info("✅ GitLab MCP server connected")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to connect to GitLab MCP: {e}")
        
        # Initialize Filesystem MCP Server for local git repos
        try:
            workspace_dir = os.getcwd()
            fs_params = StdioServerParameters(
                command="npx",
                args=["@modelcontextprotocol/server-filesystem"],
                env={"MCP_FILESYSTEM_ALLOWED_DIRECTORIES": workspace_dir}
            )
            
            fs_session = await stdio_client(fs_params)
            await fs_session.initialize()
            self.mcp_sessions['filesystem'] = fs_session
            success_count += 1
            self.logger.info("✅ Filesystem MCP server connected")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Filesystem MCP: {e}")
        
        return success_count > 0
    
    async def search_github_repositories(self, query: str, language: str = "python") -> List[GitSearchResult]:
        """Search GitHub repositories via MCP for similar code"""
        results = []
        
        if 'github' not in self.mcp_sessions:
            return results
        
        try:
            # Use GitHub MCP server to search repositories
            search_response = await self.mcp_sessions['github'].call_tool(
                "search_repositories",
                arguments={
                    "query": f"{query} language:{language}",
                    "sort": "stars",
                    "order": "desc",
                    "per_page": 10
                }
            )
            
            # Process search results
            if hasattr(search_response, 'content') and search_response.content:
                repos = json.loads(search_response.content[0].text) if search_response.content else []
                
                for repo in repos.get('items', [])[:5]:  # Limit to top 5 repos
                    # Get repository contents
                    repo_contents = await self._get_repository_contents(
                        repo['full_name'], 
                        query, 
                        language
                    )
                    results.extend(repo_contents)
            
        except Exception as e:
            self.logger.error(f"Error searching GitHub repositories: {e}")
        
        return results
    
    async def _get_repository_contents(self, repo_name: str, query: str, language: str) -> List[GitSearchResult]:
        """Get specific file contents from a repository"""
        results = []
        
        try:
            # Search for files in the repository
            file_search = await self.mcp_sessions['github'].call_tool(
                "search_code",
                arguments={
                    "q": f"{query} repo:{repo_name} language:{language}",
                    "per_page": 5
                }
            )
            
            if hasattr(file_search, 'content') and file_search.content:
                files = json.loads(file_search.content[0].text) if file_search.content else []
                
                for file_info in files.get('items', []):
                    # Get file content
                    file_content = await self.mcp_sessions['github'].read_resource(
                        f"github://{repo_name}/{file_info['path']}"
                    )
                    
                    if file_content and hasattr(file_content, 'contents'):
                        results.append(GitSearchResult(
                            repository=repo_name,
                            file_path=file_info['path'],
                            content=file_content.contents[:1000],  # First 1000 chars
                            similarity_score=file_info.get('score', 0.0),
                            commit_hash=file_info.get('sha', ''),
                            author='',  # Would need additional API call
                            timestamp=''
                        ))
            
        except Exception as e:
            self.logger.error(f"Error getting repository contents for {repo_name}: {e}")
        
        return results
    
    async def search_local_git_history(self, workspace_path: str, query: str) -> List[GitSearchResult]:
        """Search local git history via MCP"""
        results = []
        
        if 'filesystem' not in self.mcp_sessions:
            return results
        
        try:
            # Use filesystem MCP to access local git repository
            git_log = await self.mcp_sessions['filesystem'].call_tool(
                "execute_command",
                arguments={
                    "command": "git",
                    "args": ["log", "--oneline", "--grep", query, "-n", "10"],
                    "cwd": workspace_path
                }
            )
            
            if hasattr(git_log, 'content') and git_log.content:
                commits = git_log.content[0].text.strip().split('\n') if git_log.content else []
                
                for commit_line in commits:
                    if commit_line.strip():
                        commit_hash = commit_line.split()[0]
                        
                        # Get commit details
                        commit_details = await self._get_commit_details(workspace_path, commit_hash)
                        if commit_details:
                            results.append(commit_details)
            
        except Exception as e:
            self.logger.error(f"Error searching local git history: {e}")
        
        return results
    
    async def _get_commit_details(self, workspace_path: str, commit_hash: str) -> Optional[GitSearchResult]:
        """Get details of a specific commit"""
        try:
            # Get commit diff
            diff_result = await self.mcp_sessions['filesystem'].call_tool(
                "execute_command",
                arguments={
                    "command": "git",
                    "args": ["show", "--format=%an|%ad|%s", commit_hash],
                    "cwd": workspace_path
                }
            )
            
            if hasattr(diff_result, 'content') and diff_result.content:
                diff_content = diff_result.content[0].text if diff_result.content else ""
                lines = diff_content.split('\n')
                
                if lines:
                    # Parse commit info
                    commit_info = lines[0].split('|')
                    author = commit_info[0] if len(commit_info) > 0 else ""
                    timestamp = commit_info[1] if len(commit_info) > 1 else ""
                    message = commit_info[2] if len(commit_info) > 2 else ""
                    
                    # Get code changes
                    code_lines = [line for line in lines[1:] if line.startswith('+') and not line.startswith('+++')]
                    code_content = '\n'.join(code_lines[:20])  # First 20 added lines
                    
                    return GitSearchResult(
                        repository="local",
                        file_path="",
                        content=code_content,
                        similarity_score=1.0,
                        commit_hash=commit_hash,
                        author=author,
                        timestamp=timestamp
                    )
            
        except Exception as e:
            self.logger.error(f"Error getting commit details for {commit_hash}: {e}")
        
        return None
    
    async def generate_code_from_git_context(self, 
                                           query: str, 
                                           language: str, 
                                           workspace_path: str) -> Dict[str, Any]:
        """Generate code completion using git MCP servers"""
        
        # Search multiple git sources
        github_results = await self.search_github_repositories(query, language)
        local_results = await self.search_local_git_history(workspace_path, query)
        
        # Combine results
        all_results = github_results + local_results
        
        # Sort by similarity score
        all_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Build context for AI completion
        git_context = {
            "total_results": len(all_results),
            "github_results": len(github_results),
            "local_results": len(local_results),
            "top_matches": []
        }
        
        # Add top matches to context
        for result in all_results[:5]:  # Top 5 results
            git_context["top_matches"].append({
                "repository": result.repository,
                "file_path": result.file_path,
                "content": result.content,
                "similarity": result.similarity_score,
                "author": result.author,
                "commit": result.commit_hash
            })
        
        return git_context
    
    async def cleanup(self):
        """Clean up MCP connections"""
        for session_name, session in self.mcp_sessions.items():
            try:
                await session.close()
                self.logger.info(f"✅ Closed {session_name} MCP session")
            except Exception as e:
                self.logger.error(f"Error closing {session_name} session: {e}")
        
        self.mcp_sessions.clear()

# Async wrapper for use in Flask app
class MCPGitManager:
    """Manager for MCP Git integration in Flask app"""
    
    def __init__(self):
        self.mcp_integration = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize MCP integration"""
        if not self.initialized:
            self.mcp_integration = MCPGitIntegration()
            self.initialized = await self.mcp_integration.initialize_mcp_servers()
        return self.initialized
    
    async def search_git_for_completion(self, query: str, language: str, workspace_path: str) -> Dict[str, Any]:
        """Search git repositories for code completion context"""
        if not self.initialized:
            await self.initialize()
        
        if self.mcp_integration:
            return await self.mcp_integration.generate_code_from_git_context(
                query, language, workspace_path
            )
        
        return {"error": "MCP integration not available"}
    
    async def cleanup(self):
        """Cleanup MCP connections"""
        if self.mcp_integration:
            await self.mcp_integration.cleanup()

# Global MCP manager instance
mcp_git_manager = MCPGitManager()

# Helper function for Flask integration
def run_async_in_thread(coro):
    """Run async function in thread for Flask compatibility"""
    import threading
    import asyncio
    
    result = {}
    exception = {}
    
    def run_in_thread():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result['value'] = loop.run_until_complete(coro)
        except Exception as e:
            exception['error'] = e
        finally:
            loop.close()
    
    thread = threading.Thread(target=run_in_thread)
    thread.start()
    thread.join(timeout=30)  # 30 second timeout
    
    if 'error' in exception:
        raise exception['error']
    
    return result.get('value', {})

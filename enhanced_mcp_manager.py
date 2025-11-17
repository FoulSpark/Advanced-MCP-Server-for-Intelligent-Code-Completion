#!/usr/bin/env python3
"""
Enhanced MCP Manager for Code Completion
Handles GitHub, GitLab, and local git integration with proper error handling
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
import requests

# MCP imports
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("⚠️  MCP libraries not available. Install with: pip install mcp")

@dataclass
class CodeExample:
    """Code example from repository search"""
    repository: str
    file_path: str
    content: str
    similarity_score: float
    language: str
    stars: int = 0
    author: str = ""
    commit_hash: str = ""

class EnhancedMCPManager:
    """Enhanced MCP manager with better error handling and fallbacks"""
    
    def __init__(self):
        self.mcp_sessions = {}
        self.logger = logging.getLogger(__name__)
        self.github_token = os.getenv("GITHUB_TOKEN", "")
        self.gitlab_token = os.getenv("GITLAB_TOKEN", "")
        self.initialized = False
        
    async def initialize(self) -> bool:
        """Initialize MCP servers with proper error handling"""
        if not MCP_AVAILABLE:
            self.logger.warning("MCP libraries not available - using fallback methods")
            return False
        
        success_count = 0
        
        # Try to initialize GitHub MCP Server
        if self.github_token:
            try:
                github_params = StdioServerParameters(
                    command="npx",
                    args=["@modelcontextprotocol/server-github"],
                    env={"GITHUB_PERSONAL_ACCESS_TOKEN": self.github_token}
                )
                
                # Use a different approach for session management
                github_session = await stdio_client(github_params).__aenter__()
                await github_session.initialize()
                self.mcp_sessions['github'] = github_session
                success_count += 1
                self.logger.info("✅ GitHub MCP server connected")
                
            except Exception as e:
                self.logger.warning(f"GitHub MCP failed: {e} - using direct API fallback")
                # Set up direct API fallback
                self.mcp_sessions['github_api'] = True
        
        # Try to initialize GitLab MCP Server
        if self.gitlab_token:
            try:
                gitlab_params = StdioServerParameters(
                    command="npx",
                    args=["@modelcontextprotocol/server-gitlab"],
                    env={"GITLAB_TOKEN": self.gitlab_token}
                )
                
                gitlab_session = await stdio_client(gitlab_params).__aenter__()
                await gitlab_session.initialize()
                self.mcp_sessions['gitlab'] = gitlab_session
                success_count += 1
                self.logger.info("✅ GitLab MCP server connected")
                
            except Exception as e:
                self.logger.warning(f"GitLab MCP failed: {e} - using direct API fallback")
                self.mcp_sessions['gitlab_api'] = True
        
        # Always try filesystem MCP
        try:
            workspace_dir = os.getcwd()
            fs_params = StdioServerParameters(
                command="npx",
                args=["@modelcontextprotocol/server-filesystem"],
                env={"MCP_FILESYSTEM_ALLOWED_DIRECTORIES": workspace_dir}
            )
            
            fs_session = await stdio_client(fs_params).__aenter__()
            await fs_session.initialize()
            self.mcp_sessions['filesystem'] = fs_session
            success_count += 1
            self.logger.info("✅ Filesystem MCP server connected")
            
        except Exception as e:
            self.logger.warning(f"Filesystem MCP failed: {e} - using local git fallback")
        
        self.initialized = success_count > 0 or 'github_api' in self.mcp_sessions or 'gitlab_api' in self.mcp_sessions
        return self.initialized
    
    async def search_github_repositories(self, query: str, language: str = "python") -> List[CodeExample]:
        """Search GitHub repositories for code examples"""
        results = []
        
        # Try MCP first
        if 'github' in self.mcp_sessions:
            try:
                search_response = await self.mcp_sessions['github'].call_tool(
                    "search_repositories",
                    arguments={
                        "query": f"{query} language:{language}",
                        "sort": "stars",
                        "order": "desc",
                        "per_page": 10
                    }
                )
                
                if hasattr(search_response, 'content') and search_response.content:
                    repos = json.loads(search_response.content[0].text)
                    for repo in repos.get('items', [])[:5]:
                        examples = await self._get_repository_code_examples(
                            repo['full_name'], query, language
                        )
                        results.extend(examples)
                        
            except Exception as e:
                self.logger.warning(f"GitHub MCP search failed: {e}")
        
        # Fallback to direct API
        if not results and 'github_api' in self.mcp_sessions:
            results = await self._search_github_api(query, language)
        
        return results
    
    async def _search_github_api(self, query: str, language: str) -> List[CodeExample]:
        """Direct GitHub API search fallback"""
        results = []
        
        try:
            # Search repositories
            repo_url = "https://api.github.com/search/repositories"
            headers = {"Authorization": f"token {self.github_token}"}
            params = {
                "q": f"{query} language:{language}",
                "sort": "stars",
                "order": "desc",
                "per_page": 5
            }
            
            repo_response = requests.get(repo_url, headers=headers, params=params, timeout=10)
            if repo_response.status_code == 200:
                repos = repo_response.json()
                
                for repo in repos.get('items', []):
                    # Search code in repository
                    code_url = "https://api.github.com/search/code"
                    code_params = {
                        "q": f"{query} repo:{repo['full_name']} language:{language}",
                        "per_page": 3
                    }
                    
                    code_response = requests.get(code_url, headers=headers, params=code_params, timeout=10)
                    if code_response.status_code == 200:
                        code_items = code_response.json()
                        
                        for item in code_items.get('items', []):
                            # Get file content
                            content_url = f"https://api.github.com/repos/{repo['full_name']}/contents/{item['path']}"
                            content_response = requests.get(content_url, headers=headers, timeout=10)
                            
                            if content_response.status_code == 200:
                                content_data = content_response.json()
                                if content_data.get('type') == 'file':
                                    import base64
                                    content = base64.b64decode(content_data['content']).decode('utf-8')
                                    
                                    results.append(CodeExample(
                                        repository=repo['full_name'],
                                        file_path=item['path'],
                                        content=content[:1000],  # First 1000 chars
                                        similarity_score=0.8,  # Default score
                                        language=language,
                                        stars=repo.get('stargazers_count', 0)
                                    ))
                                    
                                    if len(results) >= 10:  # Limit results
                                        break
                        
                        if len(results) >= 10:
                            break
                            
        except Exception as e:
            self.logger.error(f"GitHub API search failed: {e}")
        
        return results
    
    async def _get_repository_code_examples(self, repo_name: str, query: str, language: str) -> List[CodeExample]:
        """Get code examples from a specific repository"""
        examples = []
        
        try:
            # Search for files in the repository
            file_search = await self.mcp_sessions['github'].call_tool(
                "search_code",
                arguments={
                    "q": f"{query} repo:{repo_name} language:{language}",
                    "per_page": 3
                }
            )
            
            if hasattr(file_search, 'content') and file_search.content:
                files = json.loads(file_search.content[0].text)
                
                for file_info in files.get('items', []):
                    # Get file content
                    file_content = await self.mcp_sessions['github'].read_resource(
                        f"github://{repo_name}/{file_info['path']}"
                    )
                    
                    if file_content and hasattr(file_content, 'contents'):
                        examples.append(CodeExample(
                            repository=repo_name,
                            file_path=file_info['path'],
                            content=file_content.contents[:1000],
                            similarity_score=file_info.get('score', 0.0),
                            language=language
                        ))
                        
        except Exception as e:
            self.logger.error(f"Error getting code examples from {repo_name}: {e}")
        
        return examples
    
    async def search_local_git_history(self, workspace_path: str, query: str) -> List[CodeExample]:
        """Search local git history for code examples"""
        results = []
        
        if 'filesystem' in self.mcp_sessions:
            try:
                # Use MCP filesystem to search git
                git_log = await self.mcp_sessions['filesystem'].call_tool(
                    "execute_command",
                    arguments={
                        "command": "git",
                        "args": ["log", "--oneline", "--grep", query, "-n", "10"],
                        "cwd": workspace_path
                    }
                )
                
                if hasattr(git_log, 'content') and git_log.content:
                    commits = git_log.content[0].text.strip().split('\n')
                    
                    for commit_line in commits:
                        if commit_line.strip():
                            commit_hash = commit_line.split()[0]
                            commit_details = await self._get_commit_code_example(workspace_path, commit_hash)
                            if commit_details:
                                results.append(commit_details)
                                
            except Exception as e:
                self.logger.warning(f"Local git MCP search failed: {e}")
        
        # Fallback to direct git commands
        if not results:
            results = await self._search_local_git_fallback(workspace_path, query)
        
        return results
    
    async def _search_local_git_fallback(self, workspace_path: str, query: str) -> List[CodeExample]:
        """Fallback local git search using direct commands"""
        results = []
        
        try:
            # Use subprocess to run git commands
            result = subprocess.run(
                ["git", "log", "--oneline", "--grep", query, "-n", "10"],
                cwd=workspace_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                commits = result.stdout.strip().split('\n')
                
                for commit_line in commits:
                    if commit_line.strip():
                        commit_hash = commit_line.split()[0]
                        
                        # Get commit diff
                        diff_result = subprocess.run(
                            ["git", "show", "--format=%an|%ad|%s", commit_hash],
                            cwd=workspace_path,
                            capture_output=True,
                            text=True,
                            timeout=10
                        )
                        
                        if diff_result.returncode == 0:
                            diff_content = diff_result.stdout
                            lines = diff_content.split('\n')
                            
                            if lines:
                                commit_info = lines[0].split('|')
                                author = commit_info[0] if len(commit_info) > 0 else ""
                                timestamp = commit_info[1] if len(commit_info) > 1 else ""
                                
                                # Get code changes
                                code_lines = [line for line in lines[1:] if line.startswith('+') and not line.startswith('+++')]
                                code_content = '\n'.join(code_lines[:20])
                                
                                if code_content.strip():
                                    results.append(CodeExample(
                                        repository="local",
                                        file_path="",
                                        content=code_content,
                                        similarity_score=1.0,
                                        language="python",
                                        author=author,
                                        commit_hash=commit_hash
                                    ))
                                    
        except Exception as e:
            self.logger.error(f"Local git fallback search failed: {e}")
        
        return results
    
    async def _get_commit_code_example(self, workspace_path: str, commit_hash: str) -> Optional[CodeExample]:
        """Get code example from a specific commit"""
        try:
            diff_result = await self.mcp_sessions['filesystem'].call_tool(
                "execute_command",
                arguments={
                    "command": "git",
                    "args": ["show", "--format=%an|%ad|%s", commit_hash],
                    "cwd": workspace_path
                }
            )
            
            if hasattr(diff_result, 'content') and diff_result.content:
                diff_content = diff_result.content[0].text
                lines = diff_content.split('\n')
                
                if lines:
                    commit_info = lines[0].split('|')
                    author = commit_info[0] if len(commit_info) > 0 else ""
                    timestamp = commit_info[1] if len(commit_info) > 1 else ""
                    
                    # Get code changes
                    code_lines = [line for line in lines[1:] if line.startswith('+') and not line.startswith('+++')]
                    code_content = '\n'.join(code_lines[:20])
                    
                    if code_content.strip():
                        return CodeExample(
                            repository="local",
                            file_path="",
                            content=code_content,
                            similarity_score=1.0,
                            language="python",
                            author=author,
                            commit_hash=commit_hash
                        )
                        
        except Exception as e:
            self.logger.error(f"Error getting commit details for {commit_hash}: {e}")
        
        return None
    
    async def generate_code_context(self, query: str, language: str, workspace_path: str) -> Dict[str, Any]:
        """Generate comprehensive code context from all sources"""
        
        # Search all available sources
        github_results = []
        local_results = []
        
        if 'github' in self.mcp_sessions or 'github_api' in self.mcp_sessions:
            github_results = await self.search_github_repositories(query, language)
        
        if 'filesystem' in self.mcp_sessions or os.path.exists(os.path.join(workspace_path, '.git')):
            local_results = await self.search_local_git_history(workspace_path, query)
        
        # Combine and sort results
        all_results = github_results + local_results
        all_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Build context
        context = {
            "total_results": len(all_results),
            "github_results": len(github_results),
            "local_results": len(local_results),
            "top_matches": [],
            "mcp_enabled": self.initialized,
            "sources_available": list(self.mcp_sessions.keys())
        }
        
        # Add top matches
        for result in all_results[:5]:
            context["top_matches"].append({
                "repository": result.repository,
                "file_path": result.file_path,
                "content": result.content,
                "similarity": result.similarity_score,
                "language": result.language,
                "author": result.author,
                "commit": result.commit_hash,
                "stars": getattr(result, 'stars', 0)
            })
        
        return context
    
    async def cleanup(self):
        """Clean up MCP connections"""
        for session_name, session in self.mcp_sessions.items():
            if hasattr(session, 'close'):
                try:
                    await session.close()
                    self.logger.info(f"✅ Closed {session_name} MCP session")
                except Exception as e:
                    self.logger.error(f"Error closing {session_name} session: {e}")
        
        self.mcp_sessions.clear()
        self.initialized = False

# Global enhanced MCP manager
enhanced_mcp_manager = EnhancedMCPManager()

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



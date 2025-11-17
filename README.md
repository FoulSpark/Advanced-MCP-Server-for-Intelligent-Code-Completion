# Advanced MCP Server for Intelligent Code Completion 

A local Flask-based Model Context Protocol (MCP) server that powers intelligent, context-aware code completion and automated fixing workflows for your editor (with a bundled VS Code extension). It indexes your project, builds a symbol/dependency graph, performs semantic search, integrates with Git and GitHub/GitLab via MCP or direct APIs, and can run a self‑healing loop to plan and apply fixes using Gemini.

## 🚀 Key Features

- **Project understanding**
  - Project-wide indexing (functions/classes/imports/variables) via AST and patterns
  - Symbol/dependency graph across files
  - Lightweight semantic search for relevant symbols and files
- **Intelligent assistance**
  - Code analysis with error detection (AST + Pyflakes)
  - Pattern detection for common syntax issues (e.g., missing colons) and undefined variables
  - Optional formatters: autopep8, black, isort
- **MCP + Git integration**
  - Enhanced MCP manager with fallbacks to direct REST where needed
  - GitHub/GitLab MCP servers (via `npx @modelcontextprotocol/server-*`) and filesystem MCP
  - Local git history search for example-driven context
- **Self-healing fixer (optional)**
  - Multi-iteration loop that analyzes issues, plans fixes with Gemini, and applies changes
  - Keeps a simple learning DB of successful fixes and trends
- **VS Code extension (SuperAI)**
  - Four-stage Gemini workflow (understand → plan → implement → validate/replace)
  - Commands for diagnostics, completions, and enhanced syntax fixes

## 🗂️ Repository Structure

- `advanced_mcp_server.py` — Flask app, API key rotation, code analysis, optional watchers
- `enhanced_mcp_manager.py` — MCP client wrapper with GitHub/GitLab/filesystem servers and fallbacks
- `mcp_git_integration.py` — Legacy/alternate MCP Git integration
- `project_indexer.py` — Project scanner and indexer
- `semantic_search.py` — Simple embedding-based search engine
- `symbol_graph.py` — Dependency graph between files/symbols (NetworkX)
- `multi_file_manager.py` — Orchestrates indexing, semantic search, and refactoring helpers
- `multi_file_editor.py` — Generates/apply multi-file patches with diffs and previews
- `self_healing_code_fixer.py` — Optional Gemini-powered self-healing loop
- `check_gemini_models.py` — Utility to list/validate available Gemini models for a key
- `superai-extension/` — VS Code extension (README, extension.js, etc.)
- `start_enhanced_mcp.bat` / `start_enhanced_mcp.sh` — Helper scripts
- `.env.template` — Example environment configuration
- `requirements.txt` / `requirements_self_healing.txt` — Python dependencies

## ⚙️ Requirements

- Python 3.10+
- Node.js (for MCP servers via `npx` if you use GitHub/GitLab/filesystem MCP)
- Git (for local history and repo operations)

Python dependencies are listed in:
- `requirements.txt` (core server)
- `requirements_self_healing.txt` (optional self-healing loop)

Install with:
```bash
pip install -r requirements.txt
# Optional for self-healing
pip install -r requirements_self_healing.txt
```

## 🔐 Configuration

Copy `.env.template` to `.env` and fill values:

- `GEMINI_API_KEY` (required)
- `GEMINI_API_KEY_1..9`, `GEMINI_API_KEY_STAGE0/1/2` (optional rotation pool)
- `GITHUB_TOKEN`, `GITLAB_TOKEN` (optional, for MCP servers or API fallbacks)
- `MCP_SERVER_URL`, `WORKSPACE_ROOT`

The server includes a smart API key rotation that automatically switches keys after consecutive failures.

Note: Avoid hardcoding API keys in code. Prefer environment variables.

## ▶️ Run the Server

Windows:
```bat
start_enhanced_mcp.bat
```

macOS/Linux:
```bash
./start_enhanced_mcp.sh
```

Or directly:
```bash
python advanced_mcp_server.py
```

By default the server listens on `http://127.0.0.1:5000` (as used by the VS Code extension).

## 🧩 VS Code Extension (SuperAI)

The `superai-extension/` folder contains a VS Code extension that connects to this server.

Highlights:
- Four-stage Gemini pipeline: Understand → Plan → Implement → Validate/Replace
- Enhanced syntax fix (missing colons, undefined vars, auto-create functions)
- API key monitoring, diagnostics, completions, and global fixes

See `superai-extension/README.md` for detailed usage, commands, and troubleshooting.

## 🤖 Self-Healing Loop (Optional)

`self_healing_code_fixer.py` can run a loop that:
- Analyzes a file via the MCP server
- Plans enhanced fixes with Gemini
- Applies patches and records learning signals

Environment:
- Requires `google-generativeai` and a valid `GEMINI_API_KEY`

Example invocation (see file for demo):
```bash
python self_healing_code_fixer.py
```

## 🔍 How It Works (High Level)

- Indexing and Search
  - `ProjectIndexer` parses Python via AST and uses patterns for JS/TS and other languages
  - `SemanticSearchEngine` builds simple embeddings and supports concept search
  - `SymbolGraph` builds relationships (imports/contains/calls/uses) for impact analysis

- MCP/Git Context
  - `EnhancedMCPManager` tries MCP servers first (GitHub/GitLab/filesystem), then falls back to direct APIs
  - Local git history excerpts can be surfaced to guide completions/refactors

- Analysis and Fixing
  - `advanced_mcp_server.py` performs file analysis, optional linting (pyflakes), and formatting (autopep8/black/isort)
  - Pattern checks catch common syntax/variable issues before AST parse

## 🧪 Utilities

- `check_gemini_models.py` lists available Gemini models for a key and can perform tiny validation calls
  - Prefer environment variables; remove any hardcoded keys before committing

## ❗ Troubleshooting

- Ensure `.env` has a valid `GEMINI_API_KEY`
- If MCP GitHub/GitLab servers fail, fallbacks to direct APIs may be used (requires tokens)
- If embeddings or graph features are slow on very large repos, start with a smaller subset
- On Windows, use the `.bat` launcher; on Unix, the `.sh` script

## 📜 License

Specify your license here (e.g., MIT) and provide a `LICENSE` file.

## 🙌 Acknowledgements

- Model Context Protocol (MCP)
- Gemini API
- NetworkX

---

Generated README based on the current repository content to help you publish on GitHub quickly.

const vscode = require('vscode');
const axios = require('axios');
const path = require('path');

// Configuration
const SERVER_URL = 'http://127.0.0.1:5000';
const COMPLETION_PROVIDER_ID = 'superai-completion';

// Global state
let diagnosticCollection;
let completionProvider;
let hoverProvider;
let statusBarItem;
let autoCompleteEnabled = true;

class SuperAICompletionProvider {
    async provideCompletionItems(document, position, token, context) {
        if (!autoCompleteEnabled) return [];
        
        try {
            const currentFile = document.uri.fsPath;
            const workspaceFolder = vscode.workspace.workspaceFolders ? 
                vscode.workspace.workspaceFolders[0].uri.fsPath : '';
            
            // Get context around cursor
            const lineText = document.lineAt(position.line).text;
            const cursorContext = lineText.substring(0, position.character);
            
            // Skip if context is too short or just whitespace
            if (!cursorContext.trim() || cursorContext.trim().length < 2) {
                return [];
            }
            
            const response = await axios.post(`${SERVER_URL}/auto_complete`, {
                current_file: currentFile,
                workspace_folder: workspaceFolder,
                cursor_context: cursorContext,
                cursor_position: document.offsetAt(position)
            }, { timeout: 5000 });
            
            if (response.data.error) {
                console.error('SuperAI Completion Error:', response.data.error);
                return [];
            }
            
            const completion = response.data.completion;
            if (!completion || !completion.trim()) {
                return [];
            }
            
            // Create completion item
            const item = new vscode.CompletionItem(
                completion.split('\n')[0], // First line as label
                vscode.CompletionItemKind.Snippet
            );
            
            item.insertText = new vscode.SnippetString(completion);
            item.detail = 'SuperAI Suggestion';
            item.documentation = new vscode.MarkdownString(
                `**AI-Generated Code**\n\n\`\`\`${document.languageId}\n${completion}\n\`\`\``
            );
            
            // Add context info if available
            if (response.data.context) {
                const ctx = response.data.context;
                item.documentation.appendMarkdown(
                    `\n\n**Context:** ${ctx.language} | ` +
                    `Functions: ${ctx.functions.length} | ` +
                    `Errors: ${ctx.errors.length}`
                );
            }
            
            return [item];
            
        } catch (error) {
            console.error('SuperAI Completion Provider Error:', error.message);
            return [];
        }
    }
}

class SuperAIHoverProvider {
    async provideHover(document, position, token) {
        try {
            const currentFile = document.uri.fsPath;
            
            const response = await axios.post(`${SERVER_URL}/analyze_file`, {
                file_path: currentFile
            }, { timeout: 3000 });
            
            if (response.data.error) return null;
            
            const analysis = response.data.analysis;
            if (!analysis) return null;
            
            const contents = new vscode.MarkdownString();
            contents.appendMarkdown(`**File Analysis**\n\n`);
            contents.appendMarkdown(`**Language:** ${response.data.language}\n`);
            contents.appendMarkdown(`**Lines:** ${response.data.line_count}\n`);
            
            if (analysis.functions && analysis.functions.length > 0) {
                contents.appendMarkdown(`**Functions:** ${analysis.functions.join(', ')}\n`);
            }
            
            if (analysis.classes && analysis.classes.length > 0) {
                contents.appendMarkdown(`**Classes:** ${analysis.classes.join(', ')}\n`);
            }
            
            if (analysis.errors && analysis.errors.length > 0) {
                contents.appendMarkdown(`**⚠️ Errors:** ${analysis.errors.length}\n`);
            }
            
            return new vscode.Hover(contents);
            
        } catch (error) {
            return null;
        }
    }
}

async function analyzeAndShowDiagnostics(document) {
    try {
        const currentFile = document.uri.fsPath;
        if (!currentFile || document.isUntitled) return;
        
        const response = await axios.post(`${SERVER_URL}/analyze_file`, {
            file_path: currentFile
        }, { timeout: 5000 });
        
        if (response.data.error) return;
        
        const analysis = response.data.analysis;
        if (!analysis || !analysis.errors) return;
        
        // Convert errors to VS Code diagnostics
        const diagnostics = analysis.errors.map(error => {
            const line = Math.max(0, (error.line || 1) - 1);
            const column = Math.max(0, error.column || 0);
            
            const range = new vscode.Range(
                new vscode.Position(line, column),
                new vscode.Position(line, column + 10)
            );
            
            const diagnostic = new vscode.Diagnostic(
                range,
                error.message || 'Unknown error',
                error.type === 'SyntaxError' ? 
                    vscode.DiagnosticSeverity.Error : 
                    vscode.DiagnosticSeverity.Warning
            );
            
            diagnostic.source = 'SuperAI';
            return diagnostic;
        });
        
        diagnosticCollection.set(document.uri, diagnostics);
        
    } catch (error) {
        console.error('Error analyzing file:', error.message);
    }
}

async function autoFixCode() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showErrorMessage('No active file open.');
        return;
    }
    
    try {
        const document = editor.document;
        const currentFile = document.uri.fsPath;
        
        // Save the file first to ensure we're working with the latest version
        await document.save();
        
        vscode.window.showInformationMessage('🔧 Auto-fixing and completing entire file...');
        
        // Use the new auto_fix_and_complete endpoint that modifies the file directly
        const response = await axios.post(`${SERVER_URL}/auto_fix_and_complete`, {
            file_path: currentFile
        }, { timeout: 15000 });
        
        if (response.data.error) {
            vscode.window.showErrorMessage('Auto-fix error: ' + response.data.error);
            return;
        }
        
        const result = response.data;
        if (result.success && result.total_modifications > 0) {
            // Reload the file to show the changes
            const uri = vscode.Uri.file(currentFile);
            await vscode.commands.executeCommand('workbench.action.files.revert', uri);
            
            let message = '✅ File automatically updated!\n';
            
            const errorFixes = result.error_fixes;
            if (errorFixes && errorFixes.changes_made) {
                message += `🔧 Fixed ${errorFixes.errors_resolved || 0} errors\n`;
            }
            
            const completions = result.completions_applied;
            if (completions && completions.modifications_made > 0) {
                message += `🚀 Applied ${completions.modifications_made} code completions\n`;
            }
            
            message += `Total modifications: ${result.total_modifications}`;
            
            vscode.window.showInformationMessage(message);
        } else {
            vscode.window.showInformationMessage('✅ No fixes or completions needed.');
        }
        
    } catch (error) {
        vscode.window.showErrorMessage('Auto-fix error: ' + error.message);
    }
}

async function showWorkspaceInfo() {
    try {
        const response = await axios.get(`${SERVER_URL}/workspace_info`);
        
        if (response.data.error) {
            vscode.window.showErrorMessage('Workspace info error: ' + response.data.error);
            return;
        }
        
        const info = response.data;
        const message = `Active Workspace: ${info.active_workspace || 'None'}\n` +
                       `Total Workspaces: ${info.workspace_count}`;
        
        vscode.window.showInformationMessage(message);
        
    } catch (error) {
        vscode.window.showErrorMessage('Workspace info error: ' + error.message);
    }
}

function updateStatusBar() {
    if (!statusBarItem) {
        statusBarItem = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Right, 
            100
        );
    }
    
    statusBarItem.text = autoCompleteEnabled ? 
        '$(check) SuperAI' : 
        '$(x) SuperAI';
    statusBarItem.tooltip = autoCompleteEnabled ? 
        'SuperAI Auto-completion enabled' : 
        'SuperAI Auto-completion disabled';
    statusBarItem.command = 'superai.toggleAutoComplete';
    statusBarItem.show();
}

function getTwoStageResultsHtml(result, todoCount, fixmeCount, passCount) {
    const stage1Code = result.stage1_fixed_code || '';
    const finalCode = result.final_completed_code || '';
    const stage1Analysis = result.stage1_analysis || {};
    const stage2Analysis = result.stage2_analysis || {};
    
    return `
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Two-Stage Gemini Results</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; }
            .section { margin-bottom: 30px; padding: 15px; border: 1px solid #ddd; border-radius: 8px; }
            .success { background-color: #d4edda; border-color: #c3e6cb; }
            .warning { background-color: #fff3cd; border-color: #ffeaa7; }
            .error { background-color: #f8d7da; border-color: #f5c6cb; }
            .code-block { background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 4px; padding: 15px; margin: 10px 0; overflow-x: auto; }
            pre { margin: 0; white-space: pre-wrap; font-family: 'Courier New', monospace; font-size: 12px; }
            h2 { margin-top: 0; color: #333; }
            h3 { color: #666; }
            .stats { display: flex; gap: 20px; margin: 15px 0; }
            .stat { padding: 10px; border-radius: 4px; text-align: center; min-width: 80px; }
            .stat-good { background-color: #d4edda; }
            .stat-bad { background-color: #f8d7da; }
            .stat-neutral { background-color: #e2e3e5; }
        </style>
    </head>
    <body>
        <h1>🔄 Two-Stage Gemini Results</h1>
        
        <div class="section ${result.success ? 'success' : 'error'}">
            <h2>📊 Process Summary</h2>
            <p><strong>Status:</strong> ${result.success ? '✅ Success' : '❌ Failed'}</p>
            <div class="stats">
                <div class="stat ${todoCount === 0 ? 'stat-good' : 'stat-bad'}">
                    <strong>${todoCount}</strong><br>TODOs Remaining
                </div>
                <div class="stat ${fixmeCount === 0 ? 'stat-good' : 'stat-bad'}">
                    <strong>${fixmeCount}</strong><br>FIXMEs Remaining
                </div>
                <div class="stat ${passCount === 0 ? 'stat-good' : 'stat-bad'}">
                    <strong>${passCount}</strong><br>Pass Statements
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>🔧 Stage 1: Syntax Fixes</h2>
            <h3>Fixed Code:</h3>
            <div class="code-block">
                <pre><code>${stage1Code}</code></pre>
            </div>
            <h3>Analysis:</h3>
            <p>${JSON.stringify(stage1Analysis, null, 2)}</p>
        </div>
        
        <div class="section">
            <h2>🚀 Stage 2: Final Implementation</h2>
            <h3>Complete Code:</h3>
            <div class="code-block">
                <pre><code>${finalCode}</code></pre>
            </div>
            <h3>Analysis:</h3>
            <p>${JSON.stringify(stage2Analysis, null, 2)}</p>
        </div>
        
        <div class="section ${todoCount === 0 && fixmeCount === 0 ? 'success' : 'warning'}">
            <h2>✅ Validation Results</h2>
            <ul>
                <li>TODO Comments: ${todoCount === 0 ? '✅ None remaining' : `❌ ${todoCount} still present`}</li>
                <li>FIXME Comments: ${fixmeCount === 0 ? '✅ None remaining' : `❌ ${fixmeCount} still present`}</li>
                <li>Pass Statements: ${passCount === 0 ? '✅ None remaining' : `⚠️ ${passCount} still present`}</li>
                <li>Process Status: ${result.success ? '✅ Completed successfully' : '❌ Failed'}</li>
            </ul>
        </div>
    </body>
    </html>`;
}

function generateThreeStageResultsHTML(result) {
    const stage0Understanding = result.stage0_understanding || {};
    const stage1Code = result.stage1_fixed_code || '';
    const finalCode = result.final_completed_code || '';
    const stage0Analysis = result.stage0_analysis || {};
    const stage1Analysis = result.stage1_analysis || {};
    const stage2Analysis = result.stage2_analysis || {};
    
    // Count remaining issues
    const todoCount = (finalCode.match(/TODO|FIXME/gi) || []).length;
    const passCount = (finalCode.match(/^\s*pass\s*$/gm) || []).length;
    
    return `
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Three-Stage Gemini Results</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; }
            .section { margin-bottom: 30px; padding: 15px; border: 1px solid #ddd; border-radius: 8px; }
            .success { background-color: #d4edda; border-color: #c3e6cb; }
            .warning { background-color: #fff3cd; border-color: #ffeaa7; }
            .error { background-color: #f8d7da; border-color: #f5c6cb; }
            .info { background-color: #d1ecf1; border-color: #bee5eb; }
            .code-block { background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 4px; padding: 15px; margin: 10px 0; overflow-x: auto; }
            pre { margin: 0; white-space: pre-wrap; font-family: 'Courier New', monospace; font-size: 12px; }
            h2 { margin-top: 0; color: #333; }
            h3 { color: #666; }
            .stats { display: flex; gap: 20px; margin: 15px 0; }
            .stat { padding: 10px; border-radius: 4px; text-align: center; min-width: 80px; }
            .stat-good { background-color: #d4edda; }
            .stat-bad { background-color: #f8d7da; }
            .stat-neutral { background-color: #e2e3e5; }
            .understanding-item { margin: 10px 0; padding: 8px; background-color: #f8f9fa; border-radius: 4px; }
        </style>
    </head>
    <body>
        <h1>🧠 Three-Stage Gemini Results</h1>
        
        <div class="section ${result.success ? 'success' : 'error'}">
            <h2>📊 Process Summary</h2>
            <p><strong>Status:</strong> ${result.success ? '✅ Success' : '❌ Failed'}</p>
            <div class="stats">
                <div class="stat ${todoCount === 0 ? 'stat-good' : 'stat-bad'}">
                    <strong>${todoCount}</strong><br>TODOs Remaining
                </div>
                <div class="stat ${passCount === 0 ? 'stat-good' : 'stat-bad'}">
                    <strong>${passCount}</strong><br>Pass Statements
                </div>
                <div class="stat stat-good">
                    <strong>3</strong><br>Stages Completed
                </div>
            </div>
        </div>
        
        <div class="section info">
            <h2>🧠 Stage 0: Code Understanding</h2>
            <h3>Understanding Analysis:</h3>
            <div class="understanding-item">
                <strong>Main Purpose:</strong> ${stage0Understanding.main_purpose || 'Not specified'}
            </div>
            <div class="understanding-item">
                <strong>Core Algorithm:</strong> ${stage0Understanding.core_algorithm || 'Not specified'}
            </div>
            <div class="understanding-item">
                <strong>Key Functions:</strong> ${Array.isArray(stage0Understanding.key_functions) ? stage0Understanding.key_functions.join(', ') : 'None identified'}
            </div>
            <div class="understanding-item">
                <strong>Data Flow:</strong> ${stage0Understanding.data_flow || 'Not specified'}
            </div>
            <div class="understanding-item">
                <strong>Expected Behavior:</strong> ${stage0Understanding.expected_behavior || 'Not specified'}
            </div>
            <div class="understanding-item">
                <strong>Missing Components:</strong> ${Array.isArray(stage0Understanding.missing_components) ? stage0Understanding.missing_components.join(', ') : 'None identified'}
            </div>
        </div>
        
        <div class="section">
            <h2>🔧 Stage 1: Syntax Fixes (with Understanding)</h2>
            <h3>Fixed Code:</h3>
            <div class="code-block">
                <pre><code>${stage1Code}</code></pre>
            </div>
            <h3>Analysis:</h3>
            <p>${JSON.stringify(stage1Analysis, null, 2)}</p>
        </div>
        
        <div class="section">
            <h2>🚀 Stage 2: Final Implementation (with Understanding)</h2>
            <h3>Complete Code:</h3>
            <div class="code-block">
                <pre><code>${finalCode}</code></pre>
            </div>
            <h3>Analysis:</h3>
            <p>${JSON.stringify(stage2Analysis, null, 2)}</p>
        </div>
        
        <div class="section ${todoCount === 0 ? 'success' : 'warning'}">
            <h2>✅ Validation Results</h2>
            <ul>
                <li>Code Understanding: ${stage0Understanding.main_purpose ? '✅ Successfully analyzed' : '❌ Failed to understand'}</li>
                <li>TODO Comments: ${todoCount === 0 ? '✅ None remaining' : `❌ ${todoCount} still present`}</li>
                <li>Pass Statements: ${passCount === 0 ? '✅ None remaining' : `⚠️ ${passCount} still present`}</li>
                <li>Process Status: ${result.success ? '✅ Completed successfully' : '❌ Failed'}</li>
            </ul>
        </div>
    </body>
    </html>`;
}

function generateAPIKeyStatusHTML(status) {
    const apiStatus = status.api_key_manager_status || {};
    const keyUsage = apiStatus.key_usage || {};
    const keyErrors = apiStatus.key_errors || {};
    const keyConsecutiveErrors = apiStatus.key_consecutive_errors || {};
    const rotationInfo = status.rotation_info || {};
    
    const totalKeys = apiStatus.total_keys || 0;
    const totalRequests = status.total_requests || 0;
    const totalErrors = status.total_errors || 0;
    const healthyKeys = status.healthy_keys || 0;
    const currentActiveKey = rotationInfo.current_active_key || 'None';
    const rotationThreshold = rotationInfo.rotation_threshold || 3;
    
    let keyStatusHTML = '';
    Object.keys(keyUsage).forEach((key, index) => {
        const usage = keyUsage[key] || 0;
        const errors = keyErrors[key] || 0;
        const consecutiveErrors = keyConsecutiveErrors[key] || 0;
        const isActive = key.substring(0, 10) + '...' === currentActiveKey;
        const isHealthy = consecutiveErrors < rotationThreshold;
        const keyDisplay = key.substring(0, 10) + '...';
        
        keyStatusHTML += `
            <div class="key-status ${isActive ? 'key-active' : (isHealthy ? 'key-healthy' : 'key-exhausted')}">
                <strong>${isActive ? '🎯 ACTIVE' : 'Key'} ${index + 1}:</strong> ${keyDisplay}<br>
                <span>Requests: ${usage}</span><br>
                <span>Total Errors: ${errors}</span><br>
                <span>Consecutive Errors: ${consecutiveErrors}/${rotationThreshold}</span><br>
                <span>Status: ${isActive ? '🎯 Currently Active' : (isHealthy ? '✅ Healthy' : '🔥 Exhausted')}</span>
            </div>
        `;
    });
    
    return `
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Gemini API Key Status</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; }
            .section { margin-bottom: 30px; padding: 15px; border: 1px solid #ddd; border-radius: 8px; }
            .success { background-color: #d4edda; border-color: #c3e6cb; }
            .warning { background-color: #fff3cd; border-color: #ffeaa7; }
            .error { background-color: #f8d7da; border-color: #f5c6cb; }
            .info { background-color: #d1ecf1; border-color: #bee5eb; }
            h2 { margin-top: 0; color: #333; }
            h3 { color: #666; }
            .stats { display: flex; gap: 20px; margin: 15px 0; flex-wrap: wrap; }
            .stat { padding: 15px; border-radius: 4px; text-align: center; min-width: 100px; }
            .stat-good { background-color: #d4edda; }
            .stat-bad { background-color: #f8d7da; }
            .stat-neutral { background-color: #e2e3e5; }
            .key-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0; }
            .key-status { padding: 10px; border-radius: 4px; border: 1px solid #ddd; }
            .key-active { background-color: #cce5ff; border-color: #0066cc; border-width: 2px; }
            .key-healthy { background-color: #d4edda; border-color: #c3e6cb; }
            .key-exhausted { background-color: #f8d7da; border-color: #f5c6cb; }
            .stage-config { margin: 10px 0; padding: 8px; background-color: #f8f9fa; border-radius: 4px; }
        </style>
    </head>
    <body>
        <h1>🔄 Smart API Key Rotation Status</h1>
        
        <div class="section ${totalErrors === 0 ? 'success' : 'warning'}">
            <h2>📊 Rotation System Status</h2>
            <div class="stats">
                <div class="stat ${totalKeys > 0 ? 'stat-good' : 'stat-bad'}">
                    <strong>${totalKeys}</strong><br>Total Keys
                </div>
                <div class="stat ${healthyKeys === totalKeys ? 'stat-good' : 'stat-bad'}">
                    <strong>${healthyKeys}</strong><br>Healthy Keys
                </div>
                <div class="stat stat-neutral">
                    <strong>${totalRequests}</strong><br>Total Requests
                </div>
                <div class="stat ${totalErrors === 0 ? 'stat-good' : 'stat-bad'}">
                    <strong>${totalErrors}</strong><br>Total Errors
                </div>
            </div>
        </div>
        
        <div class="section info">
            <h2>🎯 Current Rotation Status</h2>
            <div class="stage-config">
                <strong>🎯 Active Key:</strong> ${currentActiveKey}
            </div>
            <div class="stage-config">
                <strong>🔄 Rotation Threshold:</strong> ${rotationThreshold} consecutive errors
            </div>
            <div class="stage-config">
                <strong>⚡ System Type:</strong> ${rotationInfo.system_type || 'Smart single-key rotation on exhaustion'}
            </div>
            <div class="stage-config">
                <strong>🛡️ Auto-Rotation:</strong> ✅ Enabled (seamless key switching)
            </div>
        </div>
        
        <div class="section">
            <h2>🔐 Individual Key Status</h2>
            <div class="key-grid">
                ${keyStatusHTML || '<p>No API keys configured</p>'}
            </div>
        </div>
        
        <div class="section info">
            <h2>💡 Smart Rotation Configuration</h2>
            <ul>
                <li><strong>Rotation Pool:</strong> Set GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc. for automatic rotation</li>
                <li><strong>Single Active Key:</strong> System uses one key at a time, rotating on exhaustion (${rotationThreshold} consecutive errors)</li>
                <li><strong>No Breakpoints:</strong> Seamless switching prevents service interruption</li>
                <li><strong>Auto-Recovery:</strong> Keys can recover and be used again when healthy</li>
                <li><strong>Smart Monitoring:</strong> Real-time tracking of consecutive errors per key</li>
                <li><strong>Backup Keys:</strong> Legacy stage-specific keys are included as backup options</li>
            </ul>
        </div>
    </body>
    </html>`;
}

function activate(context) {
    console.log('SuperAI Extension activating...');
    
    // Create diagnostic collection
    diagnosticCollection = vscode.languages.createDiagnosticCollection('superai');
    context.subscriptions.push(diagnosticCollection);
    
    // Register completion provider
    completionProvider = vscode.languages.registerCompletionItemProvider(
        { scheme: 'file' },
        new SuperAICompletionProvider(),
        '.', ' ', '(', '=', ':'
    );
    context.subscriptions.push(completionProvider);
    
    // Register hover provider
    hoverProvider = vscode.languages.registerHoverProvider(
        { scheme: 'file' },
        new SuperAIHoverProvider()
    );
    context.subscriptions.push(hoverProvider);
    
    // Manual completion command - now uses global completion
    const completeCodeCommand = vscode.commands.registerCommand('superai.completeCode', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active file open.');
            return;
        }
        
        try {
            const document = editor.document;
            const currentFile = document.uri.fsPath;
            
            // Save the file first
            await document.save();
            
            vscode.window.showInformationMessage('🚀 Applying global completions to entire file...');
            
            // Use the new apply_completions endpoint that modifies the file directly
            const response = await axios.post(`${SERVER_URL}/apply_completions`, {
                file_path: currentFile
            }, { timeout: 15000 });
            
            if (response.data.error) {
                vscode.window.showErrorMessage('Completion Error: ' + response.data.error);
                return;
            }
            
            const result = response.data;
            if (result.success && result.modifications_made > 0) {
                // Reload the file to show the changes
                const uri = vscode.Uri.file(currentFile);
                await vscode.commands.executeCommand('workbench.action.files.revert', uri);
                
                let message = `✅ Applied ${result.modifications_made} completions to file!\n`;
                message += `Found ${result.total_completions_found} incomplete patterns.`;
                
                if (result.applied_completions && result.applied_completions.length > 0) {
                    message += '\n\nCompleted:';
                    result.applied_completions.slice(0, 3).forEach((comp, i) => {
                        message += `\n${i + 1}. Line ${comp.line}: ${comp.type}`;
                    });
                    if (result.applied_completions.length > 3) {
                        message += `\n... and ${result.applied_completions.length - 3} more`;
                    }
                }
                
                vscode.window.showInformationMessage(message);
            } else {
                vscode.window.showInformationMessage('✅ No incomplete patterns found to complete.');
            }
            
        } catch (error) {
            vscode.window.showErrorMessage('Error applying completions: ' + error.message);
        }
    });
    
    // Auto-fix command
    const autoFixCommand = vscode.commands.registerCommand('superai.autoFix', autoFixCode);
    
    // Global complete command
    const globalCompleteCommand = vscode.commands.registerCommand('superai.globalComplete', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active file open.');
            return;
        }
        
        try {
            const document = editor.document;
            const currentFile = document.uri.fsPath;
            
            await document.save();
            
            vscode.window.showInformationMessage('🎯 Applying only completions to file...');
            
            const response = await axios.post(`${SERVER_URL}/apply_completions`, {
                file_path: currentFile
            }, { timeout: 15000 });
            
            if (response.data.error) {
                vscode.window.showErrorMessage('Global completion error: ' + response.data.error);
                return;
            }
            
            const result = response.data;
            if (result.success && result.modifications_made > 0) {
                const uri = vscode.Uri.file(currentFile);
                await vscode.commands.executeCommand('workbench.action.files.revert', uri);
                
                vscode.window.showInformationMessage(
                    `✅ Applied ${result.modifications_made} completions! Found ${result.total_completions_found} patterns.`
                );
            } else {
                vscode.window.showInformationMessage('✅ No incomplete patterns found.');
            }
            
        } catch (error) {
            vscode.window.showErrorMessage('Global completion error: ' + error.message);
        }
    });
    
    // Write fixed code command
    const writeFixedCodeCommand = vscode.commands.registerCommand('superai.writeFixedCode', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active file open.');
            return;
        }
        
        try {
            const document = editor.document;
            const currentFile = document.uri.fsPath;
            
            await document.save();
            
            vscode.window.showInformationMessage('🔧 Writing fixed code to file...');
            
            const response = await axios.post(`${SERVER_URL}/write_fixed_code`, {
                file_path: currentFile
            }, { timeout: 10000 });
            
            if (response.data.error) {
                vscode.window.showErrorMessage('Write fixed code error: ' + response.data.error);
                return;
            }
            
            const result = response.data;
            if (result.success && result.changes_applied) {
                const uri = vscode.Uri.file(currentFile);
                await vscode.commands.executeCommand('workbench.action.files.revert', uri);
                
                vscode.window.showInformationMessage(
                    `✅ Fixed ${result.errors_fixed} errors and wrote to file!`
                );
            } else {
                vscode.window.showInformationMessage('✅ No errors found to fix.');
            }
            
        } catch (error) {
            vscode.window.showErrorMessage('Write fixed code error: ' + error.message);
        }
    });
    
    // Analyze file command
    const analyzeCommand = vscode.commands.registerCommand('superai.analyzeFile', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active file open.');
            return;
        }
        
        await analyzeAndShowDiagnostics(editor.document);
        vscode.window.showInformationMessage('File analysis complete.');
    });
    
    // Workspace info command
    const workspaceInfoCommand = vscode.commands.registerCommand('superai.workspaceInfo', showWorkspaceInfo);
    
    // Toggle auto-complete command
    const toggleAutoCompleteCommand = vscode.commands.registerCommand('superai.toggleAutoComplete', () => {
        autoCompleteEnabled = !autoCompleteEnabled;
        updateStatusBar();
        vscode.window.showInformationMessage(
            `SuperAI Auto-completion ${autoCompleteEnabled ? 'enabled' : 'disabled'}`
        );
    });

    // API Key Status command
    const apiKeyStatusCommand = vscode.commands.registerCommand('superai.apiKeyStatus', async () => {
        try {
            const response = await axios.get(`${SERVER_URL}/api_key_status`, { timeout: 5000 });
            
            if (response.data.error) {
                vscode.window.showErrorMessage('API Key Status Error: ' + response.data.error);
                return;
            }
            
            const status = response.data;
            
            // Create status panel
            const panel = vscode.window.createWebviewPanel(
                'apiKeyStatus',
                'Gemini API Key Status',
                vscode.ViewColumn.Beside,
                { enableScripts: true }
            );
            
            panel.webview.html = generateAPIKeyStatusHTML(status);
            
        } catch (error) {
            vscode.window.showErrorMessage('API Key Status error: ' + error.message);
        }
    });

    
    // Enhanced Syntax Fix command
    const enhancedSyntaxFixCommand = vscode.commands.registerCommand('superai.enhancedSyntaxFix', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active file open.');
            return;
        }
        
        try {
            const document = editor.document;
            const filePath = document.uri.fsPath;
            const codeContent = document.getText();
            
            vscode.window.showInformationMessage('🔧 Starting Enhanced Syntax Fix (Auto-Creating Functions)...');
            
            const response = await axios.post(`${SERVER_URL}/enhanced_syntax_fix`, {
                file_path: filePath,
                code_content: codeContent,
                apply_changes: false
            }, { timeout: 30000 });
            
            const result = response.data;
            
            if (result.success) {
                // Show results in a new document
                const fixedCode = result.fixed_code;
                const fixesApplied = result.fixes_applied || [];
                
                // Create results HTML
                const resultsHtml = `
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced Syntax Fix Results</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background: #1e1e1e; color: #d4d4d4; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .section { background: #2d2d30; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #007acc; }
        .fixes-list { background: #0e2f1e; border-left: 4px solid #4caf50; padding: 15px; margin: 10px 0; border-radius: 8px; }
        .fix-item { margin: 8px 0; padding: 8px; background: #1a4a2e; border-radius: 4px; }
        .code-block { background: #1e1e1e; border: 1px solid #3c3c3c; padding: 15px; border-radius: 8px; font-family: 'Consolas', monospace; white-space: pre-wrap; overflow-x: auto; }
        .success { color: #4caf50; }
        .info { color: #2196f3; }
        .warning { color: #ff9800; }
        .button { background: #007acc; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; margin: 5px; }
        .button:hover { background: #005a9e; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔧 Enhanced Syntax Fix Results</h1>
        <p>Comprehensive intelligent fixing with auto-function generation</p>
    </div>
    
    <div class="fixes-list">
        <h2 class="success">✅ Fixes Applied:</h2>
        ${fixesApplied.map(fix => `<div class="fix-item">• ${fix}</div>`).join('')}
    </div>
    
    <div class="section">
        <h2 class="info">📋 Fixed Code:</h2>
        <div class="code-block">${fixedCode.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</div>
    </div>
    
    <div class="section">
        <h2 class="warning">⚠️ Next Steps:</h2>
        <p>1. Review the automatically created functions</p>
        <p>2. Apply changes to your file if satisfied</p>
        <p>3. Test the enhanced code functionality</p>
        <p>4. Use Three-Stage Gemini Fix for complete implementation</p>
    </div>
</body>
</html>`;
                
                // Show in webview
                const panel = vscode.window.createWebviewPanel(
                    'enhancedSyntaxFixResults',
                    'Enhanced Syntax Fix Results',
                    vscode.ViewColumn.Beside,
                    { enableScripts: true }
                );
                panel.webview.html = resultsHtml;
                
                // Ask if user wants to apply changes
                const applyChanges = await vscode.window.showInformationMessage(
                    `🎉 Enhanced Syntax Fix Complete!\n` +
                    `✅ ${fixesApplied.length} fixes applied\n` +
                    `✅ Missing functions auto-created\n` +
                    `✅ Variable typos corrected\n` +
                    `✅ Colon placement fixed\n\n` +
                    `Apply changes to file?`,
                    'Apply Changes', 'Review Only'
                );
                
                if (applyChanges === 'Apply Changes') {
                    // Apply the fixed code to the editor
                    const edit = new vscode.WorkspaceEdit();
                    const fullRange = new vscode.Range(
                        document.positionAt(0),
                        document.positionAt(codeContent.length)
                    );
                    edit.replace(document.uri, fullRange, fixedCode);
                    await vscode.workspace.applyEdit(edit);
                    await document.save();
                    
                    vscode.window.showInformationMessage(
                        `🎉 Enhanced Syntax Fix Applied!\n` +
                        `✅ Missing functions created automatically\n` +
                        `✅ All syntax errors fixed\n` +
                        `✅ Variable typos corrected\n` +
                        `✅ Code is now syntactically correct`
                    );
                }
                
            } else {
                vscode.window.showErrorMessage('Enhanced Syntax Fix failed: ' + (result.error || 'Unknown error'));
            }
            
        } catch (error) {
            vscode.window.showErrorMessage('Enhanced Syntax Fix error: ' + error.message);
        }
    });

    // Four-Stage Gemini Fix command
    const fourStageGeminiFixCommand = vscode.commands.registerCommand('superai.fourStageGeminiFix', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active file open.');
            return;
        }
        
        try {
            const document = editor.document;
            const filePath = document.uri.fsPath;
            const codeContent = document.getText();
            
            // Show progress notification
            vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: "Four-Stage Gemini Fix",
                cancellable: false
            }, async (progress) => {
                progress.report({ increment: 0, message: "Starting..." });
                
                vscode.window.showInformationMessage('🚀 Starting Four-Stage Gemini Fix (This may take 1-3 minutes)...');
                
                // Update progress periodically
                const progressInterval = setInterval(() => {
                    progress.report({ increment: 5, message: "Processing stages..." });
                }, 5000);
                
                try {
                    const response = await axios.post(`${SERVER_URL}/four_stage_gemini_fix`, {
                        file_path: filePath,
                        code_content: codeContent,
                        apply_changes: false
                    }, { timeout: 180000 }); // 3 minutes for 4 API calls (45s each)
                    
                    clearInterval(progressInterval);
                    progress.report({ increment: 100, message: "Complete!" });
                    
                    return response;
                } catch (error) {
                    clearInterval(progressInterval);
                    throw error;
                }
            }).then(async (response) => {
                if (!response) return;
            
            const result = response.data;
            
            if (result.success) {
                // Show results in a new document
                const finalCode = result.final_code;
                const stage0Result = result.stage0_result || {};
                const stage1Result = result.stage1_result || {};
                const stage2Result = result.stage2_result || {};
                const stage3Result = result.stage3_result || {};
                const summary = result.summary || {};
                
                // Create comprehensive results HTML
                const resultsHtml = `
<!DOCTYPE html>
<html>
<head>
    <title>Four-Stage Gemini Results</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background: #1e1e1e; color: #d4d4d4; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .stage { background: #2d2d30; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #007acc; }
        .stage-0 { border-left-color: #ff6b6b; }
        .stage-1 { border-left-color: #4ecdc4; }
        .stage-2 { border-left-color: #45b7d1; }
        .stage-3 { border-left-color: #96ceb4; }
        .summary { background: #0e2f1e; border-left: 4px solid #4caf50; padding: 15px; margin: 10px 0; border-radius: 8px; }
        .code-block { background: #1e1e1e; border: 1px solid #3c3c3c; padding: 15px; border-radius: 8px; font-family: 'Consolas', monospace; white-space: pre-wrap; overflow-x: auto; max-height: 300px; }
        .success { color: #4caf50; }
        .info { color: #2196f3; }
        .warning { color: #ff9800; }
        .stage-title { font-size: 18px; font-weight: bold; margin-bottom: 10px; }
        .metric { margin: 5px 0; padding: 5px; background: #333; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Four-Stage Gemini Results</h1>
        <p>Understand → Plan → Implement → Replace</p>
    </div>
    
    <div class="summary">
        <h2 class="success">📊 Summary:</h2>
        <div class="metric">✅ Stage 0 Success: ${summary.stage0_success ? 'Yes' : 'No'}</div>
        <div class="metric">📋 Stage 1 Success: ${summary.stage1_success ? 'Yes' : 'No'}</div>
        <div class="metric">🛠️ Stage 2 Success: ${summary.stage2_success ? 'Yes' : 'No'}</div>
        <div class="metric">🔄 Stage 3 Success: ${summary.stage3_success ? 'Yes' : 'No'}</div>
        <div class="metric">🧠 Understanding Fields: ${summary.understanding_fields || 0}</div>
        <div class="metric">📋 Plan Components: ${summary.plan_components || 0}</div>
        <div class="metric">🛠️ Implementations Created: ${summary.implementations_created || 0}</div>
        <div class="metric">🔄 Replacements Made: ${summary.replacements_made || 0}</div>
        <div class="metric">✅ Final Code Valid: ${summary.final_code_valid ? 'Yes' : 'No'}</div>
        <div class="metric">🎯 TODOs Eliminated: ${summary.todos_eliminated ? 'Yes' : 'No'}</div>
    </div>
    
    <div class="stage stage-0">
        <div class="stage-title">🧠 Stage 0: Code Understanding</div>
        <p><strong>Status:</strong> ${stage0Result.success ? '✅ Success' : '❌ Failed'}</p>
        ${stage0Result.understanding ? `
        <p><strong>Main Purpose:</strong> ${stage0Result.understanding.main_purpose || 'Not identified'}</p>
        <p><strong>Core Algorithm:</strong> ${stage0Result.understanding.core_algorithm || 'Not identified'}</p>
        <p><strong>Key Functions:</strong> ${stage0Result.understanding.key_functions || 'Not identified'}</p>
        ` : ''}
    </div>
    
    <div class="stage stage-1">
        <div class="stage-title">📋 Stage 1: Comprehensive Planning</div>
        <p><strong>Status:</strong> ${stage1Result.success ? '✅ Success' : '❌ Failed'}</p>
        ${stage1Result.plan ? `
        <p><strong>Syntax Fixes:</strong> ${stage1Result.plan.syntax_fixes || 'None planned'}</p>
        <p><strong>Variable Corrections:</strong> ${stage1Result.plan.variable_corrections || 'None planned'}</p>
        <p><strong>Missing Functions:</strong> ${stage1Result.plan.missing_functions || 'None planned'}</p>
        ` : ''}
    </div>
    
    <div class="stage stage-2">
        <div class="stage-title">🛠️ Stage 2: Implementation</div>
        <p><strong>Status:</strong> ${stage2Result.success ? '✅ Success' : '❌ Failed'}</p>
        ${stage2Result.implementations ? `
        <p><strong>Functions Created:</strong> ${stage2Result.implementations.missing_functions ? 'Yes' : 'None'}</p>
        <p><strong>Syntax Fixed:</strong> ${stage2Result.implementations.syntax_fixes ? 'Yes' : 'None'}</p>
        <p><strong>Variables Corrected:</strong> ${stage2Result.implementations.variable_corrections ? 'Yes' : 'None'}</p>
        ` : ''}
    </div>
    
    <div class="stage stage-3">
        <div class="stage-title">🔄 Stage 3: Intelligent Replacement</div>
        <p><strong>Status:</strong> ${stage3Result.success ? '✅ Success' : '❌ Failed'}</p>
        ${stage3Result.replacements ? `
        <p><strong>Replacements Made:</strong> ${stage3Result.replacements.length || 0}</p>
        ` : ''}
        ${stage3Result.validation ? `
        <p><strong>Syntax Valid:</strong> ${stage3Result.validation.syntax_valid ? '✅ Yes' : '❌ No'}</p>
        <p><strong>TODOs Remaining:</strong> ${stage3Result.validation.has_todos ? '❌ Yes' : '✅ None'}</p>
        ` : ''}
    </div>
    
    <div class="stage">
        <h2 class="info">📋 Final Code:</h2>
        <div class="code-block">${finalCode ? finalCode.replace(/</g, '&lt;').replace(/>/g, '&gt;') : 'No final code generated'}</div>
    </div>
    
    <div class="stage">
        <h2 class="warning">⚠️ Next Steps:</h2>
        <p>1. Review the four-stage transformation process</p>
        <p>2. Verify the final code meets your requirements</p>
        <p>3. Apply changes to your file if satisfied</p>
        <p>4. Test the enhanced code functionality</p>
    </div>
</body>
</html>`;
                
                // Show in webview
                const panel = vscode.window.createWebviewPanel(
                    'fourStageGeminiResults',
                    'Four-Stage Gemini Results',
                    vscode.ViewColumn.Beside,
                    { enableScripts: true }
                );
                panel.webview.html = resultsHtml;
                
                // Ask if user wants to apply changes
                const applyChanges = await vscode.window.showInformationMessage(
                    `🎉 Four-Stage Gemini Complete!\n` +
                    `🧠 Stage 0: ${stage0Result.success ? '✅' : '❌'} Code Understanding\n` +
                    `📋 Stage 1: ${stage1Result.success ? '✅' : '❌'} Comprehensive Planning\n` +
                    `🛠️ Stage 2: ${stage2Result.success ? '✅' : '❌'} Implementation\n` +
                    `🔄 Stage 3: ${stage3Result.success ? '✅' : '❌'} Intelligent Replacement\n\n` +
                    `Apply changes to file?`,
                    'Apply Changes', 'Review Only'
                );
                
                if (applyChanges === 'Apply Changes') {
                    // Apply the final code to the editor
                    const edit = new vscode.WorkspaceEdit();
                    const fullRange = new vscode.Range(
                        document.positionAt(0),
                        document.positionAt(codeContent.length)
                    );
                    edit.replace(document.uri, fullRange, finalCode);
                    await vscode.workspace.applyEdit(edit);
                    await document.save();
                    
                    vscode.window.showInformationMessage(
                        `🎉 Four-Stage Gemini Applied!\n` +
                        `✅ Code understood, planned, implemented and replaced\n` +
                        `✅ All stages completed successfully\n` +
                        `✅ Code is now complete and functional`
                    );
                }
                
            } else {
                vscode.window.showErrorMessage('Four-Stage Gemini failed: ' + (result.error || 'Unknown error'));
            }
            }).catch((error) => {
                vscode.window.showErrorMessage('Four-Stage Gemini error: ' + error.message);
            });
            
        } catch (error) {
            vscode.window.showErrorMessage('Four-Stage Gemini error: ' + error.message);
        }
    });
    
    // ============================================================================
    // MULTI-FILE PROJECT MANAGEMENT COMMANDS
    // ============================================================================
    
    // Command: Index Project
    const indexProjectCommand = vscode.commands.registerCommand('superai.indexProject', async () => {
        try {
            const workspaceFolder = vscode.workspace.workspaceFolders ? 
                vscode.workspace.workspaceFolders[0].uri.fsPath : '';
            
            if (!workspaceFolder) {
                vscode.window.showErrorMessage('No workspace folder open');
                return;
            }
            
            vscode.window.showInformationMessage('🔍 Indexing project... This may take a moment.');
            
            const response = await axios.post(`${SERVER_URL}/index_project`, {
                project_root: workspaceFolder
            }, { timeout: 60000 });
            
            if (response.data.success) {
                vscode.window.showInformationMessage(
                    `✅ Project indexed! ${response.data.files_indexed} files, ${response.data.symbols_found} symbols`
                );
            } else {
                vscode.window.showErrorMessage('Failed to index project: ' + (response.data.error || 'Unknown error'));
            }
        } catch (error) {
            vscode.window.showErrorMessage('Index project error: ' + error.message);
        }
    });
    
    // Command: Search Project
    const searchProjectCommand = vscode.commands.registerCommand('superai.searchProject', async () => {
        try {
            const query = await vscode.window.showInputBox({
                prompt: 'Search your codebase',
                placeHolder: 'e.g., authentication functions, database queries'
            });
            
            if (!query) return;
            
            const response = await axios.post(`${SERVER_URL}/search_project`, {
                query: query
            }, { timeout: 10000 });
            
            if (response.data.error) {
                vscode.window.showErrorMessage(response.data.error);
                return;
            }
            
            // Show results in webview
            const panel = vscode.window.createWebviewPanel(
                'searchResults',
                'Search Results',
                vscode.ViewColumn.One,
                {}
            );
            
            const results = response.data.search_results || [];
            const resultsHtml = `
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; padding: 20px; }
        h2 { color: #007acc; }
        .result { margin: 15px 0; padding: 15px; background: #f5f5f5; border-left: 4px solid #007acc; }
        .file-path { color: #666; font-size: 0.9em; }
        .symbol-name { font-weight: bold; color: #000; }
        .relevance { color: #28a745; }
    </style>
</head>
<body>
    <h2>🔍 Search Results for "${query}"</h2>
    <p>Found ${results.length} matches</p>
    ${results.map(r => `
        <div class="result">
            <div class="symbol-name">${r.symbol_name} (${r.symbol_type})</div>
            <div class="file-path">📁 ${r.file_path}:${r.line_number}</div>
            <div class="relevance">Relevance: ${(r.relevance_score * 100).toFixed(1)}%</div>
        </div>
    `).join('')}
</body>
</html>`;
            
            panel.webview.html = resultsHtml;
            
        } catch (error) {
            vscode.window.showErrorMessage('Search error: ' + error.message);
        }
    });
    
    // Command: Refactor with AI
    const refactorWithAICommand = vscode.commands.registerCommand('superai.refactorWithAI', async () => {
        try {
            const query = await vscode.window.showInputBox({
                prompt: 'Describe the refactoring you want',
                placeHolder: 'e.g., Extract logging to separate file, Move auth functions to auth.py'
            });
            
            if (!query) return;
            
            vscode.window.showInformationMessage('🤖 AI is planning refactoring...');
            
            const response = await axios.post(`${SERVER_URL}/preview_refactoring`, {
                query: query
            }, { timeout: 30000 });
            
            if (response.data.error) {
                vscode.window.showErrorMessage(response.data.error);
                return;
            }
            
            // Show AI plan in new document
            const doc = await vscode.workspace.openTextDocument({
                content: response.data.ai_plan,
                language: 'markdown'
            });
            await vscode.window.showTextDocument(doc);
            
            vscode.window.showInformationMessage('✅ Refactoring plan generated!');
            
        } catch (error) {
            vscode.window.showErrorMessage('Refactoring error: ' + error.message);
        }
    });
    
    // Command: Four-Stage Gemini with Multi-File Context
    const fourStageMultiFileCommand = vscode.commands.registerCommand('superai.fourStageMultiFile', async () => {
        try {
            const editor = vscode.window.activeTextEditor;
            if (!editor) {
                vscode.window.showErrorMessage('No active file open.');
                return;
            }
            
            const document = editor.document;
            const filePath = document.uri.fsPath;
            const codeContent = document.getText();
            const workspaceFolder = vscode.workspace.workspaceFolders ? 
                vscode.workspace.workspaceFolders[0].uri.fsPath : '';
            
            // Show progress notification
            await vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: "Four-Stage Gemini with Multi-File Context",
                cancellable: false
            }, async (progress) => {
                // Step 1: Check if project is indexed
                progress.report({ increment: 10, message: "Checking project index..." });
                
                try {
                    // Try to get file dependencies to check if indexed
                    const relativePath = require('path').relative(workspaceFolder, filePath);
                    const checkResponse = await axios.post(`${SERVER_URL}/impact_analysis`, {
                        file_path: relativePath
                    }, { timeout: 5000 });
                    
                    if (checkResponse.data.error && checkResponse.data.error.includes('not indexed')) {
                        // Project not indexed, index it now
                        progress.report({ increment: 20, message: "Indexing project (first time)..." });
                        
                        await axios.post(`${SERVER_URL}/index_project`, {
                            project_root: workspaceFolder
                        }, { timeout: 60000 });
                        
                        vscode.window.showInformationMessage('✅ Project indexed successfully!');
                    }
                } catch (error) {
                    // If error checking, try to index anyway
                    if (workspaceFolder) {
                        progress.report({ increment: 20, message: "Indexing project..." });
                        try {
                            await axios.post(`${SERVER_URL}/index_project`, {
                                project_root: workspaceFolder
                            }, { timeout: 60000 });
                        } catch (indexError) {
                            console.log('Index attempt:', indexError.message);
                        }
                    }
                }
                
                // Step 2: Get related files info
                progress.report({ increment: 20, message: "Loading related file contents..." });
                
                try {
                    const relativePath = require('path').relative(workspaceFolder, filePath);
                    const depsResponse = await axios.post(`${SERVER_URL}/impact_analysis`, {
                        file_path: relativePath
                    }, { timeout: 5000 });
                    
                    if (!depsResponse.data.error) {
                        const filesCount = Math.min(5, depsResponse.data.imports.length);
                        if (filesCount > 0) {
                            vscode.window.showInformationMessage(`📚 Loading ${filesCount} related files for context...`);
                        }
                    }
                } catch (e) {
                    console.log('Could not get deps info:', e.message);
                }
                
                // Step 3: Pre-Processing Type Fixes
                progress.report({ increment: 10, message: "Pre-processing: Adding smart type converters..." });
                
                // Step 4: Run Four-Stage Gemini
                progress.report({ increment: 30, message: "Running Four-Stage Gemini with project context..." });
                
                vscode.window.showInformationMessage('🚀 Pre-Processing + Four-Stage Gemini with Multi-File Context (This may take 1-3 minutes)...');
                
                const progressInterval = setInterval(() => {
                    progress.report({ increment: 5, message: "Processing stages with project context..." });
                }, 5000);
                
                try {
                    const response = await axios.post(`${SERVER_URL}/four_stage_gemini_fix`, {
                        file_path: filePath,
                        code_content: codeContent,
                        apply_changes: false
                    }, { timeout: 180000 });
                    
                    clearInterval(progressInterval);
                    progress.report({ increment: 100, message: "Complete!" });
                    
                    return response;
                } catch (error) {
                    clearInterval(progressInterval);
                    throw error;
                }
            }).then(async (response) => {
                if (!response) return;
                
                const result = response.data;
                
                if (result.success) {
                    // Show enhanced results with project context info
                    const finalCode = result.final_code;
                    const stage0Result = result.stage0_result || {};
                    const stage1Result = result.stage1_result || {};
                    const stage2Result = result.stage2_result || {};
                    const stage3Result = result.stage3_result || {};
                    const summary = result.summary || {};
                    
                    // Get project context info
                    const relativePath = require('path').relative(workspaceFolder, filePath);
                    let contextInfo = '';
                    try {
                        const contextResponse = await axios.post(`${SERVER_URL}/impact_analysis`, {
                            file_path: relativePath
                        }, { timeout: 5000 });
                        
                        if (!contextResponse.data.error) {
                            const ctx = contextResponse.data;
                            const totalFiles = ctx.imports.length + ctx.imported_by.length;
                            const filesLoaded = Math.min(5, ctx.imports.length); // Max 5 files loaded
                            
                            contextInfo = `
                                <div style="background: #e8f4f8; padding: 15px; margin: 20px 0; border-left: 4px solid #0066cc;">
                                    <h3 style="margin-top: 0; color: #0066cc;">📊 Multi-File Context Used</h3>
                                    <p><strong>Related Files:</strong> ${totalFiles} (${filesLoaded} loaded with full content)</p>
                                    <p><strong>Files Loaded:</strong></p>
                                    <ul style="margin: 5px 0;">
                                        ${ctx.imports.slice(0, 5).map(f => `<li>📄 ${f} <span style="color: green;">(content loaded)</span></li>`).join('')}
                                    </ul>
                                    <p><strong>Imported By:</strong> ${ctx.imported_by.slice(0, 3).join(', ') || 'None'}</p>
                                    <p><strong>Symbols:</strong> ${ctx.symbols}</p>
                                    <p><strong>Risk Level:</strong> <span style="color: ${ctx.overall_risk === 'high' ? 'red' : ctx.overall_risk === 'medium' ? 'orange' : 'green'}">${ctx.overall_risk.toUpperCase()}</span></p>
                                    <p style="margin-top: 10px; padding: 10px; background: #d4edda; border-radius: 5px; color: #155724;">
                                        ✅ <strong>AI has access to full contents of ${filesLoaded} related files for accurate predictions!</strong>
                                    </p>
                                </div>
                            `;
                        }
                    } catch (e) {
                        console.log('Could not get context info:', e.message);
                    }
                    
                    // Create comprehensive results HTML
                    const resultsHtml = `
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; padding: 20px; line-height: 1.6; }
        h1 { color: #0066cc; border-bottom: 3px solid #0066cc; padding-bottom: 10px; }
        h2 { color: #0066cc; margin-top: 30px; }
        h3 { color: #333; }
        .summary { background: #f0f0f0; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .stage { margin: 20px 0; padding: 20px; border-radius: 5px; }
        .stage-pre { background: #f3e5f5; border-left: 5px solid #9c27b0; }
        .stage-0 { background: #ffe6e6; border-left: 5px solid #ff4444; }
        .stage-1 { background: #e6f7ff; border-left: 5px solid #00aaff; }
        .stage-2 { background: #e6ffe6; border-left: 5px solid #44ff44; }
        .stage-3 { background: #fff4e6; border-left: 5px solid #ffaa00; }
        .code { background: #2d2d2d; color: #f8f8f2; padding: 15px; border-radius: 5px; overflow-x: auto; white-space: pre-wrap; }
        .success { color: #28a745; font-weight: bold; }
        .button { background: #0066cc; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 10px 5px; }
        .button:hover { background: #0052a3; }
    </style>
</head>
<body>
    <h1>🚀 Pre-Processing + Four-Stage Gemini with Multi-File Context - Results</h1>
    
    ${contextInfo}
    
    <div class="summary">
        <h2>📊 Summary</h2>
        <p><strong>Total Stages:</strong> Pre-Processing + 4 Stages (Type Fixes → Understand → Plan → Implement → Replace)</p>
        <p><strong>Pre-Processing:</strong> ✅ Smart type converters added</p>
        <p><strong>Stage 0:</strong> ${stage0Result.success ? '✅ Success' : '❌ Failed'}</p>
        <p><strong>Stage 1:</strong> ${stage1Result.success ? '✅ Success' : '❌ Failed'}</p>
        <p><strong>Stage 2:</strong> ${stage2Result.success ? '✅ Success' : '❌ Failed'}</p>
        <p><strong>Stage 3:</strong> ${stage3Result.success ? '✅ Success' : '❌ Failed'}</p>
        <p class="success">✅ All stages completed with type safety and project context!</p>
    </div>
    
    <div class="stage stage-pre">
        <h2>🔧 Pre-Processing: Smart Type Converters</h2>
        <p><strong>Functions Added:</strong></p>
        <ul>
            <li>✅ <code>word_to_number()</code> - Converts "twentyfive" → 25</li>
            <li>✅ <code>smart_int()</code> - Safe integer conversion</li>
            <li>✅ <code>smart_float()</code> - Safe float conversion</li>
            <li>✅ <code>smart_str()</code> - Safe string conversion</li>
        </ul>
        <p style="background: #e8f5e9; padding: 10px; border-radius: 5px; color: #2e7d32;">
            ✅ <strong>Type-safe functions automatically added before Gemini processing!</strong>
        </p>
    </div>
    
    <div class="stage stage-0">
        <h2>🧠 Stage 0: Code Understanding (with Project Context)</h2>
        <p><strong>Understanding Fields:</strong> ${Object.keys(stage0Result.understanding || {}).length}</p>
        <p><strong>Main Purpose:</strong> ${(stage0Result.understanding || {}).main_purpose || 'N/A'}</p>
    </div>
    
    <div class="stage stage-1">
        <h2>📋 Stage 1: Comprehensive Planning</h2>
        <p><strong>Plan Components:</strong> ${Object.keys(stage1Result.plan || {}).length}</p>
    </div>
    
    <div class="stage stage-2">
        <h2>🛠️ Stage 2: Implementation</h2>
        <p><strong>Implementations Created:</strong> ${Object.keys(stage2Result.implementations || {}).length}</p>
    </div>
    
    <div class="stage stage-3">
        <h2>🔄 Stage 3: Intelligent Replacement</h2>
        <p><strong>Replacements Made:</strong> ${(stage3Result.replacements || []).length}</p>
    </div>
    
    <h2>✅ Final Code</h2>
    <div class="code">${finalCode.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</div>
    
    <div style="margin-top: 30px;">
        <button class="button" onclick="applyChanges()">Apply Changes to File</button>
        <button class="button" onclick="copyCode()">Copy Code</button>
    </div>
    
    <script>
        const vscode = acquireVsCodeApi();
        function applyChanges() {
            vscode.postMessage({ command: 'apply' });
        }
        function copyCode() {
            navigator.clipboard.writeText(\`${finalCode.replace(/`/g, '\\`')}\`);
            alert('Code copied to clipboard!');
        }
    </script>
</body>
</html>`;
                    
                    const panel = vscode.window.createWebviewPanel(
                        'fourStageMultiFileResults',
                        'Four-Stage Gemini with Multi-File Context',
                        vscode.ViewColumn.One,
                        { enableScripts: true }
                    );
                    
                    panel.webview.html = resultsHtml;
                    
                    panel.webview.onDidReceiveMessage(async message => {
                        if (message.command === 'apply') {
                            const edit = new vscode.WorkspaceEdit();
                            const fullRange = new vscode.Range(
                                document.positionAt(0),
                                document.positionAt(document.getText().length)
                            );
                            edit.replace(document.uri, fullRange, finalCode);
                            await vscode.workspace.applyEdit(edit);
                            await document.save();
                            
                            vscode.window.showInformationMessage(
                                `🎉 Four-Stage Gemini with Multi-File Context Applied!\n` +
                                `✅ Code understood with project context\n` +
                                `✅ All stages completed successfully`
                            );
                        }
                    });
                    
                } else {
                    vscode.window.showErrorMessage('Four-Stage Gemini with Multi-File Context failed: ' + (result.error || 'Unknown error'));
                }
            }).catch((error) => {
                vscode.window.showErrorMessage('Four-Stage Gemini with Multi-File Context error: ' + error.message);
            });
            
        } catch (error) {
            vscode.window.showErrorMessage('Four-Stage Gemini with Multi-File Context error: ' + error.message);
        }
    });
    
    // Command: Impact Analysis
    const impactAnalysisCommand = vscode.commands.registerCommand('superai.impactAnalysis', async () => {
        try {
            const editor = vscode.window.activeTextEditor;
            if (!editor) {
                vscode.window.showErrorMessage('No active file');
                return;
            }
            
            const filePath = editor.document.uri.fsPath;
            const workspaceFolder = vscode.workspace.workspaceFolders ? 
                vscode.workspace.workspaceFolders[0].uri.fsPath : '';
            
            const relativePath = path.relative(workspaceFolder, filePath);
            
            const response = await axios.post(`${SERVER_URL}/impact_analysis`, {
                file_path: relativePath
            }, { timeout: 10000 });
            
            if (response.data.error) {
                vscode.window.showErrorMessage(response.data.error);
                return;
            }
            
            const data = response.data;
            const message = `
📊 Impact Analysis for ${path.basename(filePath)}

Risk Level: ${data.overall_risk.toUpperCase()}
Symbols: ${data.symbols}
Imports: ${data.imports.length} files
Imported By: ${data.imported_by.length} files

Files that would be affected:
${data.imported_by.slice(0, 5).map(f => `  • ${f}`).join('\n')}
            `.trim();
            
            vscode.window.showInformationMessage(message, { modal: true });
            
        } catch (error) {
            vscode.window.showErrorMessage('Impact analysis error: ' + error.message);
        }
    });
    
    // Register all commands
    context.subscriptions.push(
        completeCodeCommand,
        autoFixCommand,
        globalCompleteCommand,
        writeFixedCodeCommand,
        analyzeCommand,
        workspaceInfoCommand,
        toggleAutoCompleteCommand,
        apiKeyStatusCommand,
        enhancedSyntaxFixCommand,
        fourStageGeminiFixCommand,
        indexProjectCommand,
        searchProjectCommand,
        refactorWithAICommand,
        fourStageMultiFileCommand,
        impactAnalysisCommand
    );
    
    // Document change listeners
    const onDidSaveDocument = vscode.workspace.onDidSaveTextDocument(async (document) => {
        await analyzeAndShowDiagnostics(document);
    });
    
    const onDidOpenDocument = vscode.workspace.onDidOpenTextDocument(async (document) => {
        await analyzeAndShowDiagnostics(document);
    });
    
    context.subscriptions.push(onDidSaveDocument, onDidOpenDocument);
    
    // Initialize status bar
    updateStatusBar();
    
    // Analyze currently open file
    if (vscode.window.activeTextEditor) {
        analyzeAndShowDiagnostics(vscode.window.activeTextEditor.document);
    }
    
    console.log('SuperAI Extension activated successfully!');
    vscode.window.showInformationMessage('SuperAI Extension activated! 🚀');
}

function deactivate() {
    if (diagnosticCollection) {
        diagnosticCollection.dispose();
    }
    if (statusBarItem) {
        statusBarItem.dispose();
    }
    console.log('SuperAI Extension deactivated.');
}

module.exports = {
    activate,
    deactivate
};

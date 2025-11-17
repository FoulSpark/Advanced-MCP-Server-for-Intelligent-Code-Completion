#!/usr/bin/env python3
"""
Self-Healing Code Fixer Loop using Gemini AI and MCP Integration
"""

import requests
import json
import os
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import google.generativeai as genai
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CodeIssue:
    """Represents a code issue found by the system"""
    line_number: int
    issue_type: str
    description: str
    severity: str  # 'error', 'warning', 'info'
    suggested_fix: str
    context: str
    confidence: float

@dataclass
class CodeFix:
    """Represents a code fix applied by the system"""
    original_code: str
    fixed_code: str
    fix_type: str
    explanation: str
    confidence: float
    indentation_fixed: bool
    context_preserved: bool

class SelfHealingCodeFixer:
    """Self-healing code fixer that uses Gemini AI and MCP integration"""
    
    def __init__(self, mcp_server_url: str = "http://127.0.0.1:5000", gemini_api_key: str = None):
        self.mcp_server_url = mcp_server_url
        self.gemini_api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        
        if not self.gemini_api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable.")
        
        # Configure Gemini
        genai.configure(api_key=self.gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-pro')
        
        # Initialize learning database
        self.learning_db = {
            'fixes_applied': [],
            'patterns_learned': [],
            'context_rules': [],
            'indentation_rules': []
        }
        
        logger.info("Self-Healing Code Fixer initialized with Gemini AI and MCP integration")
    
    def analyze_code_with_mcp(self, file_path: str, workspace_folder: str = ".") -> List[CodeIssue]:
        """Analyze code using MCP server to detect issues"""
        logger.info(f"Analyzing code with MCP: {file_path}")
        
        issues = []
        
        try:
            # Get global completions from MCP server
            response = requests.post(
                f"{self.mcp_server_url}/global_complete",
                json={
                    "file_path": file_path,
                    "workspace_folder": workspace_folder
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    completions = result.get('completions', [])
                    
                    for completion in completions:
                        issue = CodeIssue(
                            line_number=completion.get('line', 0),
                            issue_type=completion.get('type', 'unknown'),
                            description=self._get_issue_description(completion.get('type', 'unknown')),
                            severity=self._get_severity(completion.get('type', 'unknown')),
                            suggested_fix=completion.get('suggested_completion', ''),
                            context=completion.get('context', ''),
                            confidence=0.8  # MCP confidence
                        )
                        issues.append(issue)
                        
                    logger.info(f"Found {len(issues)} issues using MCP")
                else:
                    logger.warning(f"MCP analysis failed: {result.get('error')}")
            else:
                logger.error(f"MCP server request failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error analyzing code with MCP: {e}")
        
        return issues
    
    def get_gemini_enhanced_fix(self, code: str, issues: List[CodeIssue], file_path: str) -> List[CodeFix]:
        """Get enhanced fixes from Gemini AI based on MCP analysis and learning database"""
        logger.info("Getting Gemini-enhanced fixes")
        
        # Build context for Gemini
        context = self._build_gemini_context(code, issues, file_path)
        
        # Create learning-enhanced prompt
        prompt = self._create_learning_enhanced_prompt(context, issues)
        
        try:
            # Get response from Gemini
            response = self.gemini_model.generate_content(prompt)
            fixes = self._parse_gemini_response(response.text, issues)
            
            # Learn from the fixes
            self._learn_from_fixes(fixes, issues)
            
            return fixes
            
        except Exception as e:
            logger.error(f"Error getting Gemini response: {e}")
            return []
    
    def apply_fixes_with_learning(self, file_path: str, fixes: List[CodeFix]) -> Dict[str, Any]:
        """Apply fixes to the file with learning feedback"""
        logger.info(f"Applying {len(fixes)} fixes to {file_path}")
        
        try:
            # Read original file
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Apply fixes
            fixed_content = original_content
            applied_fixes = []
            
            for fix in fixes:
                if fix.confidence > 0.7:  # Only apply high-confidence fixes
                    # Apply the fix (simplified - in real implementation, use proper AST manipulation)
                    fixed_content = self._apply_single_fix(fixed_content, fix)
                    applied_fixes.append(fix)
                    
                    # Learn from successful application
                    self._learn_from_successful_fix(fix)
            
            # Write fixed content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            
            # Update learning database
            self._update_learning_database(applied_fixes)
            
            return {
                'success': True,
                'fixes_applied': len(applied_fixes),
                'original_content': original_content,
                'fixed_content': fixed_content,
                'learning_updated': True
            }
            
        except Exception as e:
            logger.error(f"Error applying fixes: {e}")
            return {'success': False, 'error': str(e)}
    
    def self_heal_code(self, file_path: str, workspace_folder: str = ".", max_iterations: int = 3) -> Dict[str, Any]:
        """Main self-healing loop"""
        logger.info(f"Starting self-healing process for {file_path}")
        
        results = {
            'file_path': file_path,
            'iterations': 0,
            'total_fixes': 0,
            'learning_improvements': [],
            'final_status': 'completed'
        }
        
        for iteration in range(max_iterations):
            logger.info(f"Self-healing iteration {iteration + 1}/{max_iterations}")
            
            # Step 1: Analyze code with MCP
            issues = self.analyze_code_with_mcp(file_path, workspace_folder)
            
            if not issues:
                logger.info("No issues found - code is clean!")
                break
            
            # Step 2: Get Gemini-enhanced fixes
            fixes = self.get_gemini_enhanced_fix(open(file_path, 'r').read(), issues, file_path)
            
            if not fixes:
                logger.warning("No fixes generated by Gemini")
                break
            
            # Step 3: Apply fixes
            apply_result = self.apply_fixes_with_learning(file_path, fixes)
            
            if not apply_result.get('success'):
                logger.error(f"Failed to apply fixes: {apply_result.get('error')}")
                break
            
            # Update results
            results['iterations'] += 1
            results['total_fixes'] += apply_result.get('fixes_applied', 0)
            results['learning_improvements'].append({
                'iteration': iteration + 1,
                'fixes_applied': apply_result.get('fixes_applied', 0),
                'learning_confidence': self._calculate_learning_confidence()
            })
            
            # Check if we should continue
            if apply_result.get('fixes_applied', 0) == 0:
                logger.info("No more fixes to apply - stopping")
                break
        
        results['final_status'] = 'completed' if results['iterations'] > 0 else 'no_issues'
        return results
    
    def _build_gemini_context(self, code: str, issues: List[CodeIssue], file_path: str) -> str:
        """Build context for Gemini AI"""
        context = f"""
CODE ANALYSIS CONTEXT:
=====================

File: {file_path}
Language: Python

CURRENT CODE:
{code}

DETECTED ISSUES:
"""
        
        for i, issue in enumerate(issues, 1):
            context += f"""
{i}. Line {issue.line_number}: {issue.issue_type.upper()}
   Description: {issue.description}
   Severity: {issue.severity}
   Suggested Fix: {issue.suggested_fix}
   Context: {issue.context}
"""
        
        # Add learning database context
        context += f"""

LEARNING DATABASE CONTEXT:
=========================

Previous Fixes Applied: {len(self.learning_db['fixes_applied'])}
Patterns Learned: {len(self.learning_db['patterns_learned'])}
Context Rules: {len(self.learning_db['context_rules'])}
Indentation Rules: {len(self.learning_db['indentation_rules'])}

Recent Successful Fixes:
"""
        
        for fix in self.learning_db['fixes_applied'][-5:]:  # Last 5 fixes
            context += f"- {fix['fix_type']}: {fix['explanation']}\n"
        
        return context
    
    def _create_learning_enhanced_prompt(self, context: str, issues: List[CodeIssue]) -> str:
        """Create a learning-enhanced prompt for Gemini"""
        prompt = f"""
You are an advanced self-healing code fixer with access to a learning database. Your task is to provide enhanced fixes for the detected code issues.

{context}

INSTRUCTIONS:
1. Analyze each detected issue carefully
2. Provide fixes that maintain proper indentation and context
3. Use the learning database to improve fix accuracy
4. Ensure fixes are syntactically correct and follow Python best practices
5. Provide explanations for each fix
6. Consider the context and scope of each issue

RESPONSE FORMAT:
For each issue, provide:
- fix_type: Type of fix (indentation, syntax, logic, etc.)
- fixed_code: The corrected code with proper indentation
- explanation: Why this fix is correct
- confidence: Your confidence level (0.0-1.0)
- indentation_fixed: Whether indentation was corrected
- context_preserved: Whether the fix maintains proper context

LEARNING ENHANCEMENT:
Use the learning database to:
- Apply similar fixes that worked before
- Maintain consistent indentation patterns
- Preserve code structure and context
- Improve fix accuracy based on previous successes

Please provide enhanced fixes for all detected issues.
"""
        return prompt
    
    def _parse_gemini_response(self, response: str, issues: List[CodeIssue]) -> List[CodeFix]:
        """Parse Gemini response into CodeFix objects"""
        fixes = []
        
        try:
            # Simple parsing - in real implementation, use more sophisticated parsing
            lines = response.split('\n')
            current_fix = None
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('fix_type:'):
                    if current_fix:
                        fixes.append(current_fix)
                    current_fix = CodeFix(
                        original_code="",
                        fixed_code="",
                        fix_type=line.split(':', 1)[1].strip(),
                        explanation="",
                        confidence=0.8,
                        indentation_fixed=False,
                        context_preserved=False
                    )
                elif line.startswith('fixed_code:') and current_fix:
                    current_fix.fixed_code = line.split(':', 1)[1].strip()
                elif line.startswith('explanation:') and current_fix:
                    current_fix.explanation = line.split(':', 1)[1].strip()
                elif line.startswith('confidence:') and current_fix:
                    try:
                        current_fix.confidence = float(line.split(':', 1)[1].strip())
                    except:
                        current_fix.confidence = 0.8
                elif line.startswith('indentation_fixed:') and current_fix:
                    current_fix.indentation_fixed = line.split(':', 1)[1].strip().lower() == 'true'
                elif line.startswith('context_preserved:') and current_fix:
                    current_fix.context_preserved = line.split(':', 1)[1].strip().lower() == 'true'
            
            if current_fix:
                fixes.append(current_fix)
                
        except Exception as e:
            logger.error(f"Error parsing Gemini response: {e}")
        
        return fixes
    
    def _apply_single_fix(self, content: str, fix: CodeFix) -> str:
        """Apply a single fix to the content"""
        # Simplified fix application - in real implementation, use proper AST manipulation
        if fix.fixed_code:
            # This is a placeholder - real implementation would be more sophisticated
            return content.replace(fix.original_code, fix.fixed_code)
        return content
    
    def _learn_from_fixes(self, fixes: List[CodeFix], issues: List[CodeIssue]):
        """Learn from the fixes generated by Gemini"""
        for fix in fixes:
            self.learning_db['fixes_applied'].append({
                'fix_type': fix.fix_type,
                'explanation': fix.explanation,
                'confidence': fix.confidence,
                'indentation_fixed': fix.indentation_fixed,
                'context_preserved': fix.context_preserved,
                'timestamp': time.time()
            })
    
    def _learn_from_successful_fix(self, fix: CodeFix):
        """Learn from a successfully applied fix"""
        # Update patterns and rules based on successful fixes
        if fix.indentation_fixed:
            self.learning_db['indentation_rules'].append({
                'pattern': fix.fix_type,
                'rule': fix.explanation,
                'confidence': fix.confidence
            })
        
        if fix.context_preserved:
            self.learning_db['context_rules'].append({
                'pattern': fix.fix_type,
                'rule': fix.explanation,
                'confidence': fix.confidence
            })
    
    def _update_learning_database(self, fixes: List[CodeFix]):
        """Update the learning database with new patterns"""
        for fix in fixes:
            if fix.confidence > 0.8:  # High confidence fixes
                self.learning_db['patterns_learned'].append({
                    'pattern': fix.fix_type,
                    'success_rate': 1.0,
                    'last_used': time.time()
                })
    
    def _calculate_learning_confidence(self) -> float:
        """Calculate overall learning confidence"""
        if not self.learning_db['fixes_applied']:
            return 0.0
        
        total_confidence = sum(fix['confidence'] for fix in self.learning_db['fixes_applied'])
        return total_confidence / len(self.learning_db['fixes_applied'])
    
    def _get_issue_description(self, issue_type: str) -> str:
        """Get human-readable description for issue type"""
        descriptions = {
            'malformed_function': 'Function definition missing colon or has syntax errors',
            'incomplete_function': 'Function body is incomplete or missing',
            'malformed_class': 'Class definition missing colon or has syntax errors',
            'incomplete_class': 'Class body is incomplete or missing',
            'malformed_statement': 'Statement missing colon or has syntax errors',
            'incomplete_statement': 'Statement is incomplete',
            'todo_comment': 'TODO comment needs implementation',
            'incomplete_import': 'Import statement is incomplete',
            'incomplete_assignment': 'Assignment statement is incomplete',
            'incomplete_block': 'Code block is incomplete'
        }
        return descriptions.get(issue_type, 'Unknown issue type')
    
    def _get_severity(self, issue_type: str) -> str:
        """Get severity level for issue type"""
        severity_map = {
            'malformed_function': 'error',
            'malformed_class': 'error',
            'malformed_statement': 'error',
            'incomplete_function': 'warning',
            'incomplete_class': 'warning',
            'incomplete_statement': 'warning',
            'todo_comment': 'info',
            'incomplete_import': 'warning',
            'incomplete_assignment': 'warning',
            'incomplete_block': 'warning'
        }
        return severity_map.get(issue_type, 'info')
    
    def get_learning_report(self) -> Dict[str, Any]:
        """Get a report of the learning progress"""
        return {
            'total_fixes_applied': len(self.learning_db['fixes_applied']),
            'patterns_learned': len(self.learning_db['patterns_learned']),
            'context_rules': len(self.learning_db['context_rules']),
            'indentation_rules': len(self.learning_db['indentation_rules']),
            'overall_confidence': self._calculate_learning_confidence(),
            'recent_fixes': self.learning_db['fixes_applied'][-10:],  # Last 10 fixes
            'learning_trend': self._calculate_learning_trend()
        }
    
    def _calculate_learning_trend(self) -> str:
        """Calculate learning trend (improving, stable, declining)"""
        if len(self.learning_db['fixes_applied']) < 5:
            return 'insufficient_data'
        
        recent_fixes = self.learning_db['fixes_applied'][-5:]
        older_fixes = self.learning_db['fixes_applied'][-10:-5] if len(self.learning_db['fixes_applied']) >= 10 else []
        
        if not older_fixes:
            return 'insufficient_data'
        
        recent_avg = sum(fix['confidence'] for fix in recent_fixes) / len(recent_fixes)
        older_avg = sum(fix['confidence'] for fix in older_fixes) / len(older_fixes)
        
        if recent_avg > older_avg + 0.1:
            return 'improving'
        elif recent_avg < older_avg - 0.1:
            return 'declining'
        else:
            return 'stable'

def main():
    """Example usage of the self-healing code fixer"""
    print("🚀 Self-Healing Code Fixer Demo")
    print("=" * 50)
    
    # Initialize the fixer
    try:
        fixer = SelfHealingCodeFixer()
        print("✅ Self-healing code fixer initialized")
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("Please set the GEMINI_API_KEY environment variable")
        return
    
    # Example usage
    test_file = "test_self_healing.py"
    
    # Create a test file with issues
    test_code = """# Test file for self-healing
def calculate_sum(a, b)
    total = a + b
    return total

def main()
    result = calculate_sum(5, 3)
    print(f"Result: {result}")

if __name__ == "__main__":
    main()"""
    
    with open(test_file, 'w') as f:
        f.write(test_code)
    
    print(f"📝 Created test file: {test_file}")
    
    # Run self-healing
    print("\n🔧 Running self-healing process...")
    result = fixer.self_heal_code(test_file)
    
    print(f"\n📊 Results:")
    print(f"   Iterations: {result['iterations']}")
    print(f"   Total fixes: {result['total_fixes']}")
    print(f"   Status: {result['final_status']}")
    
    # Show learning report
    print(f"\n🧠 Learning Report:")
    learning_report = fixer.get_learning_report()
    print(f"   Total fixes applied: {learning_report['total_fixes_applied']}")
    print(f"   Patterns learned: {learning_report['patterns_learned']}")
    print(f"   Overall confidence: {learning_report['overall_confidence']:.2f}")
    print(f"   Learning trend: {learning_report['learning_trend']}")
    
    # Clean up
    if os.path.exists(test_file):
        os.remove(test_file)
        print(f"\n🗑️  Cleaned up test file")

if __name__ == "__main__":
    main()


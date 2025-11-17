"""
Multi-File Editor - Generates and applies patches/diffs across multiple files
Provides preview and approval workflow for code changes
"""
import difflib
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class FilePatch:
    """Represents a patch for a single file"""
    file_path: str
    original_content: str
    modified_content: str
    diff: str
    line_changes: Dict[str, int]
    
    def to_dict(self):
        return {
            'file_path': self.file_path,
            'diff': self.diff,
            'line_changes': self.line_changes,
            'preview': self._generate_preview()
        }
    
    def _generate_preview(self) -> str:
        """Generate a human-readable preview of changes"""
        lines = self.diff.split('\n')
        preview_lines = []
        
        for line in lines[:50]:
            if line.startswith('+') and not line.startswith('+++'):
                preview_lines.append(f"  {line}")
            elif line.startswith('-') and not line.startswith('---'):
                preview_lines.append(f"  {line}")
            elif line.startswith('@@'):
                preview_lines.append(f"\n{line}\n")
        
        return '\n'.join(preview_lines)

@dataclass
class MultiFilePatch:
    """Represents patches for multiple files"""
    patches: List[FilePatch]
    description: str
    total_files: int
    total_additions: int
    total_deletions: int
    
    def to_dict(self):
        return {
            'description': self.description,
            'total_files': self.total_files,
            'total_additions': self.total_additions,
            'total_deletions': self.total_deletions,
            'patches': [p.to_dict() for p in self.patches]
        }

class MultiFileEditor:
    """Handles multi-file editing with diff generation and preview"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.pending_patches: List[MultiFilePatch] = []
    
    def generate_diff(self, original: str, modified: str, file_path: str) -> str:
        """Generate unified diff between original and modified content"""
        original_lines = original.splitlines(keepends=True)
        modified_lines = modified.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm=''
        )
        
        return ''.join(diff)
    
    def create_file_patch(self, file_path: str, original_content: str, 
                         modified_content: str) -> FilePatch:
        """Create a patch for a single file"""
        
        diff = self.generate_diff(original_content, modified_content, file_path)
        
        additions = sum(1 for line in diff.split('\n') if line.startswith('+') and not line.startswith('+++'))
        deletions = sum(1 for line in diff.split('\n') if line.startswith('-') and not line.startswith('---'))
        
        return FilePatch(
            file_path=file_path,
            original_content=original_content,
            modified_content=modified_content,
            diff=diff,
            line_changes={'added': additions, 'removed': deletions}
        )
    
    def create_multi_file_patch(self, changes: Dict[str, Tuple[str, str]], 
                               description: str) -> MultiFilePatch:
        """Create patches for multiple files"""
        
        patches = []
        total_additions = 0
        total_deletions = 0
        
        for file_path, (original, modified) in changes.items():
            patch = self.create_file_patch(file_path, original, modified)
            patches.append(patch)
            total_additions += patch.line_changes['added']
            total_deletions += patch.line_changes['removed']
        
        multi_patch = MultiFilePatch(
            patches=patches,
            description=description,
            total_files=len(patches),
            total_additions=total_additions,
            total_deletions=total_deletions
        )
        
        self.pending_patches.append(multi_patch)
        return multi_patch
    
    def preview_changes(self, multi_patch: MultiFilePatch) -> str:
        """Generate a preview of all changes"""
        preview = []
        preview.append(f"📝 {multi_patch.description}")
        preview.append(f"📊 Changes: {multi_patch.total_files} files, "
                      f"+{multi_patch.total_additions} -{multi_patch.total_deletions} lines\n")
        
        for i, patch in enumerate(multi_patch.patches, 1):
            preview.append(f"\n{'='*60}")
            preview.append(f"File {i}/{multi_patch.total_files}: {patch.file_path}")
            preview.append(f"Changes: +{patch.line_changes['added']} -{patch.line_changes['removed']}")
            preview.append(f"{'='*60}\n")
            preview.append(patch.diff)
        
        return '\n'.join(preview)
    
    def apply_patch(self, patch: FilePatch, backup: bool = True) -> bool:
        """Apply a single file patch"""
        try:
            file_path = self.project_root / patch.file_path
            
            if backup and file_path.exists():
                backup_path = file_path.with_suffix(file_path.suffix + '.backup')
                with open(file_path, 'r', encoding='utf-8') as f:
                    backup_content = f.read()
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(backup_content)
                logger.info(f"💾 Backup created: {backup_path}")
            
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(patch.modified_content)
            
            logger.info(f"✅ Applied patch to {patch.file_path}")
            return True
        
        except Exception as e:
            logger.error(f"❌ Failed to apply patch to {patch.file_path}: {e}")
            return False
    
    def apply_multi_file_patch(self, multi_patch: MultiFilePatch, 
                              backup: bool = True) -> Dict[str, bool]:
        """Apply patches to multiple files"""
        results = {}
        
        logger.info(f"🔄 Applying {multi_patch.total_files} patches...")
        
        for patch in multi_patch.patches:
            success = self.apply_patch(patch, backup)
            results[patch.file_path] = success
        
        successful = sum(1 for success in results.values() if success)
        logger.info(f"✅ Applied {successful}/{multi_patch.total_files} patches successfully")
        
        return results

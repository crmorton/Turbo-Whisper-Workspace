#!/usr/bin/env python3
"""
Project Tree Visualization Tool

This script generates a tree view of the project structure,
respecting .gitignore files and providing useful metadata.
"""
import os
import sys
import stat
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional
import click
from rich.console import Console
from rich.tree import Tree
from rich.markup import escape
from datetime import datetime

try:
    from gitignore_parser import parse_gitignore
except ImportError:
    def parse_gitignore(gitignore_file):
        """Simple fallback if gitignore_parser is not available"""
        return lambda f: False

@dataclass
class FileNode:
    """Represents a file or directory in the tree with additional context"""
    path: Path
    is_dir: bool
    stat_info: os.stat_result = None
    references: Set[Path] = field(default_factory=set)
    referenced_by: Set[Path] = field(default_factory=set)
    imports: Set[str] = field(default_factory=set)

class SmartTree:
    def __init__(self, root_path: str, display_mode: str = 'classic'):
        self.root = Path(root_path).resolve()
        self.nodes: Dict[Path, FileNode] = {}
        self.gitignore_patterns = self._load_gitignore()
        self.display_mode = display_mode
        self.use_color = True
        self.console = Console()
    
    COLORS = {
        'depth': '\033[36m',  # cyan for depth
        'perm': '\033[33m',   # yellow for permissions
        'id': '\033[35m',     # magenta for uid/gid
        'size': '\033[32m',   # green for size
        'time': '\033[34m',   # blue for timestamp
        'reset': '\033[0m'    # reset
    }
    
    def _load_gitignore(self):
        """Load gitignore patterns if available"""
        gitignore_path = self.root / '.gitignore'
        if gitignore_path.exists():
            try:
                return parse_gitignore(gitignore_path)
            except Exception as e:
                print(f"Error parsing .gitignore: {e}")
                return lambda f: False
        return lambda f: False
    
    def _get_file_emoji(self, mode):
        """Get appropriate emoji for file type"""
        if stat.S_ISDIR(mode): return "📁"
        elif stat.S_ISLNK(mode): return "🔗"
        elif mode & stat.S_IXUSR: return "⚙️"
        elif stat.S_ISSOCK(mode): return "🔌"
        elif stat.S_ISFIFO(mode): return "📝"
        elif stat.S_ISBLK(mode): return "💾"
        elif stat.S_ISCHR(mode): return "📺"
        else: return "📄"
    
    def format_hex_node(self, path: Path, depth: int = 0) -> str:
        """Format a node in hex format with all metadata"""
        node = self.nodes[path]
        stat_info = node.stat_info
        
        # Convert values to hex
        perms_hex = f"{stat_info.st_mode & 0o777:03x}"
        uid_hex = f"{stat_info.st_uid:x}"
        gid_hex = f"{stat_info.st_gid:x}"
        size_hex = f"{stat_info.st_size:x}"
        time_hex = f"{int(stat_info.st_mtime):x}"
        depth_hex = f"{depth:x}"
        
        emoji = self._get_file_emoji(stat_info.st_mode)
        
        if self.use_color:
            return (f"{self.COLORS['depth']}{depth_hex}{self.COLORS['reset']} "
                   f"{self.COLORS['perm']}{perms_hex}{self.COLORS['reset']} "
                   f"{self.COLORS['id']}{uid_hex} {gid_hex}{self.COLORS['reset']} "
                   f"{self.COLORS['size']}{size_hex}{self.COLORS['reset']} "
                   f"{self.COLORS['time']}{time_hex}{self.COLORS['reset']} "
                   f"{emoji} {path.name}")
        else:
            return f"{depth_hex} {perms_hex} {uid_hex} {gid_hex} {size_hex} {time_hex} {emoji} {path.name}"
    
    def format_classic_node(self, path: Path) -> str:
        """Format node in classic tree style"""
        node = self.nodes[path]
        stat_info = node.stat_info
        
        emoji = self._get_file_emoji(stat_info.st_mode)
        size_str = f"{stat_info.st_size:,} bytes" if not stat.S_ISDIR(stat_info.st_mode) else "directory"
        mod_time = datetime.fromtimestamp(stat_info.st_mtime).strftime("%Y-%m-%d %H:%M")
        
        return f"{emoji} {path.name} ({size_str}, modified: {mod_time})"
    
    def scan(self, max_depth=10, filter_ignored=True):
        """Scan the directory tree and build node structure"""
        def scan_dir(current: Path, depth=0):
            if depth > max_depth:
                return
            
            try:
                for item in current.iterdir():
                    # Skip if matches gitignore pattern
                    if filter_ignored and self.gitignore_patterns(str(item)):
                        continue
                    
                    try:
                        stat_info = item.stat()
                        is_dir = item.is_dir() and not item.is_symlink()
                        
                        # Create node
                        self.nodes[item] = FileNode(
                            path=item,
                            is_dir=is_dir,
                            stat_info=stat_info
                        )
                        
                        # Recursively scan directories
                        if is_dir:
                            scan_dir(item, depth + 1)
                    except (PermissionError, FileNotFoundError):
                        # Handle permission errors gracefully
                        pass
            except (PermissionError, FileNotFoundError):
                pass
        
        # Start scanning from root
        self.nodes[self.root] = FileNode(
            path=self.root,
            is_dir=True, 
            stat_info=self.root.stat()
        )
        scan_dir(self.root)
    
    def build_rich_tree(self):
        """Build and display a rich tree visualization"""
        def add_to_tree(node: Tree, path: Path, depth=0):
            if path not in self.nodes:
                return
            
            current = self.nodes[path]
            child_paths = sorted(
                [p for p in self.nodes if p.parent == path],
                key=lambda p: (not self.nodes[p].is_dir, p.name)
            )
            
            for child_path in child_paths:
                child = self.nodes[child_path]
                
                if self.display_mode == 'hex':
                    node_text = self.format_hex_node(child_path, depth)
                else:
                    node_text = self.format_classic_node(child_path)
                
                child_node = node.add(node_text)
                if child.is_dir:
                    add_to_tree(child_node, child_path, depth + 1)
        
        root_node = Tree(self.format_classic_node(self.root) if self.display_mode != 'hex' 
                          else self.format_hex_node(self.root))
        add_to_tree(root_node, self.root)
        self.console.print(root_node)

def main():
    """Main function to run the tree visualization"""
    # Get project root directory
    project_root = Path('/data/source/Turbo-Whisper-Workspace').resolve()
    
    # Create and configure SmartTree
    tree = SmartTree(str(project_root), display_mode='classic')
    
    # Scan with gitignore support
    print(f"Scanning project structure at {project_root}...")
    tree.scan(max_depth=5, filter_ignored=True)
    
    # Display the tree
    print("\nProject Structure:")
    tree.build_rich_tree()

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
update_cover.py - Update 封面.md with sub-headings from linked articles.

For each list item in 封面.md that contains a markdown link to a .md file,
reads the linked article, extracts all second-level (##) headings,
and inserts them as indented sub-items under the list item.

Usage:
    python update_cover.py <path_to_封面.md>
"""

import re
import sys
import urllib.parse
from pathlib import Path


def find_link_paths(line: str) -> list[str]:
    """Extract paths from markdown links [text](path) pointing to .md files."""
    pattern = r'\[([^\]]+)\]\(([^)]+\.md[^)]*)\)'
    paths = [path for _, path in re.findall(pattern, line)]
    # Strip anchors (#section) and query params (?foo=bar) from paths
    return [p.split('#')[0].split('?')[0] for p in paths]


def extract_headings(file_path: Path) -> list[str]:
    """Extract second-level headings (##) from a markdown file."""
    if not file_path.exists():
        return []

    headings = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if re.match(r'^##\s+', stripped) and not re.match(r'^###\s+', stripped):
                headings.append(stripped.lstrip('#').strip())
    return headings


def leading_whitespace(line: str) -> str:
    """Return leading whitespace of a line."""
    m = re.match(r'^(\s*)', line)
    return m.group(1) if m else ''


def detect_line_ending(lines: list[str]) -> str:
    """Detect line ending from the first line of the file."""
    if not lines:
        return '\n'
    if lines[0].endswith('\r\n'):
        return '\r\n'
    return '\n'


def process_cover(cover_path: str) -> None:
    cover = Path(cover_path)
    if not cover.exists():
        print(f"Error: {cover} not found", file=sys.stderr)
        sys.exit(1)

    cover_dir = cover.parent
    lines = cover.read_text(encoding='utf-8').splitlines(keepends=True)
    le = detect_line_ending(lines)

    new_lines: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        is_list_item = stripped.startswith('- ') or stripped.startswith('* ')
        link_paths = find_link_paths(stripped) if is_list_item else []

        if not link_paths:
            new_lines.append(line)
            i += 1
            continue

        new_lines.append(line)

        cur_indent = leading_whitespace(line)
        cur_depth = len(cur_indent)

        i += 1
        while i < len(lines):
            next_depth = len(leading_whitespace(lines[i]))
            if next_depth <= cur_depth:
                break
            i += 1

        for path in link_paths:
            decoded = urllib.parse.unquote(path)
            linked_file = (cover_dir / decoded).resolve()
            headings = extract_headings(linked_file)
            if headings:
                print(f"  {linked_file.name}: {len(headings)} headings")
            sub_indent = cur_indent + "  "
            for h in headings:
                new_lines.append(f"{sub_indent}- {h}{le}")

    cover.write_text(''.join(new_lines), encoding='utf-8')
    print(f"Done: {cover}")


def main():
    if len(sys.argv) != 2 or sys.argv[1] in ('-h', '--help'):
        print("Usage: python update_cover.py <path_to_封面.md>", file=sys.stderr)
        sys.exit(1)
    process_cover(sys.argv[1])


if __name__ == '__main__':
    main()

"""Turn an uploaded notebook or script into executable code with a line map."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


@dataclass
class Source:
    kind: str                     # notebook | script
    code: str                     # executable Python
    line_map: list[tuple[int, int]] = field(default_factory=list)   # code line (1-based) -> (cell, line in cell)
    cells: list[dict] = field(default_factory=list)                 # {"type", "source"}
    markdown: str = ""

    def where(self, line: int | None) -> str:
        if not line or line < 1 or line > len(self.line_map):
            return ""
        cell, ln = self.line_map[line - 1]
        return f"cell {cell + 1}, line {ln}" if self.kind == "notebook" else f"line {ln}"

    def code_lines(self) -> list[str]:
        return self.code.split("\n")


_MAGIC = re.compile(r"^\s*(%|!)")


def load(name: str, content: bytes | str) -> Source:
    text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content
    if name.lower().endswith(".ipynb"):
        nb = json.loads(text)
        cells = []
        code_lines, line_map, md = [], [], []
        for ci, cell in enumerate(nb.get("cells", [])):
            src = cell.get("source", "")
            src = "".join(src) if isinstance(src, list) else src
            cells.append({"type": cell.get("cell_type"), "source": src})
            if cell.get("cell_type") == "markdown":
                md.append(src)
                continue
            if cell.get("cell_type") != "code":
                continue
            for li, line in enumerate(src.split("\n")):
                if _MAGIC.match(line):
                    line = "pass  # " + line.strip()   # keep line numbers stable
                code_lines.append(line)
                line_map.append((ci, li + 1))
            code_lines.append("")
            line_map.append((ci, len(src.split("\n")) + 1))
        return Source("notebook", "\n".join(code_lines), line_map, cells, "\n\n".join(md))
    lines = text.split("\n")
    comments = "\n".join(l.lstrip("# ") for l in lines if l.strip().startswith("#"))
    return Source("script", text, [(0, i + 1) for i in range(len(lines))], [{"type": "code", "source": text}],
                  comments)

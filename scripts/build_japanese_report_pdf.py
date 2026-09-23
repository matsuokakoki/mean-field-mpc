from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper" / "report_ja.md"
OUTPUT = ROOT / "paper" / "report_ja.pdf"
FONT_CANDIDATES = [
    Path("/winfonts/meiryo.ttc"),
    Path("/winfonts/msgothic.ttc"),
    Path("/winfonts/YuGothM.ttc"),
]


def _font() -> font_manager.FontProperties:
    for path in FONT_CANDIDATES:
        if path.exists():
            return font_manager.FontProperties(fname=str(path))
    raise FileNotFoundError("No Japanese font found under /winfonts")


def _clean_inline(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = text.replace("$", "")
    return text


def _wrap(text: str, width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    current_width = 0.0
    tokens = re.findall(r"[A-Za-z0-9_.%+-]+|\s+|.", text)
    for token in tokens:
        token_width = sum(
            min(0.5 if char.isascii() else 1.0, 0.7)
            if unicodedata.category(char).startswith("P")
            else 0.5 if char.isascii() else 1.0
            for char in token
        )
        if current_width + token_width > width and current:
            lines.append(current.rstrip())
            current = token.lstrip()
            current_width = 0.0 if token.isspace() else token_width
        else:
            current += token
            current_width += token_width
    if current.strip():
        lines.append(current.rstrip())
    return lines


def _blocks(markdown: str) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for raw in markdown.splitlines():
        line = raw.strip()
        if not line:
            parsed.append(("space", ""))
        elif line.startswith("# "):
            parsed.append(("title", _clean_inline(line[2:])))
        elif line.startswith("## "):
            parsed.append(("heading", _clean_inline(line[3:])))
        else:
            parsed.append(("body", _clean_inline(line)))
    return parsed


def main() -> None:
    font = _font()
    bold_font = font.copy()
    bold_font.set_weight("bold")
    blocks = _blocks(SOURCE.read_text(encoding="utf-8"))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(OUTPUT) as pdf:
        fig = Figure(figsize=(8.27, 11.69))
        ax = fig.add_axes((0, 0, 1, 1))
        ax.axis("off")
        y = 0.94
        page = 1

        def new_page() -> None:
            nonlocal fig, ax, y, page
            ax.text(0.5, 0.035, f"{page}", ha="center", va="bottom", fontsize=8, fontproperties=font)
            pdf.savefig(fig)
            page += 1
            fig = Figure(figsize=(8.27, 11.69))
            ax = fig.add_axes((0, 0, 1, 1))
            ax.axis("off")
            y = 0.94

        for kind, text in blocks:
            if kind == "space":
                y -= 0.018
                continue
            if kind == "title":
                size, spacing, fp, width = 19, 0.036, bold_font, 32
                y -= 0.012
            elif kind == "heading":
                size, spacing, fp, width = 14, 0.031, bold_font, 38
                y -= 0.018
            else:
                size, spacing, fp, width = 10.5, 0.024, font, 42

            for line in _wrap(text, width):
                if y < 0.08:
                    new_page()
                ax.text(0.10, y, line, ha="left", va="top", fontsize=size, fontproperties=fp)
                y -= spacing

        if y < 0.08:
            new_page()
        else:
            ax.text(0.5, 0.035, f"{page}", ha="center", va="bottom", fontsize=8, fontproperties=font)
            pdf.savefig(fig)

    print(OUTPUT)


if __name__ == "__main__":
    main()

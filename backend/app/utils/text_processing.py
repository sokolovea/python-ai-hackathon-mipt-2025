import re
import logging
from typing import List, Dict, Tuple


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def clean_text(text: str, remove_empty_lines: bool = True) -> str:
    """
    Очищает текст от лишних пробелов:
    - заменяет множественные пробелы на один
    - удаляет пробелы в начале/конце строк
    - удаляет пустые строки (если remove_empty_lines=True)
    """
    lines = text.splitlines()
    cleaned_lines = [re.sub(r'\s+', ' ', line).strip() for line in lines]
    if remove_empty_lines:
        cleaned_lines = [line for line in cleaned_lines if line]
    return '\n'.join(cleaned_lines)

def parse_markdown(markdown_text: str) -> list:
    """Парсер Markdown с очисткой пробелов (исходная реализация)"""
    elements = []
    lines = markdown_text.splitlines()
    i = 0
    prev_empty = False

    while i < len(lines):
        line = lines[i].strip()

        if not line:
            if not prev_empty:
                elements.append(("newline", ""))
                prev_empty = True
            i += 1
            continue

        prev_empty = False

        if line.startswith("# "):
            elements.append(("heading1", clean_text(line[2:])))
        elif line.startswith("## "):
            elements.append(("heading2", clean_text(line[3:])))
        elif line.startswith("### "):
            elements.append(("heading3", clean_text(line[4:])))
        elif line.startswith("#### "):
            elements.append(("heading4", clean_text(line[5:])))
        elif line.startswith("##### "):
            elements.append(("heading5", clean_text(line[6:])))
        elif line.startswith("\\["):
            math_content = []
            while i < len(lines):
                l = lines[i].strip()
                if l.endswith("\\]"):
                    math_content.append(clean_text(l[:-2]))
                    break
                math_content.append(clean_text(l[2:] if i == 0 else l))
                i += 1
            elements.append(("block_math", "\n".join(math_content)))
        else:
            cleaned_line = clean_text(line)
            parts = re.split(r'(\\\(.*?\\\)|\$.*?\$)', cleaned_line)
            formatted_parts = []

            for part in parts:
                if not part:
                    continue
                if part.startswith("\\(") and part.endswith("\\)"):
                    formatted_parts.append(("inline_math", clean_text(part[2:-2])))
                elif part.startswith("$") and part.endswith("$"):
                    formatted_parts.append(("inline_math", clean_text(part[1:-1])))
                else:
                    text_parts = []
                    bold_parts = re.split(r'(\*\*.*?\*\*)', part)
                    for bp in bold_parts:
                        if bp.startswith("") and bp.endswith(""):
                            text_parts.append(("bold", clean_text(bp[2:-2])))
                        else:
                            italic_parts = re.split(r'(\*.*?\*)', bp)
                            for ip in italic_parts:
                                if ip.startswith("*") and ip.endswith("*"):
                                    text_parts.append(("italic", clean_text(ip[1:-1])))
                                else:
                                    text_parts.append(("plain", clean_text(ip)))
                    formatted_parts.extend(text_parts)

            elements.append(("paragraph", formatted_parts))
        i += 1

    logging.debug(f"Распознано {len(elements)} элементов")
    return elements
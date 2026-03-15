"""
pdf_parser.py - Parsing dei PDF con pdfplumber

Scelta rispetto a unstructured[pdf]:
  - pdfplumber è stabile su Windows e Python 3.13 senza dipendenze pesanti
  - Ottimo per PDF nativi digitali (generati da Word, sistemi informatici, ecc.)
  - Gestisce testo, layout, tabelle con buona accuratezza
  - Se in futuro servisse OCR per PDF scansionati, si può aggiungere pytesseract

Strategia di riconoscimento struttura:
  - Titoli: font size > soglia, o testo in grassetto, o pattern numerici (1., 1.1, Art.)
  - Tabelle: rilevate nativamente da pdfplumber
  - Bullet list: righe che iniziano con simboli di lista
  - Paragrafi: blocchi di testo continuo
"""

import re
from pathlib import Path
from typing import Optional
from loguru import logger

import pdfplumber
from llama_index.core import Document

from preprocessing.config import MIN_TEXT_LENGTH


# ─────────────────────────────────────────────────────────────────────────────
# COSTANTI PER RICONOSCIMENTO STRUTTURA
# ─────────────────────────────────────────────────────────────────────────────

TITLE_PATTERNS = [
    r"^(art(icolo)?\.?\s*\d+)",
    r"^\d+(\.\d+)*\.?\s+[A-Z]",
    r"^(capitolo|sezione|allegato|appendice)\s",
    r"^[IVXLC]+\.\s+[A-Z]",
]
TITLE_REGEX = re.compile("|".join(TITLE_PATTERNS), re.IGNORECASE)
BULLET_SYMBOLS = {"•", "·", "–", "—", "►", "▪", "▸", "○", "●", "-", "*"}
TITLE_SIZE_RATIO = 1.15


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_text(text: str) -> str:
    text = text.replace("-\n", "")
    text = " ".join(text.split())
    return text.strip()

def _is_bullet(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return stripped[0] in BULLET_SYMBOLS

def _is_title_by_pattern(text: str) -> bool:
    return bool(TITLE_REGEX.match(text.strip()))

def _is_title_by_font(fontsize: Optional[float], avg_fontsize: float) -> bool:
    if fontsize is None or avg_fontsize == 0:
        return False
    return fontsize >= avg_fontsize * TITLE_SIZE_RATIO

def _build_section_path(hierarchy: list) -> str:
    return " > ".join(hierarchy) if hierarchy else ""


# ─────────────────────────────────────────────────────────────────────────────
# PARSER PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────

class PDFParser:
    def __init__(self, min_text_length: int = MIN_TEXT_LENGTH):
        self.min_text_length = min_text_length

    def parse(self, pdf_path) -> list:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF non trovato: {pdf_path}")

        logger.info(f"Parsing: {pdf_path.name}")

        sections = []
        current_section = self._new_section(source=pdf_path.name)
        section_hierarchy = []

        with pdfplumber.open(str(pdf_path)) as pdf:
            logger.debug(f"  Pagine totali: {len(pdf.pages)}")
            avg_fontsize = self._compute_global_avg_fontsize(pdf)

            for page_num, page in enumerate(pdf.pages, start=1):

                # Tabelle
                tables = page.extract_tables() or []
                table_texts = set()
                for table in tables:
                    table_text = self._table_to_text(table)
                    if table_text:
                        current_section["content"].append({
                            "type": "table", "text": table_text, "page": page_num,
                        })
                        current_section["has_table"] = True
                        table_texts.add(table_text[:50])

                # Testo
                raw_text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                lines = raw_text.split("\n")

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    if any(line[:50] in t for t in table_texts):
                        continue

                    line_fontsize = self._get_line_fontsize(page, line)

                    if _is_title_by_pattern(line) or _is_title_by_font(line_fontsize, avg_fontsize):
                        if self._section_has_content(current_section):
                            current_section["section_path"] = _build_section_path(section_hierarchy)
                            sections.append(current_section)

                        section_hierarchy = self._update_hierarchy(
                            section_hierarchy, line, line_fontsize, avg_fontsize
                        )
                        current_section = self._new_section(
                            source=pdf_path.name, title=line, page=page_num,
                        )
                        current_section["content"].append({
                            "type": "title", "text": _normalize_text(line), "page": page_num,
                        })

                    elif _is_bullet(line):
                        current_section["content"].append({
                            "type": "bullet", "text": _normalize_text(line), "page": page_num,
                        })
                        current_section["has_list"] = True

                    else:
                        current_section["content"].append({
                            "type": "paragraph", "text": _normalize_text(line), "page": page_num,
                        })

        if self._section_has_content(current_section):
            current_section["section_path"] = _build_section_path(section_hierarchy)
            sections.append(current_section)

        documents = self._sections_to_documents(sections)
        logger.info(f"  Document estratti: {len(documents)}")
        return documents

    def parse_directory(self, dir_path) -> list:
        dir_path = Path(dir_path)
        pdf_files = sorted(dir_path.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"Nessun PDF trovato in: {dir_path}")
            return []
        logger.info(f"Trovati {len(pdf_files)} PDF in {dir_path}")
        all_documents = []
        for pdf_file in pdf_files:
            try:
                docs = self.parse(pdf_file)
                all_documents.extend(docs)
            except Exception as e:
                logger.error(f"Errore parsing {pdf_file.name}: {e}")
        logger.info(f"Totale Document estratti: {len(all_documents)}")
        return all_documents

    def _new_section(self, source="", title="", page=None):
        return {
            "source": source, "section_title": title, "section_path": "",
            "page_number": page, "content": [], "has_table": False, "has_list": False,
        }

    def _section_has_content(self, section: dict) -> bool:
        total_text = " ".join(
            c["text"] for c in section["content"] if c["type"] != "title"
        )
        return len(total_text.strip()) >= self.min_text_length

    def _sections_to_documents(self, sections: list) -> list:
        documents = []
        for section in sections:
            parts = []
            for item in section["content"]:
                if item["type"] == "title":
                    parts.append(f"\n## {item['text']}\n")
                elif item["type"] == "bullet":
                    parts.append(f"  • {item['text']}")
                elif item["type"] == "table":
                    parts.append(f"\n[TABELLA]\n{item['text']}\n[/TABELLA]")
                else:
                    parts.append(item["text"])

            full_text = "\n".join(parts).strip()
            if len(full_text) < self.min_text_length:
                continue

            metadata = {
                "source": section["source"],
                "section_title": section["section_title"],
                "section_path": section["section_path"],
                "page_number": section["page_number"] or 0,
                "has_table": section["has_table"],
                "has_list": section["has_list"],
                "char_count": len(full_text),
            }
            documents.append(Document(text=full_text, metadata=metadata))
        return documents

    def _table_to_text(self, table: list) -> str:
        if not table:
            return ""
        rows = []
        for row in table:
            cells = [str(cell).strip() if cell else "" for cell in row]
            rows.append(" | ".join(cells))
        return "\n".join(rows)

    def _compute_global_avg_fontsize(self, pdf) -> float:
        sizes = []
        for page in pdf.pages[:min(5, len(pdf.pages))]:
            for char in page.chars:
                if char.get("size"):
                    sizes.append(char["size"])
        return sum(sizes) / len(sizes) if sizes else 12.0

    def _get_line_fontsize(self, page, line: str) -> Optional[float]:
        line_clean = line.strip()[:20]
        for char in page.chars:
            if char.get("text") and char["text"] in line_clean:
                return char.get("size")
        return None

    def _update_hierarchy(self, hierarchy, title, fontsize, avg_fontsize):
        title_clean = _normalize_text(title)
        if not hierarchy:
            return [title_clean]
        if fontsize and fontsize > avg_fontsize * 1.3:
            return [title_clean]
        if len(hierarchy) < 3:
            return hierarchy + [title_clean]
        return hierarchy[:-1] + [title_clean]


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pdf_parser.py <path_to_pdf>")
        sys.exit(1)

    parser = PDFParser()
    docs = parser.parse(sys.argv[1])

    print(f"\n{'='*60}")
    print(f"Documenti estratti: {len(docs)}")
    print(f"{'='*60}\n")

    for i, doc in enumerate(docs[:5]):
        print(f"--- Document {i+1} ---")
        print(f"Sezione    : {doc.metadata.get('section_path', 'N/A')}")
        print(f"Pagina     : {doc.metadata.get('page_number', 'N/A')}")
        print(f"Tabelle    : {doc.metadata.get('has_table', False)}")
        print(f"Liste      : {doc.metadata.get('has_list', False)}")
        print(f"Testo (200): {doc.text[:200]}...")
        print()
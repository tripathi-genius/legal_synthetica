import re
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import Optional


CASE_HEADER_RE = re.compile(
    r"=+\s*\n#\s*Case ID:\s*(?P<case_id>\S+)\s*\n=+\s*\n",
    re.MULTILINE,
)

SECTION_HEADING_RE = re.compile(
    r"^###\s*(?P<num>[1-6])\.\s*[^\n]*\n",
    re.MULTILINE,
)

SECTION_NAMES = {
    1: "law",
    2: "arguments_for",
    3: "arguments_against",
    4: "acceptance_logic",
    5: "rejection_logic",
    6: "final_conclusion",
}

REQUIRED_KEYS = ["facts", "law", "arguments_for", "arguments_against",
                 "acceptance_logic", "rejection_logic", "final_conclusion"]


# --------------------------------------------------------------------------- #
# Text cleanup helpers
# --------------------------------------------------------------------------- #

def normalize_unicode(text: str) -> str:
    """NFKC-normalize and tidy a few typographic characters that otherwise
    fragment tokenizers inconsistently across a large corpus."""
    text = unicodedata.normalize("NFKC", text)
    # NFKC turns U+2011 (non-breaking hyphen) into U+2010 (hyphen), not ASCII '-',
    # so both must be handled post-normalization.
    text = text.replace("\u2010", "-").replace("\u2011", "-")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def strip_hr_and_ws(text: str) -> str:
    """Remove leading/trailing horizontal-rule lines ('---', '===') and
    surrounding whitespace left over from slicing between headings."""
    lines = text.strip("\n").split("\n")
    while lines and re.match(r"^[-=]{3,}\s*$", lines[0].strip()):
        lines.pop(0)
    while lines and re.match(r"^[-=]{3,}\s*$", lines[-1].strip()):
        lines.pop()
    return "\n".join(lines).strip()


def strip_leading_h2(text: str, heading_text: str) -> str:
    stripped = text.lstrip("\n")
    if stripped.lower().startswith(f"## {heading_text}".lower()):
        nl = stripped.find("\n")
        stripped = stripped[nl + 1:] if nl != -1 else ""
    return stripped.strip()


def _is_table_separator(line: str) -> bool:
    # Strip ALL pipe characters (not just the outer ones) before checking,
    # since multi-column separator rows like "|---|---|" have interior pipes.
    core = line.strip().replace("|", "").replace(" ", "")
    return bool(core) and set(core) <= set("-:")


def _is_table_row(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and s.endswith("|") and len(s) > 1


def markdown_table_to_bullets(text: str) -> str:
    """Convert every contiguous 2+-column markdown table in `text` into a
    bulleted list: '- **Col1** — Col2 [ | Col3 ...]'. Non-table lines are
    passed through unchanged."""
    lines = text.split("\n")
    out = []
    i, n = 0, len(lines)
    while i < n:
        if _is_table_row(lines[i]):
            block = []
            while i < n and (_is_table_row(lines[i]) or _is_table_separator(lines[i])):
                block.append(lines[i])
                i += 1
            content_rows = [l for l in block if not _is_table_separator(l)]
            if len(content_rows) >= 2:  # header + at least one data row
                for row in content_rows[1:]:
                    cells = [c.strip() for c in row.strip().strip("|").split("|")]
                    cells = [re.sub(r"<br\s*/?>", "\n  ", c) for c in cells]
                    if not any(cells):
                        continue
                    head = re.sub(r"\*\*(.+?)\*\*", r"\1", cells[0]).strip()
                    rest = " — ".join(c for c in cells[1:] if c)
                    if head and rest:
                        out.append(f"- **{head}** — {rest}")
                    elif head:
                        out.append(f"- {head}")
            else:
                out.extend(block)  # not really a table, keep as-is
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out).strip()


def clean_prose(text: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Core split / parse
# --------------------------------------------------------------------------- #

def split_cases(full_text: str) -> list[tuple[str, str]]:
    """Return [(case_id, case_body), ...] for every case found in the file."""
    matches = list(CASE_HEADER_RE.finditer(full_text))
    cases = []
    for i, m in enumerate(matches):
        case_id = m.group("case_id").strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        body = full_text[start:end]
        cases.append((case_id, body))
    return cases


@dataclass
class ParsedCase:
    case_id: str
    facts: str = ""
    law: str = ""
    arguments_for: str = ""
    arguments_against: str = ""
    acceptance_logic: str = ""
    rejection_logic: str = ""
    final_conclusion: str = ""
    result_short: Optional[str] = None
    missing_sections: list = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return len(self.missing_sections) == 0


def extract_result_short(conclusion_text: str) -> Optional[str]:
    m = re.search(r"\*\*Result:?\*\*\s*(.+?)(?=\n\s*\n|\Z)", conclusion_text, re.DOTALL)
    if not m:
        m = re.search(r"\*\*Holding:?\*\*\s*(.+?)(?=\n\s*\n|\Z)", conclusion_text, re.DOTALL)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    stripped = conclusion_text.strip()
    if not stripped:
        return None
    parts = re.split(r"(?<=[.!?])\s+", stripped)
    return parts[0].strip() if parts else None


def parse_case(case_id: str, body: str) -> ParsedCase:
    body = normalize_unicode(body)
    heading_matches = list(SECTION_HEADING_RE.finditer(body))

    pc = ParsedCase(case_id=case_id)

    facts_end = heading_matches[0].start() if heading_matches else len(body)
    facts_raw = body[:facts_end]
    facts_raw = strip_leading_h2(facts_raw, "Micro Case Facts")
    pc.facts = strip_hr_and_ws(clean_prose(facts_raw))

    found_nums = set()
    for i, m in enumerate(heading_matches):
        num = int(m.group("num"))
        found_nums.add(num)
        seg_start = m.end()
        seg_end = heading_matches[i + 1].start() if i + 1 < len(heading_matches) else len(body)
        raw_section = body[seg_start:seg_end]
        raw_section = strip_hr_and_ws(raw_section)

        if num in (1, 2, 3):
            raw_section = markdown_table_to_bullets(raw_section)
        raw_section = clean_prose(raw_section)

        setattr(pc, SECTION_NAMES[num], raw_section)

    pc.missing_sections = [
        SECTION_NAMES[n] for n in range(1, 7) if n not in found_nums
    ] + (["facts"] if not pc.facts.strip() else [])

    if pc.final_conclusion:
        pc.result_short = extract_result_short(pc.final_conclusion)

    return pc


def load_and_parse_all(full_text: str) -> tuple[list[ParsedCase], dict]:
    raw_cases = split_cases(full_text)
    parsed = []
    stats = {"total_found": len(raw_cases), "complete": 0, "incomplete": 0,
              "missing_section_counts": {}, "incomplete_case_ids": []}
    for case_id, body in raw_cases:
        pc = parse_case(case_id, body)
        parsed.append(pc)
        if pc.is_complete:
            stats["complete"] += 1
        else:
            stats["incomplete"] += 1
            stats["incomplete_case_ids"].append(case_id)
            for s in pc.missing_sections:
                stats["missing_section_counts"][s] = stats["missing_section_counts"].get(s, 0) + 1
    return parsed, stats

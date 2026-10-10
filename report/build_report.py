"""
Builds the B.Tech project report from the university template (Report.docx).

    .venv/Scripts/python report/build_report.py [--force]     # -> report/MetaDerm-XA_Report.docx

WARNING: this regenerates the whole document. Once the user edits the .docx by hand, do NOT rebuild;
patch the user's file in place. The script refuses to overwrite a file edited after the last build
unless --force is given (only with the user's explicit OK).

* front matter: title, guide designation, panel members, student names, abstract, typo fixes
* CONTENTS / LIST OF TABLES / LIST OF FIGURES / LIST OF ABBREVIATIONS as tables in the template's
  3-column format; page numbers are PAGEREF fields to bookmarks (Word updates them on open)
* chapters from report/content.py; chapter number and chapter title on separate lines
* page numbers: cover none; page 2 .. end of front matter lower roman starting at i;
  Introduction onwards arabic starting at 1
"""

import hashlib
import re
import shutil
import struct
import sys
import tempfile
import zipfile
from html import escape, unescape
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "Report.docx"
OUT = ROOT / "report" / "MetaDerm-XA_Report.docx"

TITLE = "MetaDerm-XA: Metadata-Aware and Explainable Melanoma Detection under Dataset Shift"
GUIDE = "Kolluru Venkata Anusha"
GUIDE_DESIGNATION = "Assistant Professor"
STUDENTS = ["Kiran Kumar Gedela", "Tanneru Preethi", "Gantasala Sri Hari Subhash Guptha", "Aggnour Abhilash"]
PANEL = ("the members of the project review panel, Dr. Balaji T. K., Dr. Ravi Sankar Puppala and "
         "Dr. K. G. Raghavendra Narayan, Assistant Professors, School of Computing,")
UNNUMBERED = {"REFERENCES", "LIST OF PUBLICATIONS", "PLAGIARISM REPORT"}

# text fixes applied to whole paragraphs (joined across runs, unescaped)
FRONT_FIXES = [
    ("<Title of the Project>", TITLE),
    ("Vishwa Vidypeetham", "Vishwa Vidyapeetham"),
    ("amrit a vishwa", "amrita vishwa"),
    (f"{GUIDE}, Designation, Computer Science and Engineering-Artificial Intelligence",
     f"{GUIDE}, {GUIDE_DESIGNATION}, School of Computing"),
    (f"{GUIDE}, Designation, Amrita School of Computing", f"{GUIDE}, {GUIDE_DESIGNATION}, Amrita School of Computing"),
    ("project guide  ", "project guide "),
    ("<all Panel members Name, Designation>", PANEL),
    ("which helped me to complete", "which helped us to complete"),
    ("and  has not formed", "and has not formed"),
]


# ----------------------------------------------------------------------------- small XML helpers
def run(text, bold=False, italic=False, size=None):
    rpr = ""
    if bold or italic or size:
        rpr = "<w:rPr>" + ("<w:b/>" if bold else "") + ("<w:i/>" if italic else "") + \
              (f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>' if size else "") + "</w:rPr>"
    return f'<w:r>{rpr}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def rich(text):
    """**bold** and *italic* inline markup → runs."""
    out = []
    for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text):
        if part.startswith("**"):
            out.append(run(part[2:-2], bold=True))
        elif part.startswith("*") and len(part) > 1:
            out.append(run(part[1:-1], italic=True))
        elif part:
            out.append(run(part))
    return "".join(out)


def para(text, style="BodyText", align="both", page_break=False, keep_next=False, runs=None):
    ppr = f'<w:pStyle w:val="{style}"/>' + ("<w:keepNext/>" if keep_next else "") + \
          ("<w:pageBreakBefore/>" if page_break else "") + '<w:spacing w:after="120" w:line="360" w:lineRule="auto"/>' + \
          (f'<w:jc w:val="{align}"/>' if align else "")
    return f"<w:p><w:pPr>{ppr}</w:pPr>{runs if runs is not None else rich(text)}</w:p>"


def bullet(text):
    return (f'<w:p><w:pPr><w:pStyle w:val="ListParagraph"/><w:ind w:left="720" w:hanging="360"/>'
            f'<w:spacing w:after="80" w:line="360" w:lineRule="auto"/><w:jc w:val="both"/></w:pPr>'
            f'{run(chr(8226) + chr(9))}{rich(text)}</w:p>')


def page_break_para():
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


def front_heading(text, bookmark):
    """Centered bold 16 pt heading, same look as the template's CONTENTS heading."""
    return (f'<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:after="240"/></w:pPr>'
            f'<w:bookmarkStart w:id="{bookmark[0]}" w:name="{bookmark[1]}"/>'
            f'<w:r><w:rPr><w:b/><w:bCs/><w:sz w:val="32"/><w:szCs w:val="32"/></w:rPr><w:t>{escape(text)}</w:t></w:r>'
            f'<w:bookmarkEnd w:id="{bookmark[0]}"/></w:p>')


class Book:
    """Bookmarks + entries for the contents / list tables."""

    def __init__(self):
        self.n = 0
        self.contents = []  # (col1, title, bookmark, bold)
        self.tables = []    # (label, title, bookmark)
        self.figures = []
        self.chapter = 0
        self.section = 0
        self.tab_k = 0
        self.fig_k = 0

    def new(self):
        self.n += 1
        return self.n, f"mdx_ref_{self.n}"

    @staticmethod
    def wrap(xml, bm):
        """Put a bookmark around the runs of a paragraph."""
        i = xml.find("</w:pPr>") + len("</w:pPr>")
        return xml[:i] + f'<w:bookmarkStart w:id="{bm[0]}" w:name="{bm[1]}"/>' + xml[i:-len("</w:p>")] + \
            f'<w:bookmarkEnd w:id="{bm[0]}"/></w:p>'


BOOK = Book()


def _strip_num(title):
    """Heading styles auto-number ('CHAPTER n', '1.1'), so manual numbers are dropped."""
    return re.sub(r"^(CHAPTER \d+:\s*|\d+(\.\d+)*\s+)", "", title)


def heading(title, style, page_break=False):
    clean = _strip_num(title)
    xml = para(clean, style=style, align="center" if style == "Heading1" else "left",
               page_break=page_break, keep_next=True)
    bm = BOOK.new()
    if style == "Heading1":
        if title in UNNUMBERED:  # numId 0 switches the style's chapter numbering off
            xml = xml.replace("<w:keepNext/>", '<w:keepNext/><w:numPr><w:ilvl w:val="0"/><w:numId w:val="0"/></w:numPr>', 1)
            BOOK.contents.append(("", clean.upper(), bm[1], True))
        else:  # "CHAPTER n" (auto) on one line, title on the next
            xml = xml.replace("</w:pPr>", "</w:pPr><w:r><w:br/></w:r>", 1)
            BOOK.chapter += 1
            BOOK.section = BOOK.tab_k = BOOK.fig_k = 0
            BOOK.contents.append((str(BOOK.chapter), clean.upper(), bm[1], True))
    elif style == "Heading2":
        BOOK.section += 1
        BOOK.contents.append((f"{BOOK.chapter}.{BOOK.section}", clean, bm[1], False))
    return BOOK.wrap(xml, bm)


def caption(kind, text):
    if kind == "Table":
        BOOK.tab_k += 1
        label = f"Table {BOOK.chapter}.{BOOK.tab_k}"
    else:
        BOOK.fig_k += 1
        label = f"Figure {BOOK.chapter}.{BOOK.fig_k}"
    bm = BOOK.new()
    (BOOK.tables if kind == "Table" else BOOK.figures).append((label.split()[1], text, bm[1]))
    xml = (f'<w:p><w:pPr><w:pStyle w:val="Caption"/><w:keepNext/><w:jc w:val="center"/></w:pPr>'
           f'{run(label + ": ", bold=True)}{run(text)}</w:p>')
    return BOOK.wrap(xml, bm)


def table(df: pd.DataFrame, title: str, widths=None):
    cols = list(df.columns)
    widths = widths or [9000 // len(cols)] * len(cols)
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)

    def cell(t, w, head=False):
        shade = '<w:shd w:val="clear" w:color="auto" w:fill="D9E2F3"/>' if head else ""
        return (f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{shade}</w:tcPr>'
                f'<w:p><w:pPr><w:pStyle w:val="TableContents"/><w:jc w:val="center"/></w:pPr>'
                f'{run(str(t), bold=head, size=20)}</w:p></w:tc>')

    rows = '<w:tr><w:trPr><w:tblHeader/></w:trPr>' + "".join(cell(c, w, True) for c, w in zip(cols, widths)) + "</w:tr>"
    for _, r in df.iterrows():
        rows += "<w:tr>" + "".join(cell(r[c], w) for c, w in zip(cols, widths)) + "</w:tr>"
    border = '<w:{0} w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
    borders = "".join(border.format(b) for b in ["top", "left", "bottom", "right", "insideH", "insideV"])
    return (caption("Table", title) +
            f'<w:tbl><w:tblPr><w:tblW w:w="{sum(widths)}" w:type="dxa"/><w:jc w:val="center"/>'
            f'<w:tblBorders>{borders}</w:tblBorders></w:tblPr><w:tblGrid>{grid}</w:tblGrid>{rows}</w:tbl>'
            + para(""))


class Images:
    """Registers PNG/JPEG files as media + relationships and returns inline drawing XML."""

    def __init__(self, unpacked: Path):
        self.dir = unpacked
        self.n = 0
        self.rels = (unpacked / "word/_rels/document.xml.rels").read_text(encoding="utf8")

    @staticmethod
    def _size(p: Path):
        b = p.read_bytes()
        if b[:8] == b"\x89PNG\r\n\x1a\n":
            return struct.unpack(">II", b[16:24])
        from PIL import Image
        return Image.open(p).size

    def add(self, path: Path, width_in=6.0, title="figure"):
        self.n += 1
        ext = path.suffix.lower().lstrip(".")
        name = f"metaderm_{self.n}.{ext}"
        shutil.copy(path, self.dir / "word/media" / name)
        rid = f"rIdMeta{self.n}"
        self.rels = self.rels.replace("</Relationships>",
            f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
            f'Target="media/{name}"/></Relationships>')
        w, h = self._size(path)
        cx = int(width_in * 914400)
        cy = int(cx * h / w)
        return (f'<w:p><w:pPr><w:keepNext/><w:jc w:val="center"/></w:pPr><w:r><w:drawing>'
                f'<wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
                f'<wp:docPr id="{900 + self.n}" name="{escape(title)}"/>'
                f'<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
                f'<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
                f'<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
                f'<pic:nvPicPr><pic:cNvPr id="{900 + self.n}" name="{name}"/><pic:cNvPicPr/></pic:nvPicPr>'
                f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
                f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
                f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic>'
                f'</wp:inline></w:drawing></w:r></w:p>')

    def save(self):
        (self.dir / "word/_rels/document.xml.rels").write_text(self.rels, encoding="utf8")


# ----------------------------------------------------------------------------- chapters
def render(blocks, images: Images) -> str:
    xml = []
    for b in blocks:
        kind = b[0]
        if kind == "chapter":
            xml.append(heading(b[1], "Heading1", page_break=True))
        elif kind == "h2":
            xml.append(heading(b[1], "Heading2"))
        elif kind == "h3":
            xml.append(heading(b[1], "Heading3"))
        elif kind == "p":
            xml.append(para(b[1]))
        elif kind == "bullets":
            xml.extend(bullet(t) for t in b[1])
        elif kind == "table":
            xml.append(table(b[2], b[1], b[3] if len(b) > 3 else None))
        elif kind == "csvtable":  # ('csvtable', caption, path, columns|None)
            p = ROOT / b[2]
            if p.exists():
                df = pd.read_csv(p).fillna("--")
                df = df[b[3]] if len(b) > 3 and b[3] else df
                xml.append(table(df.astype(str).replace(r"\$\\pm\$", "±", regex=True), b[1]))
            else:
                xml.append(para(f"[PENDING v2 RESULTS: {b[1]}]"))
        elif kind == "figure":  # ('figure', caption, path, width_in)
            p = ROOT / b[2]
            if p.exists():
                xml.append(images.add(p, b[3] if len(b) > 3 else 6.0, b[1]))
                xml.append(caption("Figure", b[1]))
            else:
                xml.append(para(f"[PENDING FIGURE: {b[1]} ({b[2]})]"))
        elif kind == "ref":
            xml.append(para(b[1], align="left"))
        else:
            raise ValueError(kind)
    return "".join(xml)


# ----------------------------------------------------------------------------- front-matter tables
COLS = (1581, 6607, 1134)  # template contents table


def _cell(content, w, bold=False, center=False):
    jc = '<w:jc w:val="center"/>' if center else ""
    return (f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/><w:shd w:val="clear" w:color="auto" w:fill="FFFFFF"/></w:tcPr>'
            f'<w:p><w:pPr><w:spacing w:before="120" w:after="120" w:line="240" w:lineRule="auto"/>'
            f'{jc}</w:pPr>{content}</w:p></w:tc>')


def _txt(t, bold=False):
    return f'<w:r><w:rPr>{"<w:b/>" if bold else ""}<w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr><w:t xml:space="preserve">{escape(t)}</w:t></w:r>'


def _pageref(bookmark, bold=False):
    b = "<w:b/>" if bold else ""
    rpr = f"<w:rPr>{b}<w:sz w:val=\"24\"/><w:szCs w:val=\"24\"/></w:rPr>"
    return (f'<w:r>{rpr}<w:fldChar w:fldCharType="begin" w:dirty="true"/></w:r>'
            f'<w:r>{rpr}<w:instrText xml:space="preserve"> PAGEREF {bookmark} \\h </w:instrText></w:r>'
            f'<w:r>{rpr}<w:fldChar w:fldCharType="separate"/></w:r><w:r>{rpr}<w:t>#</w:t></w:r>'
            f'<w:r>{rpr}<w:fldChar w:fldCharType="end"/></w:r>')


def _grid_table(header, rows):
    border = '<w:{0} w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
    borders = "".join(border.format(b) for b in ["top", "left", "bottom", "right", "insideH", "insideV"])
    head = '<w:tr><w:trPr><w:tblHeader/><w:trHeight w:val="584"/></w:trPr>' + "".join(
        _cell(_txt(h, True), w, True, center=i > 0) for i, (h, w) in enumerate(zip(header, COLS))) + "</w:tr>"
    body = ""
    for c1, title, bm, bold in rows:
        body += ("<w:tr><w:trPr><w:cantSplit/></w:trPr>" + _cell(_txt(c1, bold), COLS[0], center=True) +
                 _cell(_txt(title, bold), COLS[1]) + _cell(_pageref(bm, bold), COLS[2], center=True) + "</w:tr>")
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in COLS)
    return (f'<w:tbl><w:tblPr><w:tblW w:w="{sum(COLS)}" w:type="dxa"/><w:jc w:val="center"/>'
            f'<w:tblBorders>{borders}</w:tblBorders><w:tblCellMar><w:left w:w="118" w:type="dxa"/></w:tblCellMar>'
            f'</w:tblPr><w:tblGrid>{grid}</w:tblGrid>{head}{body}</w:tbl>')


def front_matter_tables(front_bm, abbreviations):
    """CONTENTS, LIST OF TABLES, LIST OF FIGURES, LIST OF ABBREVIATIONS (each on a new page)."""
    contents_rows = [("", "ABSTRACT", front_bm["ABSTRACT"], True),
                     ("", "LIST OF TABLES", front_bm["LIST OF TABLES"], True),
                     ("", "LIST OF FIGURES", front_bm["LIST OF FIGURES"], True),
                     ("", "LIST OF ABBREVIATIONS", front_bm["LIST OF ABBREVIATIONS"], True)] + BOOK.contents
    xml = front_heading("CONTENTS", BOOK.new()) + _grid_table(("CHAPTER", "TITLE", "PAGE NOs"), contents_rows)
    xml += page_break_para() + front_heading("LIST OF TABLES", front_bm["_LOT"])
    xml += _grid_table(("TABLE NO.", "TITLE", "PAGE NO."), [(a, t, b, False) for a, t, b in BOOK.tables])
    xml += page_break_para() + front_heading("LIST OF FIGURES", front_bm["_LOF"])
    xml += _grid_table(("FIGURE NO.", "TITLE", "PAGE NO."), [(a, t, b, False) for a, t, b in BOOK.figures])
    xml += page_break_para() + front_heading("LIST OF ABBREVIATIONS", front_bm["_LOA"])
    border = '<w:{0} w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
    borders = "".join(border.format(b) for b in ["top", "left", "bottom", "right", "insideH", "insideV"])
    rows = '<w:tr><w:trPr><w:tblHeader/></w:trPr>' + _cell(_txt("ABBREVIATION", True), 2600, True, True) + \
        _cell(_txt("EXPANSION", True), 6722, True, True) + "</w:tr>"
    for a, e in abbreviations:
        rows += "<w:tr>" + _cell(_txt(a), 2600) + _cell(_txt(e), 6722) + "</w:tr>"
    xml += (f'<w:tbl><w:tblPr><w:tblW w:w="9322" w:type="dxa"/><w:jc w:val="center"/><w:tblBorders>{borders}</w:tblBorders>'
            f'<w:tblCellMar><w:left w:w="118" w:type="dxa"/></w:tblCellMar></w:tblPr>'
            f'<w:tblGrid><w:gridCol w:w="2600"/><w:gridCol w:w="6722"/></w:tblGrid>{rows}</w:tbl>')
    return xml


# ----------------------------------------------------------------------------- page numbers
FOOTER_XML = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
              '<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
              'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
              '<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:fldChar w:fldCharType="begin"/></w:r>'
              '<w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r><w:r><w:fldChar w:fldCharType="separate"/></w:r>'
              '<w:r><w:t>1</w:t></w:r><w:r><w:fldChar w:fldCharType="end"/></w:r></w:p></w:ftr>')
FOOTER_REF = '<w:footerReference w:type="default" r:id="rIdMetaFooter"/>'


def _section(sect: str, fmt=None, start=None, next_page=False, footer=True) -> str:
    """Set footer / page-number format of a <w:sectPr> (schema order: footerReference first, pgNumType after pgMar)."""
    sect = re.sub(r"<w:footerReference[^>]*/>|<w:pgNumType[^>]*/>|<w:titlePg/>", "", sect)
    if next_page:
        sect = sect.replace('<w:type w:val="continuous"/>', "")
    if footer:
        sect = re.sub(r"(<w:sectPr[^>]*>)", r"\1" + FOOTER_REF, sect, count=1)
    if fmt:
        num = f'<w:pgNumType w:fmt="{fmt}"' + (f' w:start="{start}"' if start else "") + "/>"
        sect = re.sub(r"(<w:pgMar[^>]*/>)", r"\1" + num, sect, count=1)
    return sect


def page_numbers(doc: str, tmp: Path) -> str:
    sects = list(re.finditer(r"<w:sectPr.*?</w:sectPr>", doc, flags=re.S))
    first, final = sects[0], sects[-1]
    # end of front matter (after LIST OF ABBREVIATIONS table) → new section; body restarts at 1
    marker = doc.find("<!--END_FRONT-->")
    front = _section(final.group(0), "lowerRoman")
    body = _section(final.group(0), "decimal", start=1, next_page=True)
    doc = doc[:final.start()] + body + doc[final.end():]
    doc = doc.replace("<!--END_FRONT-->", f"<w:p><w:pPr>{front}</w:pPr></w:p>", 1)
    assert marker > 0
    # middle sections (signature columns etc.) continue roman
    sects = list(re.finditer(r"<w:sectPr.*?</w:sectPr>", doc, flags=re.S))
    for m in reversed(sects[1:-2]):
        doc = doc[:m.start()] + _section(m.group(0), "lowerRoman") + doc[m.end():]
    # first section (inner cover + certificate) starts roman at i ...
    m = sects[0]
    doc = doc[:m.start()] + _section(m.group(0), "lowerRoman", start=1) + doc[m.end():]
    # ... and the outer cover (page 1) becomes its own section without a page number
    pb = doc.find('<w:br w:type="page"/>')
    p_start = doc.rfind("<w:p ", 0, pb)
    p_start = max(p_start, doc.rfind("<w:p>", 0, pb))
    cover_sect = _section(first.group(0), footer=False)
    doc = doc[:pb] + doc[pb + len('<w:br w:type="page"/>'):]  # the section break replaces the page break
    doc = doc[:p_start] + f"<w:p><w:pPr>{cover_sect}</w:pPr></w:p>" + doc[p_start:]

    (tmp / "word/footer_meta.xml").write_text(FOOTER_XML, encoding="utf8")
    rels = tmp / "word/_rels/document.xml.rels"
    r = rels.read_text(encoding="utf8")
    if "rIdMetaFooter" not in r:
        rels.write_text(r.replace("</Relationships>",
            '<Relationship Id="rIdMetaFooter" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" '
            'Target="footer_meta.xml"/></Relationships>'), encoding="utf8")
    ct = tmp / "[Content_Types].xml"
    c = ct.read_text(encoding="utf8")
    if "footer_meta.xml" not in c:
        ct.write_text(c.replace("</Types>", '<Override PartName="/word/footer_meta.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/></Types>'), encoding="utf8")
    return doc


# ----------------------------------------------------------------------------- paragraph utilities
def _paragraphs(xml):
    """(start, end, plain text) of every paragraph; text joined across runs and unescaped twice
    (the template stores some placeholders as &amp;lt;...&amp;gt;)."""
    for m in re.finditer(r"<w:p[ >](?:(?!<w:p[ >]).)*?</w:p>", xml, flags=re.S):
        yield m.start(), m.end(), unescape(unescape("".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", m.group(0)))))


def _set_text(par_xml, text):
    """Replace all runs of a paragraph by one run with the first run's formatting."""
    ppr = re.search(r"<w:pPr>.*?</w:pPr>", par_xml, flags=re.S)
    rpr = re.search(r"<w:r[ >].*?(<w:rPr>.*?</w:rPr>)", par_xml, flags=re.S)
    head = re.match(r"<w:p[^>]*>", par_xml).group(0)
    return (head + (ppr.group(0) if ppr else "") +
            f'<w:r>{rpr.group(1) if rpr else ""}<w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>')


def edit_paragraphs(xml, fn):
    """fn(text) -> None (keep) | str (new text) | '' (delete paragraph). Applied back to front."""
    for s, e, t in reversed(list(_paragraphs(xml))):
        new = fn(t)
        if new is None or new == t:
            continue
        xml = xml[:s] + ("" if new == "" else _set_text(xml[s:e], new)) + xml[e:]
    return xml


# ----------------------------------------------------------------------------- build
def build():
    from content import ABBREVIATIONS, ABSTRACT, CHAPTERS  # report/content.py

    tmp = Path(tempfile.mkdtemp())
    zipfile.ZipFile(TEMPLATE).extractall(tmp)
    for p in tmp.rglob("*"):
        if p.is_symlink():
            p.unlink()
    doc = (tmp / "word/document.xml").read_text(encoding="utf8")
    doc = doc.replace('<w:color w:val="FF0000"/>', "")  # template marks placeholders in red
    doc = doc.replace("TITLE OF THE PROJECT", escape(TITLE.upper()))
    for i, s in enumerate(STUDENTS, 1):
        doc = doc.replace(f"Student{i} NAme", escape(s))

    instructions = ("note : instructions", "the dissertation report should", "your guide must verify",
                    "proposed work can be modified", "plagiarism report including ai", "new chapter should start")

    def fix(t):
        low = t.strip().lower()
        if low in ("name of the guide", "with font size 12") or low.startswith(instructions):
            return ""
        if low.startswith("maximum of 250"):
            return None  # replaced below by the abstract
        if low.startswith("designation"):  # certificate signature block: "Designation :" → the designation itself
            return GUIDE_DESIGNATION
        new = t
        for a, b in FRONT_FIXES:
            new = new.replace(a, b)
        return new if new != t else None

    doc = edit_paragraphs(doc, fix)

    # abstract (replaces the "Maximum of 250 Words" instruction), bookmarked for the contents table
    front_bm = {}
    hits = [(s, e) for s, e, t in _paragraphs(doc) if t.strip().lower().startswith("maximum of 250")]
    s, e = hits[0]
    doc = doc[:s] + "".join(para(t) for t in ABSTRACT) + doc[e:]
    abs_hits = [(s, e) for s, e, t in _paragraphs(doc) if t.strip() == "ABSTRACT"]
    s, e = abs_hits[0]
    bm = BOOK.new()
    front_bm["ABSTRACT"] = bm[1]
    doc = doc[:s] + BOOK.wrap(doc[s:e], bm) + doc[e:]

    # chapters first (fills BOOK with contents/table/figure entries)
    images = Images(tmp)
    chapters = render(CHAPTERS, images)

    # contents + lists replace the template's CONTENTS heading and table
    lot, lof, loa = (BOOK.new() for _ in range(3))
    front_bm.update({"_LOT": lot, "_LOF": lof, "_LOA": loa,
                     "LIST OF TABLES": lot[1], "LIST OF FIGURES": lof[1], "LIST OF ABBREVIATIONS": loa[1]})
    c_start = doc.rfind("<w:p ", 0, doc.find(">CONTENTS<"))
    c_end = doc.find("</w:tbl>", c_start) + len("</w:tbl>")
    doc = doc[:c_start] + front_matter_tables(front_bm, ABBREVIATIONS) + "<!--END_FRONT-->" + doc[c_end:]

    last_sect = doc.rfind("<w:sectPr")
    doc = doc[:last_sect] + chapters + doc[last_sect:]
    images.save()
    ct = (tmp / "[Content_Types].xml").read_text(encoding="utf8")
    if 'Extension="png"' not in ct:
        (tmp / "[Content_Types].xml").write_text(
            ct.replace("</Types>", '<Default Extension="png" ContentType="image/png"/></Types>'), encoding="utf8")
    doc = page_numbers(doc, tmp)
    (tmp / "word/document.xml").write_text(doc, encoding="utf8")

    # Word refreshes PAGEREF fields (page numbers in the contents/list tables) when the file opens
    settings = (tmp / "word/settings.xml").read_text(encoding="utf8")
    if "<w:updateFields" not in settings:
        settings = re.sub(r"(<w:zoom[^>]*/>)", r'\1<w:updateFields w:val="true"/>', settings, count=1)
        (tmp / "word/settings.xml").write_text(settings, encoding="utf8")

    stamp = OUT.with_suffix(".lastbuild.sha256")
    if OUT.exists() and "--force" not in sys.argv:
        current = hashlib.sha256(OUT.read_bytes()).hexdigest()
        if not stamp.exists() or stamp.read_text().strip() != current:
            raise SystemExit(f"REFUSING to overwrite {OUT}: it was edited after the last build. "
                             "Edit that file in place instead, or rerun with --force to discard the edits.")
    OUT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(tmp.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(tmp).as_posix())
    stamp.write_text(hashlib.sha256(OUT.read_bytes()).hexdigest())
    print(f"wrote {OUT}  ({images.n} images, {len(BOOK.contents)} contents rows, "
          f"{len(BOOK.tables)} tables, {len(BOOK.figures)} figures)")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    build()

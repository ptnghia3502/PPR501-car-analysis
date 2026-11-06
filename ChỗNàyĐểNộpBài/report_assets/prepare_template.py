"""Prepare a macro-free copy of the Springer LNCS Word proceedings template.

The template shipped in ``docs/Microsoft+Word+Proceedings+Template+ZIP`` is a
``.docm`` file: it contains a VBA project and a custom ribbon that only make
sense inside Word. For the report we need the same styles, numbering
definitions and running headers, but as a plain ``.docx`` that can be edited by
any word processor.

The script extracts ``splnproc2510.docm``, drops the VBA project, the custom UI
and the chart sample, rewrites the relationship files accordingly and writes the
result to ``report_assets/lncs_template``.

Run from the repository root::

    python "ChỗNàyĐểNộpBài/report_assets/prepare_template.py"
"""

from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DOCM = (
    ROOT
    / "docs"
    / "Microsoft+Word+Proceedings+Template+ZIP"
    / "splnproc2510.docm"
)
TARGET = Path(__file__).resolve().parent / "lncs_template"

DROP_PREFIXES = (
    "customUI/",
    "word/charts/",
    "word/vbaProject.bin",
    "word/vbaData.xml",
    "word/_rels/vbaProject.bin.rels",
)


def prepare() -> None:
    if not TEMPLATE_DOCM.is_file():
        raise FileNotFoundError(TEMPLATE_DOCM)

    if TARGET.exists():
        shutil.rmtree(TARGET)
    TARGET.mkdir(parents=True)

    with zipfile.ZipFile(TEMPLATE_DOCM) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        for name in names:
            if name.startswith(DROP_PREFIXES):
                continue
            destination = TARGET / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as source, destination.open("wb") as target:
                target.write(source.read())

    # The package must not advertise a macro-enabled document type.
    content_types = TARGET / "[Content_Types].xml"
    text = content_types.read_text(encoding="utf-8")
    text = re.sub(
        r'<Override PartName="/word/document.xml"[^>]*/>',
        '<Override PartName="/word/document.xml" ContentType="application/vnd.'
        'openxmlformats-officedocument.wordprocessingml.document.main+xml"/>',
        text,
    )
    text = re.sub(
        r'<Override PartName="/word/vbaProject.bin"[^>]*/>', "", text
    )
    text = re.sub(
        r'<Override PartName="/word/vbaData.xml"[^>]*/>', "", text
    )
    text = re.sub(r'<Override PartName="/customXml/[^"]*"[^>]*/>', "", text)
    content_types.write_text(text, encoding="utf-8")

    # The root relationships point at the custom UI and at the macro part.
    root_rels = TARGET / "_rels" / ".rels"
    text = root_rels.read_text(encoding="utf-8")
    text = re.sub(r'<Relationship[^>]*customUI[^>]*/>', "", text)
    text = re.sub(r'<Relationship[^>]*vbaProject[^>]*/>', "", text)
    root_rels.write_text(text, encoding="utf-8")

    document_rels = TARGET / "word" / "_rels" / "document.xml.rels"
    text = document_rels.read_text(encoding="utf-8")
    text = re.sub(r'<Relationship[^>]*vbaProject[^>]*/>', "", text)
    text = re.sub(r'<Relationship[^>]*chart[^>]*/>', "", text)
    document_rels.write_text(text, encoding="utf-8")

    # The sample chart in the template body is replaced by our own content.
    document = TARGET / "word" / "document.xml"
    text = document.read_text(encoding="utf-8")
    text = re.sub(r'<w:drawing>.*?</w:drawing>', "", text, flags=re.S)
    document.write_text(text, encoding="utf-8")

    # Running heads: the author list on even pages, the short title on odd pages.
    even_header = TARGET / "word" / "header1.xml"
    text = even_header.read_text(encoding="utf-8")
    text = text.replace(
        "F. Author and S. Author", "P. T. Nghia, T. C. Vinh and T. Q. Duy"
    )
    even_header.write_text(text, encoding="utf-8")

    odd_header = TARGET / "word" / "header2.xml"
    text = odd_header.read_text(encoding="utf-8")
    text = text.replace(
        "Contribution Title (shortened if too long) ",
        "Support System for Car Recognition, Valuation and Recommendation ",
    )
    odd_header.write_text(text, encoding="utf-8")

    print("template prepared in", TARGET)


if __name__ == "__main__":
    prepare()

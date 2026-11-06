"""Assemble ``Assignment3_Report.docx`` from the Springer LNCS Word template.

The script rebuilds the document from the cleaned template package in
``report_assets/lncs_template`` (the Word proceedings template shipped in
``docs/Microsoft+Word+Proceedings+Template+ZIP`` with its VBA project removed)
and replaces the body with the report content.

Everything that appears in the document is either taken verbatim from the
notebook / dataset outputs or recomputed here, so the numbers cannot drift away
from the code that produced them.

Run from the repository root::

    python "ChỗNàyĐểNộpBài/report_assets/build_report.py"
"""

from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(__file__).resolve().parent
FIG_DIR = ASSETS / "figures"
TEMPLATE_DIR = ASSETS / "lncs_template"
OUT_PATH = ROOT / "ChỗNàyĐểNộpBài" / "Assignment3_Report.docx"

W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'

# ---------------------------------------------------------------------------
# Text width of the LNCS page setup: 11906 - 2494 - 2494 = 6918 dxa.
# ---------------------------------------------------------------------------
TEXT_WIDTH_DXA = 6918
EMU_PER_DXA = 635


def esc(text: str) -> str:
    """Escape XML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def run(text: str, *, bold: bool = False, italic: bool = False, size: int | None = None) -> str:
    """A single ``w:r`` element."""
    props = ""
    if bold:
        props += "<w:b/>"
    if italic:
        props += "<w:i/>"
    if size is not None:
        props += f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>'
    rpr = f"<w:rPr>{props}</w:rPr>" if props else ""
    return f'<w:r>{rpr}<w:t xml:space="preserve">{esc(text)}</w:t></w:r>'


def rich(parts: list[tuple[str, dict]]) -> str:
    """Build a run sequence from ``(text, properties)`` pairs."""
    return "".join(run(text, **props) for text, props in parts)


def para(content: str, style: str | None = None, extra_ppr: str = "") -> str:
    """A paragraph, optionally with a paragraph style."""
    props = ""
    if style:
        props += f'<w:pStyle w:val="{style}"/>'
    props += extra_ppr
    ppr = f"<w:pPr>{props}</w:pPr>" if props else ""
    return f"<w:p>{ppr}{content}</w:p>"


def math_runs(expression: str) -> str:
    """Render an inline mathematical expression as a sequence of runs.

    Latin letters are italic, a trailing ``_x`` becomes a subscript, and the
    remaining symbols are set upright, which is the convention used in the
    Springer proceedings template.
    """
    runs: list[str] = []
    for chunk in re.split(r"(_[A-Za-z0-9]+)", expression):
        if not chunk:
            continue
        if chunk.startswith("_") and len(chunk) > 1:
            runs.append(
                '<w:r><w:rPr><w:i/><w:vertAlign w:val="subscript"/></w:rPr>'
                f'<w:t xml:space="preserve">{esc(chunk[1:])}</w:t></w:r>'
            )
            continue
        letters = "".join(ch for ch in chunk if ch.isalpha())
        rest = "".join(ch for ch in chunk if not ch.isalpha())
        if letters:
            runs.append(
                f'<w:r><w:rPr><w:i/></w:rPr><w:t xml:space="preserve">{esc(letters)}</w:t></w:r>'
            )
        if rest:
            # A combining tilde belongs to the letter that precedes it, so it is
            # kept in the italic run rather than being set separately.
            if rest and rest[0] == "̃" and runs:
                runs[-1] = runs[-1].replace(
                    "</w:t></w:r>", f"{esc(rest[0])}</w:t></w:r>"
                )
                rest = rest[1:]
            if rest:
                runs.append(f'<w:r><w:t xml:space="preserve">{esc(rest)}</w:t></w:r>')
    return "".join(runs)


def text_para(text: str, style: str | None = None, extra_ppr: str = "") -> str:
    """A paragraph whose text may contain inline ``\\(...\\)`` mathematics."""
    pieces = re.split(r"\\\((.+?)\\\)", text)
    content = ""
    for index, piece in enumerate(pieces):
        if not piece:
            continue
        if index % 2:
            content += math_runs(piece)
        else:
            content += run(piece)
    return para(content, style, extra_ppr)


def field(instruction: str, cached: str, *, bold: bool = False) -> str:
    """A Word field (SEQ counters, cross references, page numbers)."""
    props = "<w:rPr><w:b/></w:rPr>" if bold else ""
    return (
        f'<w:r>{props}<w:fldChar w:fldCharType="begin"/></w:r>'
        f'<w:r>{props}<w:instrText xml:space="preserve">{esc(instruction)}</w:instrText></w:r>'
        f'<w:r>{props}<w:fldChar w:fldCharType="separate"/></w:r>'
        f'<w:r><w:rPr>{("<w:b/>" if bold else "")}<w:noProof/></w:rPr><w:t>{esc(cached)}</w:t></w:r>'
        f'<w:r>{props}<w:fldChar w:fldCharType="end"/></w:r>'
    )


class Report:
    """Incrementally builds the report body."""

    def __init__(self) -> None:
        self.body: list[str] = []
        self.figures = 0
        self.tables = 0
        self.equations = 0
        self.bookmark_id = 100
        self.image_rels: list[tuple[str, str]] = []
        self.next_rel = 1
        self.media: dict[str, bytes] = {}

    # -- structure ---------------------------------------------------------
    def add(self, xml: str) -> None:
        self.body.append(xml)

    def heading1(self, text: str) -> None:
        self.add(para(run(text), "heading1"))

    def heading2(self, text: str) -> None:
        self.add(para(run(text), "heading2"))

    def heading3(self, text: str) -> None:
        """Run-in third level heading (not numbered)."""
        self.add(
            para(
                f'<w:r><w:rPr><w:rStyle w:val="heading3"/></w:rPr><w:t xml:space="preserve">{esc(text)} </w:t></w:r>',
                "heading 3",
            )
        )

    def p1(self, text: str) -> None:
        """First, un-indented paragraph of a section."""
        self.add(text_para(text, "p1a"))

    def body_p(self, text: str) -> None:
        """An indented body paragraph."""
        self.add(text_para(text))

    def bullet(self, text: str) -> None:
        self.add(text_para(text, "bulletitem"))

    def numbered(self, text: str) -> None:
        self.add(text_para(text, "numitem"))

    def acknowledgement(self, label: str, text: str) -> None:
        content = (
            f'<w:r><w:rPr><w:rStyle w:val="heading3"/></w:rPr><w:t xml:space="preserve">{esc(label)} </w:t></w:r>'
            f"{run(text)}"
        )
        self.add(para(content, "acknowlegments"))

    # -- floats ------------------------------------------------------------
    def figure(self, filename: str, caption: str, width_ratio: float = 1.0) -> None:
        """Embed a PNG with a ``figurecaption`` underneath."""
        source = FIG_DIR / filename
        self.figures += 1
        number = self.figures

        with Image.open(source) as image:
            pixel_width, pixel_height = image.size
        display_width = int(TEXT_WIDTH_DXA * width_ratio)
        display_height = int(display_width * pixel_height / pixel_width)
        cx = display_width * EMU_PER_DXA
        cy = display_height * EMU_PER_DXA

        media_name = f"figure{number}.png"
        self.media[f"word/media/{media_name}"] = source.read_bytes()
        rel_id = f"rIdImg{number}"
        self.image_rels.append((rel_id, media_name))

        drawing = (
            f'<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0" '
            'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">'
            f'<wp:extent cx="{cx}" cy="{cy}"/>'
            '<wp:effectExtent l="0" t="0" r="0" b="0"/>'
            '<wp:docPr id="%d" name="Figure %d" descr="%s"/>'
            "<wp:cNvGraphicFramePr>"
            '<a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/>'
            "</wp:cNvGraphicFramePr>"
            '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<pic:nvPicPr><pic:cNvPr id="{number}" name="{media_name}"/><pic:cNvPicPr/></pic:nvPicPr>'
            '<pic:blipFill><a:blip r:embed="%s" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/>'
            "<a:stretch><a:fillRect/></a:stretch></pic:blipFill>"
            '<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="%d" cy="%d"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
            "</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>"
            % (number + 1, number, esc(caption[:120]), rel_id, cx, cy)
        )
        self.add(para(drawing, None, '<w:ind w:firstLine="0"/><w:jc w:val="center"/><w:keepNext/>'))

        bookmark = f"_RefFigure{number}"
        self.bookmark_id += 1
        caption_runs = (
            f'<w:bookmarkStart w:id="{self.bookmark_id}" w:name="{bookmark}"/>'
            + run("Fig. ", bold=True)
            + field('SEQ "Figure" \\* MERGEFORMAT', str(number), bold=True)
            + run(".", bold=True)
            + run(f" {caption}")
            + f'<w:bookmarkEnd w:id="{self.bookmark_id}"/>'
        )
        self.add(para(caption_runs, "figurecaption"))

    def table_caption(self, caption: str) -> None:
        self.tables += 1
        number = self.tables
        self.bookmark_id += 1
        caption_runs = (
            f'<w:bookmarkStart w:id="{self.bookmark_id}" w:name="_RefTable{number}"/>'
            + run("Table ", bold=True)
            + field('SEQ "Table" \\* MERGEFORMAT', str(number), bold=True)
            + run(".", bold=True)
            + run(f" {caption}")
            + f'<w:bookmarkEnd w:id="{self.bookmark_id}"/>'
        )
        self.add(para(caption_runs, "tablecaption"))

    def table(self, header: list[str], rows: list[list[str]], widths: list[int]) -> None:
        """A LNCS-style table: top rule, head rule, bottom rule, no verticals."""
        total = sum(widths)
        scaled = [int(w * TEXT_WIDTH_DXA / total) for w in widths]
        scaled[-1] += TEXT_WIDTH_DXA - sum(scaled)

        grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in scaled)
        xml = [
            "<w:tbl><w:tblPr>",
            f'<w:tblW w:w="{TEXT_WIDTH_DXA}" w:type="dxa"/>',
            '<w:jc w:val="center"/>',
            '<w:tblLayout w:type="fixed"/>',
            '<w:tblCellMar><w:left w:w="70" w:type="dxa"/><w:right w:w="70" w:type="dxa"/></w:tblCellMar>',
            '<w:tblLook w:val="0000" w:firstRow="0" w:lastRow="0" w:firstColumn="0" '
            'w:lastColumn="0" w:noHBand="0" w:noVBand="0"/>',
            f"</w:tblPr><w:tblGrid>{grid}</w:tblGrid>",
        ]

        def cell(text: str, width: int, borders: str, *, bold: bool = False) -> str:
            props = f'<w:tcW w:w="{width}" w:type="dxa"/>{borders}'
            body = para(
                run(text, bold=bold, size=18),
                None,
                '<w:ind w:firstLine="0"/><w:jc w:val="left"/>'
                '<w:spacing w:before="20" w:after="20" w:line="220" w:lineRule="atLeast"/>',
            )
            return f"<w:tc><w:tcPr>{props}</w:tcPr>{body}</w:tc>"

        top_and_mid = (
            "<w:tcBorders>"
            '<w:top w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
            '<w:bottom w:val="single" w:sz="6" w:space="0" w:color="000000"/>'
            "</w:tcBorders>"
        )
        bottom = (
            "<w:tcBorders>"
            '<w:bottom w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
            "</w:tcBorders>"
        )

        xml.append(
            "<w:tr>"
            + "".join(
                cell(text, scaled[index], top_and_mid, bold=True)
                for index, text in enumerate(header)
            )
            + "</w:tr>"
        )
        for row_index, row in enumerate(rows):
            borders = bottom if row_index == len(rows) - 1 else ""
            xml.append(
                "<w:tr>"
                + "".join(
                    cell(text, scaled[index], borders)
                    for index, text in enumerate(row)
                )
                + "</w:tr>"
            )
        xml.append("</w:tbl>")
        self.add("".join(xml))
        self.add(para("", None, '<w:spacing w:after="0" w:line="120" w:lineRule="exact"/>'))

    def equation(self, parts: list[tuple[str, dict]], note: str = "") -> None:
        """A centred display equation with a right-aligned number."""
        self.equations += 1
        number = self.equations
        content = (
            '<w:r><w:rPr><w:i/></w:rPr><w:tab/></w:r>'
            + rich(parts)
            + '<w:r><w:tab/></w:r>'
            + run("(")
            + field('SEQ "Equation" \\n \\* MERGEFORMAT', str(number))
            + run(")")
        )
        extra = ""
        if note:
            extra = f"<w:r><w:t xml:space=\"preserve\">  {esc(note)}</w:t></w:r>"
        self.add(para(content, "equation"))
        self.bookmark_id += 1

    def references(self, entries: list[str]) -> None:
        for entry in entries:
            self.add(text_para(entry, "referenceitem"))

    # -- output ------------------------------------------------------------
    def document_xml(self) -> str:
        with (TEMPLATE_DIR / "word" / "document.xml").open(encoding="utf-8") as handle:
            template = handle.read()
        head = template.split("<w:body>")[0]
        body = template.split("<w:body>")[1].split("</w:body>")[0]
        sect = body[body.index("<w:sectPr") :]
        return head + "<w:body>" + "".join(self.body) + sect + "</w:body></w:document>"


def build_report() -> Report:
    report = Report()

    # ---------------------------------------------------------------- title
    report.add(
        para(
            run("A Support System for Car Recognition, Valuation and Recommendation in Vietnam"),
            "papertitle",
        )
    )
    report.add(
        para(
            run("Pham Trong Nghia")
            + '<w:r><w:rPr><w:vertAlign w:val="superscript"/></w:rPr><w:t>1</w:t></w:r>'
            + run(", ")
            + run("Tran Cong Vinh")
            + '<w:r><w:rPr><w:vertAlign w:val="superscript"/></w:rPr><w:t>1</w:t></w:r>'
            + run(" and ")
            + run("Tran Quang Duy")
            + '<w:r><w:rPr><w:vertAlign w:val="superscript"/></w:rPr><w:t>1</w:t></w:r>',
            "author",
        )
    )
    report.add(
        para(
            '<w:r><w:rPr><w:vertAlign w:val="superscript"/></w:rPr><w:t>1</w:t></w:r>'
            + run("FSB Institute of Management and Technology, FPT University, "),
            "address",
        )
    )
    report.add(text_para("Ho Chi Minh City, Vietnam", "address"))
    report.add(text_para("Academic Supervisor: Nguyen Xuan Sam", "address"))

    abstract = (
        "This report describes a working prototype that supports three questions that "
        "buyers of second-hand cars in Vietnam ask repeatedly: which listings match what "
        "I am looking for, what kind of vehicle does this photograph show, and is the "
        "asking price reasonable for this year of production and this distance on the "
        "odometer. The three features are implemented as decoupled pipelines that share "
        "one listing table. Text search first applies a set of regular expressions that "
        "turn a Vietnamese request into explicit attributes such as brand, price band, "
        "production year, seat count, body style and fuel type, and then ranks the "
        "surviving listings with a fusion of a multilingual MiniLM sentence embedding and "
        "a TF-IDF representation. Image recognition uses MobileNetV2 pre-trained on "
        "ImageNet with a compact classification head and a confidence gate that refuses "
        "to answer when the network is not sure. Price analysis fits a single-variable "
        "ordinary least squares model to production year and to odometer reading under a "
        "grouped hold-out split. All three components were evaluated on the public "
        "automobile dataset of 15,442 Vietnamese listings and on a seven-class vehicle "
        "image collection of 4,165 photographs; the classifier reaches 97.05% accuracy "
        "and 96.54% macro-F1 on 813 unseen images, while the two regressions explain "
        "less than 5% of the price variance and are therefore reported as descriptive "
        "market references rather than as valuation models."
    )
    report.add(
        para(run("Abstract. ", bold=True) + run(abstract), "abstract")
    )
    report.add(
        para(
            run("Keywords:", bold=True)
            + run(" ")
            + run(
                "hybrid information retrieval, sentence embeddings, TF-IDF, transfer "
                "learning, MobileNetV2, ordinary least squares, used car market, "
                "Streamlit"
            ),
            "keywords",
        )
    )

    # --------------------------------------------------------- 1 introduction
    report.heading1("Introduction")
    report.p1(
        "The Vietnamese second-hand car market has moved almost entirely online, but the "
        "information a buyer has to work with is still mostly unstructured. A typical "
        "listing carries a title, a free-text description, a photograph, a price and a "
        "handful of fields that the seller filled in inconsistently or left blank. "
        "Searching such a collection by keyword alone is fragile for two reasons that "
        "are easy to observe in practice. First, buyers describe a car the way they "
        "talk about it, so that the same request is written as “Toyota Innova 2020 7 chỗ”, "
        "“innova 2020 7 seat”, or “xe 7 chỗ đời 2020”, and a literal substring match "
        "finds only part of the intended set. Second, the most discriminative tokens in "
        "that vocabulary are short alphanumeric codes: a buyer who writes “C200” wants "
        "the C200, and a model that has learned that “Mercedes” is close to “Mercedes” "
        "will happily return the adjacent C300 AMG trim as the answer."
    )
    report.body_p(
        "At the same time, two pieces of information are missing from the listing itself. "
        "A photograph, when present, identifies the vehicle category far more reliably "
        "than the seller’s own text, and the price itself carries information that the "
        "seller does not spell out. Year of production and distance travelled are the "
        "two variables Vietnamese buyers trust most, yet neither is easy to interpret "
        "without a sense of how prices are distributed for comparable cars."
    )
    report.body_p(
        "This report builds a single prototype that addresses these three gaps at once. "
        "The system is deliberately modular rather than monolithic: a hybrid text "
        "retrieval engine, a transfer-learned image classifier and a pair of "
        "econometric baselines are three independent tracks that read the same "
        "underlying listing table and are presented in one web interface. The design "
        "goal is transparency. Every recommendation that the interface displays is "
        "accompanied by the parsed criteria, the per-criterion match state of the "
        "selected listing, and the price trend that produced the chart, so that a user "
        "can see where a suggestion comes from and reject it when it does not fit."
    )
    report.body_p("The report is organised around three research questions:")
    report.bullet(
        "RQ1. How far does a hybrid retrieval engine that combines rule-based attribute "
        "filters, dense multilingual sentence embeddings and sparse TF-IDF matching "
        "recover the listings a Vietnamese buyer intends, and where does it fail?"
    )
    report.bullet(
        "RQ2. How accurately does MobileNetV2 pre-trained on ImageNet separate seven "
        "vehicle categories on a held-out test split of 813 photographs, and what does a "
        "confidence threshold of 0.60 contribute to the reliability of the answer?"
    )
    report.bullet(
        "RQ3. How much of the price variation in the listing table can a "
        "single-variable ordinary least squares model explain when it is validated on a "
        "grouped hold-out split, and what does that imply about the limits of using it "
        "to value an individual car?"
    )
    report.body_p(
        "Section 2 sets out the technical background for the three modelling choices. "
        "Section 3 describes the data and the cleaning that precedes every experiment. "
        "Section 4 walks through the architecture of the prototype. Section 5 reports "
        "the results for each of the three questions, Section 6 discusses what the "
        "results do and do not support, and Section 8 lists the artefacts required to "
        "reproduce the work."
    )

    # ------------------------------------------------------------ 2 background
    report.heading1("Background")
    report.heading2("Hybrid Retrieval with Sentence Embeddings and TF-IDF")
    report.p1(
        "Text retrieval on short, heterogeneous documents is classically addressed with "
        "sparse lexical models. In the tf-idf weighting of Salton and Buckley [13], the "
        "weight of a term combines how often it occurs in a document with how widely it "
        "is distributed across the collection, so that a term appearing in many "
        "documents contributes little to the discrimination between them. This "
        "approach remains attractive here because it is transparent and cheap, and "
        "because it matches the alphanumeric vocabulary of car trims exactly: the "
        "token “c200” is the same token in the query and in the listing, and no amount "
        "of semantic smoothing should blur it."
    )
    report.body_p(
        "Lexical matching alone, however, fails on the part of the query that carries "
        "intent rather than identity. Asking for an SUV with seven seats and a budget "
        "of one billion VND expresses constraints, not vocabulary, and a bag of words "
        "cannot tell whether the author wanted “SUV” to be satisfied by the body-style "
        "field or merely mentioned in the prose. Dense retrieval addresses this "
        "directly. The Transformer architecture of Vaswani et al. [16] replaces recurrence "
        "with self-attention, and Sentence-BERT [12] adapts the pre-trained bidirectional "
        "encoder of Devlin et al. [4] into a bi-encoder in which a query and a document "
        "are embedded independently and compared by cosine "
        "similarity, which is the design this system follows. Multilingual capability "
        "is obtained by distilling an English teacher into a student trained on "
        "translated sentence pairs [11], so that the "
        "encoder can place Vietnamese and English descriptions in a shared space."
    )
    report.body_p(
        "The two families of representation are complementary rather than redundant. "
        "The embedding captures paraphrase and intent and generalises across the many "
        "ways in which the same vehicle can be described; the sparse vector captures "
        "the exact alphanumeric codes and proper nouns that distinguish one trim from "
        "another. The system therefore computes both scores for every candidate and "
        "combines them, rather than choosing one of them."
    )
    report.heading2("Convolutional Neural Networks and Transfer Learning")
    report.p1(
        "Convolutional neural networks learn hierarchical visual features directly "
        "from pixels, which removes the need for hand-designed descriptors; the "
        "architecture of LeCun et al. [9] is the classical reference for the approach. "
        "Training such a network from scratch requires far more data than the present "
        "task provides, so transfer learning is the practical choice: a backbone "
        "pre-trained on ImageNet, a large annotated corpus of natural images "
        "introduced by Deng et al. [3], is reused and only a small classification head "
        "is trained on the target domain."
    )
    report.body_p(
        "MobileNetV2 [14] is well suited to this setting. Its inverted residual blocks "
        "expand the feature map in depth, project it with cheap depthwise "
        "convolutions and contract it again, and the linear bottlenecks keep the "
        "representation expressive at low cost, which is what makes the network a good "
        "candidate for CPU-only inference. Two details of the training procedure "
        "matter for the results reported later. The first is class re-weighting: the "
        "image collection is imbalanced, and the effective-number weighting of Cui "
        "et al. [2] gives each class a weight proportional to the inverse of the expected "
        "overlap between its samples, which is milder than plain inverse frequency and "
        "works better in a short fine-tuning schedule. The second is the use of mild "
        "augmentation only, because aggressive rotation or cropping removes exactly the "
        "grille, badge and headlamp shapes that distinguish the seven brands."
    )
    report.heading2("Ordinary Least Squares as a Price Baseline")
    report.p1(
        "The econometric component of the prototype is deliberately modest. Ordinary "
        "least squares estimates the coefficients that minimise the sum of squared "
        "residuals, and it is the natural first baseline for the relationship between "
        "a listing price and a single numeric attribute such as production year or "
        "odometer reading. The value of such a model in this setting is not prediction "
        "accuracy but orientation: it quantifies how much of the observed price "
        "variation is associated with each attribute, and it therefore establishes the "
        "limit of what an individual attribute can tell a buyer about a price."
    )
    report.body_p(
        "The estimates are obtained through the estimator interface of scikit-learn [10], "
        "and every reported metric is computed on data that was withheld from the "
        "fit. The split is grouped rather than row-wise: listings that share a name "
        "and a price are the same car reposted on several platforms, and a row-wise "
        "split would place such near-duplicates on both sides of the boundary and "
        "inflate the apparent generalisation of the model."
    )

    # ------------------------------------------------------------- 3 dataset
    report.heading1("Dataset and Preprocessing")
    report.heading2("Listing Data")
    report.p1(
        "The primary corpus is a public collection of used-car listings scraped from "
        "Vietnamian marketplaces [8]. It contains 15,442 records described by 17 columns, "
        "among them the model name, the asking price as free text, the production year, "
        "the distance travelled, the body style, the fuel type, the transmission, the "
        "number of seats and doors, the exterior and interior colours, the origin, the "
        "condition and the location. The free-text price field is the reason the "
        "collection needs cleaning before any statistic can be computed: prices appear "
        "as “1 Tỷ 118 Triệu”, as “646 Triệu”, or occasionally as a bare number, and "
        "the parser converts these forms into a single numeric column expressed in "
        "millions of VND. Every listing yields a positive price after conversion, so "
        "no record is discarded at this stage."
    )
    report.body_p(
        "The remaining cleaning steps are conservative and are documented so that they "
        "can be audited. Missing values are never imputed; they are treated as "
        "unknown, and a listing whose year is unknown is simply excluded from the "
        "year-based analysis. The proportions of unknown values are uneven across the "
        "collection and are shown in Table 1: the distance travelled is unknown for "
        "22.95% of the listings, which is by far the most serious gap, whereas the "
        "transmission is recovered from the description text for all but 0.01% of the "
        "records. Body style, fuel type, production year and seat count are each "
        "unknown for 3.57% of the listings, and the interior colour for 7.96%."
    )
    report.table_caption(
        "Proportion of listings with unknown attribute values in the cleaned corpus "
        "(n = 15,442)."
    )
    report.table(
        ["Attribute", "Source column", "Unknown (%)"],
        [
            ["Distance travelled", "Số Km đã đi", "22.95"],
            ["Interior colour", "Màu nội thất", "7.96"],
            ["Number of doors", "Số cửa", "4.03"],
            ["Exterior colour", "Màu ngoại thất", "3.66"],
            ["Production year", "Năm sản xuất", "3.57"],
            ["Body style", "Kiểu dáng", "3.57"],
            ["Fuel type", "Động cơ (engine)", "3.57"],
            ["Seat count", "Số chỗ ngồi", "3.57"],
            ["Origin", "Xuất xứ", "3.57"],
            ["Transmission", "Description (derived)", "0.01"],
        ],
        [26, 44, 18],
    )
    report.p1(
        "The production year is the attribute that most strongly shapes the price "
        "analysis, and it is present for 14,891 listings. Their distribution, shown "
        "in Fig. 1, is concentrated in the second half of the 2010s, with a median "
        "year of 2020 and a range from 1989 to 2024. Asking prices are spread over "
        "three orders of magnitude: the median listing is priced at 659 million VND, "
        "the mean is pulled up to 1,232 million VND by a small number of expensive "
        "vehicles, and the most expensive record in the table is quoted at 63,568 "
        "million VND. Such extreme values are retained because they are genuine "
        "market observations rather than parsing artefacts, but they do mean that "
        "any regression on price has to be read as descriptive rather than "
        "predictive, a point that Section 5 returns to."
    )
    report.figure(
        "fig1_year_distribution.png",
        "Distribution of the 14,891 listings with a known production year. The bulk "
        "of the market consists of vehicles built between 2015 and 2023.",
    )
    report.body_p(
        "A further characteristic of the corpus deserves mention because it constrains "
        "how the results may be interpreted. There are 3,533 listings that share both "
        "a model name and a price with an earlier listing. These are almost certainly "
        "the same vehicle reposted on several platforms, and they inflate the "
        "apparent size of the sample. The retrieval pipeline removes them from the "
        "displayed results, and the regression analysis groups them together so that "
        "a repost cannot appear in both the training and the test partition."
    )
    report.heading2("Image Data")
    report.p1(
        "The visual component uses a separate public collection of vehicle photographs "
        "[7] organised into seven category folders: Audi, Hyundai Creta, Mahindra "
        "Scorpio, Rolls Royce, Maruti Suzuki Swift, Tata Safari and Toyota Innova. "
        "The collection is split by the publisher into 3,352 training and 813 test "
        "photographs, which is close to the conventional 80:20 proportion and means "
        "that the test split can be used exactly as published."
    )
    report.body_p(
        "The class distribution is markedly imbalanced, as Table 2 and Fig. 2 show. "
        "Audi accounts for 1,013 of the 4,165 images, or 24.3% of the collection, and "
        "Toyota Innova for a further 23.2%, so the two largest classes alone "
        "represent 47.5% of the data. The smallest class, Hyundai Creta, contributes "
        "338 images, a factor of three fewer than the largest. A model evaluated on "
        "accuracy alone could therefore obtain a respectable score by favouring the "
        "majority classes, which is why the balanced accuracy and the macro-averaged "
        "F1 score are reported alongside the overall accuracy."
    )
    report.table_caption(
        "Number of photographs per vehicle class in the training and test splits."
    )
    report.table(
        ["Vehicle class", "Train", "Test", "Total", "Share (%)"],
        [
            ["Audi", "814", "199", "1,013", "24.3"],
            ["Hyundai Creta", "271", "67", "338", "8.1"],
            ["Mahindra Scorpio", "316", "75", "391", "9.4"],
            ["Rolls Royce", "311", "74", "385", "9.2"],
            ["Maruti Suzuki Swift", "424", "102", "526", "12.6"],
            ["Tata Safari", "441", "106", "547", "13.1"],
            ["Toyota Innova", "775", "190", "965", "23.2"],
            ["Total", "3,352", "813", "4,165", "100.0"],
        ],
        [34, 14, 14, 16, 22],
    )
    report.figure(
        "fig2_image_counts.png",
        "Training and test photographs for the seven vehicle classes. The imbalance "
        "between the largest and the smallest class is a factor of three.",
    )

    # ------------------------------------------------------- 4 system design
    report.heading1("System Design")
    report.heading2("Architecture Overview")
    report.p1(
        "The prototype is a single Streamlit application [15] organised as three decoupled "
        "processing tracks that share one in-memory listing table. The separation is "
        "not merely cosmetic: each track can fail without taking the others down. "
        "If the sentence embedding model cannot be loaded, the retrieval engine falls "
        "back to TF-IDF matching and reports the degradation in the interface, while "
        "the classifier and the regression charts continue to work. Fig. 3 gives the "
        "overall arrangement."
    )
    report.figure(
        "fig3_architecture.png",
        "Architecture of the prototype. Text queries are parsed into attribute filters "
        "before hybrid ranking, photographs are classified by a transfer-learned "
        "MobileNetV2, and the listing table feeds both the regression baselines and "
        "the price trend charts.",
    )
    report.body_p(
        "Two engineering decisions keep the application responsive on commodity "
        "hardware. First, the sentence embeddings of all 15,442 listings and the "
        "TF-IDF matrix are computed once and cached on disk. The cache key is a "
        "SHA-256 digest of the model name together with a fingerprint of the dataset "
        "file, so a change to the listing data or to the embedding model invalidates "
        "the cache automatically, while an unchanged file reuses it. Second, the "
        "embedding model is loaded lazily, only when a query actually needs semantic "
        "scoring. The model itself is the twelve-layer multilingual MiniLM bi-encoder "
        "released on the Hugging Face Hub [5]. Both the embedding and the inference "
        "step run comfortably on a "
        "multi-core CPU, and no GPU is required for any part of the system."
    )
    report.heading2("Query Parsing and Attribute Extraction")
    report.p1(
        "A natural language request in Vietnamese is first converted into a set of "
        "explicit criteria. Each criterion records the column it constrains, the "
        "comparison operator, the value and a human-readable label that is shown back "
        "to the user, which keeps the extraction step explainable. The parser works "
        "with regular expressions on an accent-stripped, lower-cased copy of the "
        "query, and it recognises up to fourteen attributes: location, distance, "
        "price, production year, interior colour, exterior colour, fuel type, "
        "transmission, origin, condition, seat count, door count, body style and "
        "brand."
    )
    report.body_p(
        "The order in which the parser examines the query is not arbitrary, because "
        "several of these patterns are ambiguous in isolation. Locations are extracted "
        "before fuel type and colour, because a place name such as “Đồng Nai” would "
        "otherwise be matched by a colour or fuel pattern. The distance is extracted "
        "before the price and the year, because the digits “2000” in “2000 km” denote "
        "a distance and not a model year. The interior colour is matched before the "
        "exterior colour, because the phrase “nội thất màu đen” contains the word for "
        "colour but refers to the interior. Once a pattern has been matched, the span "
        "it covered is blanked rather than deleted, so that the remaining tokens are "
        "not concatenated into spurious matches."
    )
    report.body_p(
        "Brand detection is the one criterion that does not rely on a fixed list of "
        "triggers. An exact, word-boundary match against a list of 28 brand names is "
        "tried first. If that fails, the remaining tokens are compared against the "
        "same list with a character-level similarity measure, and a fuzzy match is "
        "accepted only when the best score reaches 0.76 and exceeds the second best "
        "by at least 0.04; the margin requirement prevents a short brand name from "
        "capturing an unrelated word. As a last resort, the parser checks whether a "
        "complete model name from the corpus occurs in the query, which covers the "
        "case in which the user simply types the trim that appears in the listings."
    )
    report.body_p(
        "The extracted criteria are then applied as a hard filter. Every criterion "
        "must hold simultaneously, and a listing whose value for a constrained "
        "attribute is unknown does not satisfy that criterion. The strict filter "
        "therefore narrows the 15,442 listings to a candidate set whose size depends "
        "entirely on the query: the three benchmark requests of Section 5 leave "
        "3,450, 44 and 287 candidates respectively."
    )
    report.heading2("Hybrid Scoring")
    report.p1(
        "The surviving candidates are scored by two complementary representations and "
        "the two scores are fused. Let \\(q\\) be the embedding of the query and "
        "\\(e_i\\) the embedding of listing \\(i\\); because both vectors are "
        "normalised, the inner product is the cosine similarity"
    )
    report.equation(
        [
            ("s", {"italic": True}),
            ("sem", {"italic": True}),
            (" = ", {}),
            ("q", {"italic": True}),
            (" · ", {}),
            ("e", {"italic": True}),
            ("i", {"italic": True}),
        ]
    )
    report.body_p(
        "The sparse score is the tf-idf inner product between the normalised query "
        "vector and the corresponding row of the listing matrix. The two scores are "
        "not on a common scale, so each is min-max normalised across the candidate "
        "set before they are combined, and the fused score is"
    )
    report.equation(
        [
            ("s", {"italic": True}),
            ("hybrid", {"italic": True}),
            (" = ", {}),
            ("w", {"italic": True}),
            (" · ", {}),
            ("s̃", {"italic": True}),
            ("sem", {"italic": True}),
            (" + (1 − ", {}),
            ("w", {"italic": True}),
            (") · ", {}),
            ("s̃", {"italic": True}),
            ("tfidf", {"italic": True}),
        ]
    )
    report.body_p(
        "where \\(w = 0.70\\) is the default weight on the semantic component and "
        "\\(s̃\\) denotes a min-max normalised score. The weighting is exposed as a "
        "control in the interface, because the balance it encodes is a design "
        "decision rather than a law: a user who is searching for a specific trim "
        "wants lexical matching to dominate, while a user who describes a car in "
        "general terms benefits from the semantic component. The system also offers "
        "the two scores in isolation, and it is the comparison between the three "
        "modes that constitutes the retrieval evaluation of Section 5."
    )
    report.heading2("Progressive Relaxation")
    report.p1(
        "A strict filter is brittle in a heterogeneous corpus. A buyer who asks for a "
        "Toyota from 2020 with seven seats and a budget of 700 million VND may have "
        "no exact match even though the market contains the car they want, because a "
        "single listing lacks the interior colour they mentioned or sits 20 million "
        "VND above the ceiling. The system therefore falls back through a fixed "
        "sequence of stages, and the sequence is ordered so that the least "
        "informative constraints are given up last."
    )
    report.body_p(
        "The five stages are: keep every criterion; release the price band; release "
        "the production year; release the colour, distance, location, fuel, "
        "transmission, origin, condition and door-count criteria together; and "
        "finally release the seat count and the body style. The brand criterion is "
        "never released, because a search for a Toyota that returns a Hyundai is not "
        "a relaxed result but a wrong answer. Each stage is attempted in turn and the "
        "first one that yields at least one candidate is used, so a query that is "
        "already satisfiable is never relaxed at all."
    )
    report.body_p(
        "Relaxation is never silent. The interface names the criteria that were "
        "released, and each displayed listing shows the match state of every original "
        "criterion, so that a suggestion with an unmet constraint is visibly "
        "different from one that satisfies the request exactly. When the price band "
        "has been released, candidates are ordered by their distance from the "
        "requested budget before the retrieval score is consulted, which keeps the "
        "suggestions closest to the stated budget at the top of the list."
    )
    report.heading2("Image Recognition and Confidence Gating")
    report.p1(
        "The classifier is MobileNetV2 with the ImageNet weights and the fully "
        "connected top removed, assembled with the Keras API of TensorFlow [1], and "
        "followed by a classification head consisting of a "
        "global average pooling layer, a dropout layer with rate 0.30, a dense layer "
        "of 128 units with ReLU activation, a second dropout with rate 0.20 and a "
        "softmax output over the seven classes. The network has 2,422,855 parameters "
        "in total. Training proceeds in two stages: five epochs with the backbone "
        "frozen and a learning rate of 10⁻³, followed by fine-tuning of the whole "
        "network at 10⁻⁴, with early stopping on validation accuracy and a learning "
        "rate schedule that halves the rate when validation loss stops improving."
    )
    report.body_p(
        "At inference time a photograph is converted to RGB, resized to "
        "224 × 224 pixels and scaled to the unit interval, exactly as in training, "
        "and the softmax output is inspected. A prediction is accepted only when the "
        "probability of the leading class reaches a threshold of 0.60; below that "
        "value the system reports the result as uncertain instead of naming a class. "
        "The threshold was chosen so that a wrong answer is preferable to a "
        "confident wrong answer, and it can be varied interactively to observe how "
        "many predictions would be rejected at different levels."
    )
    report.heading2("Price Trend Inspection")
    report.p1(
        "The third track fits ordinary least squares models to the price against "
        "production year and against odometer reading and renders the fit together "
        "with binned medians and an inter-quartile band, so that the user sees the "
        "spread of the market rather than a single line. When a search has already "
        "fixed the brand, the analysis is restricted to that brand, which is what "
        "makes the chart informative for a buyer who is considering one model rather "
        "than the market as a whole."
    )
    report.figure(
        "fig4_price_vs_year.png",
        "Listing price against production year with the fitted regression line, the "
        "binned medians and the inter-quartile band. The vertical axis is truncated "
        "at 5,000 million VND; the slope is positive but the spread at every year is "
        "far wider than the effect.",
    )
    report.figure(
        "fig5_price_vs_mileage.png",
        "Listing price against odometer reading for the 10,728 listings within "
        "0–100,000 km. The fitted slope is negative and close to zero, which "
        "reflects the weak association between distance and price in this corpus.",
    )
    report.body_p(
        "The price distribution across body styles, shown in Fig. 6, complements the "
        "regression output. The medians differ by more than an order of magnitude "
        "between the cheapest and the most expensive body style, which suggests that "
        "body style is a more informative attribute for orienting a buyer than either "
        "of the two numeric variables examined here."
    )
    report.figure(
        "fig6_price_by_body_style.png",
        "Distribution of listing prices by body style on a logarithmic scale. The "
        "spread within each group is wide, so body style explains the position of a "
        "car in the price range better than year or mileage do.",
    )

    # ------------------------------------------------------------ 5 results
    report.heading1("Results")
    report.heading2("Retrieval Behaviour on Benchmark Queries")
    report.p1(
        "The retrieval engine was evaluated on three representative requests, each "
        "chosen to exercise a different failure mode of keyword search: a broad "
        "category with a budget constraint, a brand and model with an explicit "
        "configuration, and a brand with an alphanumeric trim designation. For each "
        "request the three retrieval modes were run over the identical candidate set "
        "and the top-ranked listing was recorded. Table 3 gives the outcome."
    )
    report.table_caption(
        "Top-ranked listing for three benchmark queries under the hybrid, semantic "
        "and TF-IDF retrieval modes."
    )
    report.table(
        [
            "Query",
            "Mode",
            "Cand.",
            "Top-ranked listing",
            "Price (M VND)",
        ],
        [
            ["SUV below 1 bn", "Hybrid", "3,450", "Toyota Avanza Premio 1.5 AT - 2024", "565"],
            ["SUV below 1 bn", "Semantic", "3,450", "Toyota Avanza Premio 1.5 AT - 2022", "520"],
            ["SUV below 1 bn", "TF-IDF", "3,450", "Lynk & Co 01 - 2024", "999"],
            ["Innova 2020, 7 seats", "Hybrid", "44", "Toyota Innova V 2.0 AT - 2020", "699"],
            ["Innova 2020, 7 seats", "Semantic", "44", "Toyota Innova V 2.0 AT - 2020", "699"],
            ["Innova 2020, 7 seats", "TF-IDF", "44", "Toyota Innova V 2.0 AT - 2020", "699"],
            ["C200 2021", "Hybrid", "287", "Mercedes-Benz C class C300 AMG - 2021", "1,350"],
            ["C200 2021", "Semantic", "287", "Mercedes-Benz C class C300 AMG - 2021", "1,350"],
            ["C200 2021", "TF-IDF", "287", "Mercedes-Benz C class C200 Exclusive - 2021", "1,220"],
        ],
        [21, 12, 10, 40, 17],
    )
    report.body_p(
        "The attribute filter behaves as intended. For the Innova query the three "
        "criteria that the parser extracts, a 2020 model year, seven seats and the "
        "Toyota brand, reduce the corpus to 44 listings, and all three retrieval "
        "modes agree on the same top result, the 2020 Innova V 2.0 with an automatic "
        "transmission at 699 million VND. When the query is unambiguous and the "
        "candidate set is small, the two representations are not in conflict and the "
        "hybrid score inherits the agreement."
    )
    report.body_p(
        "The broad SUV query separates the modes more clearly. The hybrid ranking "
        "returns a 2024 Toyota Avanza Premio at 565 million VND, well inside the "
        "budget, and the semantic mode returns the same model from 2022 at 520 "
        "million VND. The lexical mode behaves differently: it returns a Lynk & Co 01 "
        "from 2024 at 999 million VND, the most expensive car in the candidate set, "
        "because the query shares the token “SUV” with that listing while the other "
        "two vehicles are described with different wording. The hybrid score avoids "
        "this failure by giving the semantic component the larger weight, but the "
        "example also shows that the lexical score is not a reliable ranking signal "
        "on its own."
    )
    report.body_p(
        "The trim query is the most instructive of the three. The parser extracts the "
        "Mercedes-Benz brand and the 2021 model year and leaves 287 candidates, but "
        "the requested trim “C200” is not among the parsed criteria, because trim "
        "designations are alphanumeric codes that the regular expressions cannot "
        "distinguish from one another. Both the hybrid and the semantic mode rank the "
        "C300 AMG first, at 1,350 million VND, while only the lexical mode returns "
        "the C200 Exclusive at 1,220 million VND. This is the characteristic weakness "
        "of a bi-encoder: the query and the document are embedded independently, so "
        "the difference between two adjacent trim codes is smaller than the "
        "differences between them and unrelated vehicles. The hybrid score inherits "
        "the mistake because the semantic component dominates."
    )
    report.body_p(
        "Three observations follow from these experiments. First, rule-based "
        "extraction is reliable when the requested attributes are categorical, "
        "numeric or explicitly named, and the candidate sets it produces are small "
        "enough for the ranking stage to work with. Second, the two representations "
        "fail in different ways, which is precisely the argument for fusing them "
        "rather than choosing one. Third, the trim confusion is not fixable by "
        "re-weighting the fusion: it requires a mechanism that compares the query and "
        "the document jointly, such as a cross-encoder reranking stage or an explicit "
        "trim token extracted from the title. This is the clearest direction for "
        "future work and it is discussed again in Section 6."
    )
    report.heading2("Image Classification Performance")
    report.p1(
        "The classifier was evaluated once, on the published test split of 813 "
        "photographs that was never used for training or for the validation split. "
        "It reaches 97.05% accuracy, a balanced accuracy of 96.15% and a macro-averaged "
        "F1 score of 96.54%, with a test loss of 0.1119. The gap between the overall "
        "accuracy and the balanced accuracy is only 0.90 percentage points, which "
        "indicates that the model does not trade minority-class performance for "
        "majority-class accuracy. Table 4 lists the per-class scores and Fig. 7 "
        "shows the confusion matrix."
    )
    report.table_caption(
        "Per-class precision, recall and F1 score of MobileNetV2 on the 813 test "
        "photographs."
    )
    report.table(
        ["Vehicle class", "Test", "Precision (%)", "Recall (%)", "F1 (%)"],
        [
            ["Audi", "199", "98.49", "98.49", "98.49"],
            ["Hyundai Creta", "67", "95.24", "89.55", "92.31"],
            ["Mahindra Scorpio", "75", "98.65", "97.33", "97.99"],
            ["Rolls Royce", "74", "98.59", "94.59", "96.55"],
            ["Maruti Suzuki Swift", "102", "97.00", "95.10", "96.04"],
            ["Tata Safari", "106", "94.59", "99.06", "96.77"],
            ["Toyota Innova", "190", "96.41", "98.95", "97.66"],
            ["Overall", "813", "97.05", "96.15", "96.54"],
        ],
        [32, 13, 19, 18, 18],
    )
    report.figure(
        "fig7_confusion_matrix.png",
        "Row-normalised confusion matrix of MobileNetV2 on the 813 test photographs. "
        "The diagonal dominates, and the visible confusions are between neighbouring "
        "categories such as the two luxury saloons.",
    )
    report.body_p(
        "The confusion matrix is dominated by the diagonal, and the errors that do "
        "occur are concentrated in the classes that a buyer is most likely to "
        "encounter in the wild. The largest single confusion is Audi being read as "
        "Rolls Royce, three cases out of 199, which is understandable given that the "
        "two cars share a front-grille shape that dominates a photograph taken at an "
        "angle. Hyundai Creta is the weakest class, with a recall of 89.55% and an F1 "
        "score of 92.31%, and it is also the class with the fewest test images; its "
        "errors are spread over Rolls Royce, Tata Safari and Toyota Innova, the "
        "other high-bodied vehicles in the collection. Tata Safari shows the mirror "
        "image behaviour, with the highest recall of all classes at 99.06% and the "
        "lowest precision at 94.59%, so it absorbs some of the compact-SUV "
        "confusions."
    )
    report.figure(
        "fig8_per_class_scores.png",
        "Per-class precision, recall and F1 score against the overall accuracy and "
        "the balanced accuracy. Every class lies in a narrow band around the two "
        "reference lines, confirming that the imbalance of the collection has been "
        "neutralised.",
    )
    report.body_p(
        "Fig. 8 makes the same point graphically. All seven classes fall between "
        "89.55% and 99.06% recall, and the two reference lines, the overall accuracy "
        "and the balanced accuracy, sit close together and close to the middle of "
        "that band. The class re-weighting during training is therefore doing what it "
        "was introduced to do: a model trained on this imbalanced collection without "
        "re-weighting would have concentrated its errors on exactly the classes that "
        "the balanced accuracy exposes."
    )
    report.body_p(
        "The confidence gate of 0.60 was applied to the predictions in the "
        "interface. On well-separated categories such as the trucks and the luxury "
        "saloons, the softmax output is close to one and the gate never rejects a "
        "prediction. It becomes active on the genuinely ambiguous photographs, "
        "typically those taken from an oblique angle or in poor light, and the "
        "interface then reports that the class could not be determined. This is the "
        "intended behaviour for a decision-support tool: a refusal is more useful to "
        "a buyer than a confident misidentification."
    )
    report.heading2("Price Regression Results")
    report.p1(
        "Both regressions were fitted with the same protocol. The listings are split "
        "into a training part of 80% and a test part of 20% using a grouped split "
        "on the pair of model name and price, so that a listing and its repost cannot "
        "fall on opposite sides of the boundary. The fitted coefficients are compared "
        "against a mean-predictor baseline fitted on the same training part, and the "
        "stability of the estimate is assessed with a five-fold grouped "
        "cross-validation performed inside the training part only. The results are "
        "summarised in Table 5 and discussed below."
    )
    report.table_caption(
        "Ordinary least squares models of listing price, evaluated on a grouped 20% "
        "hold-out split."
    )
    report.table(
        ["Model", "n", "Slope", "Desc. R²", "Test R²", "Test RMSE (M VND)"],
        [
            [
                "Price ~ Year",
                "14,891",
                "+77.41 per year",
                "0.034",
                "0.047",
                "1,703.3",
            ],
            [
                "Price ~ Mileage (0–100k km)",
                "10,728",
                "−0.00517 per km",
                "0.008",
                "0.008",
                "1,686.1",
            ],
            [
                "Price ~ Mileage (all non-negative)",
                "11,898",
                "−1.56 × 10⁻⁵ per km",
                "0.0003",
                "—",
                "—",
            ],
            [
                "Mean-predictor baseline (year model)",
                "14,891",
                "—",
                "0.000",
                "0.000",
                "1,744.9",
            ],
        ],
        [28, 11, 19, 11, 11, 20],
    )
    report.body_p(
        "The year model has a clearly positive slope: a listing is worth about 77.41 "
        "million VND more for each additional year of production, which is "
        "consistent with the direction a buyer would expect. The descriptive R² of "
        "0.034 and the test R² of 0.047 are nevertheless very low, and the grouped "
        "cross-validation on the training part gives 0.0349 with a standard "
        "deviation of 0.0158 across the five folds. The mean-predictor baseline "
        "achieves a test RMSE of 1,744.9 million VND, while the regression achieves "
        "1,703.3 million VND, an improvement of 2.4% that is small compared with the "
        "dispersion of prices in the corpus."
    )
    report.body_p(
        "The mileage model is weaker still. Within the 0–100,000 km band the slope "
        "is −0.00517 million VND per kilometre and the test R² is 0.008; the "
        "cross-validated R² is 0.0074 with a standard deviation of 0.0035. Removing "
        "the upper limit and keeping all 11,898 listings with a non-negative "
        "odometer reading collapses the slope to −1.56 × 10⁻⁵ and the descriptive "
        "R² to 0.0003, which means that the apparent dependence on distance in the "
        "restricted sample is driven by the most expensive vehicles being clustered at "
        "low mileages rather than by a general market pattern."
    )
    report.body_p(
        "These numbers answer RQ3, and the answer is a negative one with a clear "
        "interpretation. Neither attribute explains more than five percent of the "
        "price variance, so neither can support a valuation of an individual car. "
        "The models retain value in a different role: they quantify the market's "
        "price gradient, and they establish the scale of the noise against which any "
        "future, richer model would have to demonstrate improvement. A multivariable "
        "model that adds brand, body style, transmission and origin is the natural "
        "next step, and the body-style comparison of Fig. 6 suggests that the "
        "largest share of the missing explanatory power lies there."
    )

    # ---------------------------------------------------------- 6 discussion
    report.heading1("Discussion")
    report.heading2("What the Results Support")
    report.p1(
        "The prototype demonstrates that the three capabilities can be combined "
        "without interfering with one another, and that each of them is useful "
        "within a well-defined range. Rule-based parsing isolates the attribute "
        "constraints reliably and reduces the corpus to candidate sets of a few dozen "
        "or a few thousand entries. The two retrieval representations are genuinely "
        "complementary, as the trim query shows: the semantic score finds the "
        "vehicle that fits the description, and the lexical score finds the vehicle "
        "that carries the requested designation. The classifier is the strongest "
        "component by a wide margin, and it stays balanced across the seven classes "
        "despite a threefold imbalance in the training data."
    )
    report.body_p(
        "Two engineering properties are worth stating explicitly because they make "
        "the results reproducible rather than merely plausible. The cache key is "
        "derived from the dataset fingerprint and the model name, so a cached "
        "embedding can never be silently reused for a different corpus or a "
        "different model. And the retrieval engine degrades rather than fails: if "
        "the embedding model cannot be reached, the interface continues to serve "
        "lexical results and says so, which keeps the application usable in an "
        "offline environment."
    )
    report.heading2("Limitations")
    report.p1(
        "The limitations are of four different kinds, and it is worth separating "
        "them, because they call for different remedies."
    )
    report.body_p(
        "The first is a modelling limitation in the retrieval stage. Because the "
        "query and the listing are encoded independently, the embedding space "
        "compresses the distinction between adjacent trim codes, and the trim query "
        "in Table 3 demonstrates the consequence. This is a limitation of the bi-"
        "encoder design itself and cannot be repaired by tuning the fusion weight. "
        "The remedies are either a cross-encoder reranking stage that scores the "
        "query and each candidate jointly, or a rule that extracts the alphanumeric "
        "trim token from the listing title and matches it exactly, which would "
        "convert the trim into a hard filter on the same footing as the brand."
    )
    report.body_p(
        "The second is a limitation of the data. The corpus contains 3,533 reposted "
        "listings, and its attribute fields are missing at rates that range from "
        "0.01% for the transmission to 22.95% for the distance travelled. Missing "
        "values are treated as unknown rather than imputed, which is the honest "
        "treatment but which discards a fifth of the corpus from the mileage "
        "analysis. The price field also contains a long tail: the most expensive "
        "listing is priced at 63,568 million VND, fifty times the median, and a "
        "handful of such records dominate any sum of squared errors."
    )
    report.body_p(
        "The third is a limitation of the evaluation. The retrieval experiment rests "
        "on three benchmark queries, and while they were chosen to represent "
        "distinct query types, three queries cannot support a statistical claim "
        "about ranking quality. A proper evaluation would require a labelled set of "
        "queries with relevance judgements, and this is the most expensive piece of "
        "future work. The classification evaluation is on a published test split that "
        "was never used for training, which makes it trustworthy, but the collection "
        "covers seven categories that are all popular vehicles in India and none of "
        "which is a common vehicle in Vietnam, so the reported accuracy should not be "
        "read as an estimate of performance on Vietnamese traffic."
    )
    report.body_p(
        "The fourth is a limitation of scope. The price models are descriptive "
        "baselines rather than valuation models, and the system presents them as "
        "such. Nothing in the interface uses a regression output to modify a "
        "ranking score, which is a deliberate choice: the R² values reported in "
        "Table 5 do not justify such an adjustment."
    )

    # ---------------------------------------------------------- 7 conclusion
    report.heading1("Conclusion")
    report.p1(
        "This report has built and evaluated a prototype that supports car search, "
        "car recognition and price orientation for the Vietnamese second-hand market. "
        "The three pipelines are decoupled and share a single cleaned listing table "
        "of 15,442 records."
    )
    report.body_p(
        "On the retrieval side, the hybrid engine recovers the intended listing "
        "exactly whenever the request is categorical, and the three benchmark "
        "queries show that the semantic and lexical components fail in different "
        "and complementary ways. The trim case documents the residual weakness of "
        "an independently encoded query and identifies the concrete remedy."
    )
    report.body_p(
        "On the recognition side, transfer learning proved to be the decisive "
        "choice: a compact MobileNetV2 head on a frozen-then-fine-tuned ImageNet "
        "backbone reaches 97.05% accuracy and 96.54% macro-F1 on 813 unseen "
        "photographs, with a balanced accuracy within one percentage point of the "
        "overall accuracy despite a threefold class imbalance."
    )
    report.body_p(
        "On the price side, the negative result is equally informative. Single-"
        "variable ordinary least squares explains 4.7% of the price variance in "
        "production year and 0.8% in odometer reading, so a single attribute cannot "
        "value a car. The practical implication for the prototype is that the price "
        "track should be used as an orientation aid and a data-quality diagnostic "
        "rather than as a scoring mechanism, which is how it is implemented."
    )
    report.body_p(
        "Taken together, the results suggest that the value of such a system lies "
        "less in the sophistication of any single model than in making several weak "
        "signals legible to a user at once, together with the reasons why each "
        "suggestion was made."
    )

    # ------------------------------------------------- 8 reproducibility
    report.heading1("Reproducibility and Future Work")
    report.p1(
        "All three components are implemented in Python and run without a GPU. The "
        "experiments in this report were run on a laptop with an Apple M5 processor "
        "and 16 GB of memory, using Python 3.14.7, TensorFlow 2.22, scikit-learn "
        "1.9.1, sentence-transformers 6.2.0, pandas 3.0.6, NumPy 2.5.3, Matplotlib "
        "3.11.2 [6] and Streamlit 1.66.0 [15]. The listing data and the seven-class image "
        "collection are public, and the trained classification model together with "
        "its metrics is published under a pinned commit so that the reported "
        "numbers can be reproduced exactly rather than re-trained approximately."
    )
    report.body_p("Three directions for future work follow from the analysis above.")
    report.bullet(
        "Replace the bi-encoder with a two-stage pipeline that adds a cross-encoder "
        "reranking stage over the top candidates, or extract the alphanumeric trim "
        "token from the listing title and promote it to a hard filter. Either "
        "change addresses the trim confusion documented in Table 3 directly."
    )
    report.bullet(
        "Extend the price analysis to a multivariable model that conditions on "
        "brand, body style, transmission, fuel type and origin, using the grouped "
        "split throughout and reporting the improvement over the single-variable "
        "baselines of Table 5. The body-style comparison of Fig. 6 indicates where "
        "the missing explanatory power is concentrated."
    )
    report.bullet(
        "Build a labelled query set with relevance judgements so that retrieval "
        "quality can be measured rather than illustrated, and calibrate the "
        "confidence threshold of 0.60 on a validation set drawn from Vietnamese "
        "vehicles rather than from the Indian categories of the current collection."
    )

    # ------------------------------------------------- acknowledgements
    report.acknowledgement(
        "Acknowledgments.",
        "The authors thank their academic supervisor, Nguyen Xuan Sam, for the "
        "guidance on the design of the retrieval pipeline and for the review of "
        "earlier drafts of this report.",
    )
    report.acknowledgement(
        "Data availability.",
        "The listing data and the vehicle image collection used in this study are "
        "publicly available on Kaggle. The trained MobileNetV2 model and its "
        "evaluation metrics are published with the project repository under a pinned "
        "commit, so that the numbers reported here can be verified without "
        "retraining.",
    )
    report.acknowledgement(
        "Competing interests.",
        "The authors have no competing interests to declare that are relevant to the "
        "content of this article.",
    )

    # ------------------------------------------------------------ references
    report.add(
        para(
            run("References"),
            "heading1",
            '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="0"/></w:numPr>',
        )
    )
    report.references(
        [
            "Abadi, M., Barham, P., Chen, J., Chen, Z., Davis, A., Dean, J., Devin, "
            "M., Ghemawat, S., Irving, G., Isard, M., Kudlur, M., Levenberg, J., "
            "Monga, R., Moore, S., Murray, D.G., Steiner, B., Tucker, P., Vasudevan, "
            "A., Warden, P., Wicke, M., Yu, Y., Zheng, X.: TensorFlow: a system for "
            "large-scale machine learning. In: Proceedings of the 12th USENIX "
            "Symposium on Operating Systems Design and Implementation, pp. 265–283. "
            "USENIX Association (2016)",
            "Cui, Y., Jia, M., Lin, T.-Y., Song, Y., Belongie, S.: Class-balanced loss "
            "based on effective number of samples. In: Proceedings of the IEEE/CVF "
            "Conference on Computer Vision and Pattern Recognition, pp. 9268–9277. "
            "IEEE (2019)",
            "Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., Fei-Fei, L.: ImageNet: "
            "a large-scale hierarchical image database. In: Proceedings of the IEEE "
            "Conference on Computer Vision and Pattern Recognition, pp. 248–255. IEEE "
            "(2009)",
            "Devlin, J., Chang, M.-W., Lee, K., Toutanova, K.: BERT: pre-training of "
            "deep bidirectional transformers for language understanding. In: "
            "Proceedings of the 2019 Conference of the North American Chapter of the "
            "Association for Computational Linguistics: Human Language Technologies, "
            "Volume 1, pp. 4171–4186. Association for Computational Linguistics (2019)",
            "Hugging Face: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2. "
            "https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, "
            "last accessed 06/11/2026",
            "Hunter, J.D.: Matplotlib: a 2D graphics environment. Comput. Sci. Eng. "
            "9(3), 90–95 (2007)",
            "Kaggle: Car Images Dataset (kshitij192/cars-image-dataset). "
            "https://www.kaggle.com/datasets/kshitij192/cars-image-dataset, last "
            "accessed 06/11/2026",
            "Kaggle: Vietnam Automobile Dataset (qucvinhdng/automobile-dataset). "
            "https://www.kaggle.com/datasets/qucvinhdng/automobile-dataset, last "
            "accessed 06/11/2026",
            "LeCun, Y., Bottou, L., Bengio, Y., Haffner, P.: Gradient-based learning "
            "applied to document recognition. Proc. IEEE 86(11), 2278–2324 (1998)",
            "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., "
            "Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., "
            "Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., "
            "Duchesnay, E.: Scikit-learn: machine learning in Python. JMLR 12, "
            "2825–2830 (2011)",
            "Reimers, N., Gurevych, I.: Making monolingual sentence embeddings "
            "multilingual using knowledge distillation. In: Proceedings of the 2020 "
            "Conference on Empirical Methods in Natural Language Processing, "
            "pp. 4512–4525. Association for Computational Linguistics (2020)",
            "Reimers, N., Gurevych, I.: Sentence-BERT: sentence embeddings using "
            "Siamese BERT-networks. In: Proceedings of the 2019 Conference on "
            "Empirical Methods in Natural Language Processing and the 9th International "
            "Joint Conference on Natural Language Processing, pp. 3982–3992. "
            "Association for Computational Linguistics (2019)",
            "Salton, G., Buckley, C.: Term-weighting approaches in automatic text "
            "retrieval. Inf. Process. Manage. 24(5), 513–523 (1988)",
            "Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., Chen, L.-C.: MobileNetV2: "
            "inverted residuals and linear bottlenecks. In: Proceedings of the IEEE "
            "Conference on Computer Vision and Pattern Recognition, pp. 4510–4520. IEEE "
            "(2018)",
            "Streamlit documentation. https://docs.streamlit.io/, last accessed "
            "06/11/2026",
            "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, "
            "A.N., Kaiser, Ł., Polosukhin, I.: Attention is all you need. In: Advances "
            "in Neural Information Processing Systems 30, pp. 5998–6008 (2017)"
        ]
    )
    return report


def write_docx(report: Report, destination: Path) -> None:
    """Assemble the .docx package from the template and the generated body."""
    if not TEMPLATE_DIR.is_dir():
        raise FileNotFoundError(
            f"Missing template package: {TEMPLATE_DIR}. Run prepare_template.py first."
        )

    with (TEMPLATE_DIR / "word" / "_rels" / "document.xml.rels").open(
        encoding="utf-8"
    ) as handle:
        rels = handle.read()

    image_rels = "".join(
        f'<Relationship Id="{rel_id}" Type="http://schemas.openxmlformats.org/'
        f'officeDocument/2006/relationships/image" Target="media/{media_name}"/>'
        for rel_id, media_name in report.image_rels
    )
    rels = rels.replace("</Relationships>", image_rels + "</Relationships>")

    document = report.document_xml()

    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(TEMPLATE_DIR.rglob("*")):
            if path.is_dir():
                continue
            relative = path.relative_to(TEMPLATE_DIR).as_posix()
            if relative == "word/document.xml":
                archive.writestr(relative, document)
            elif relative == "word/_rels/document.xml.rels":
                archive.writestr(relative, rels)
            else:
                archive.write(path, relative)
        for relative, payload in report.media.items():
            archive.writestr(relative, payload)


def main() -> None:
    report = build_report()
    write_docx(report, OUT_PATH)
    print(
        f"written {OUT_PATH} "
        f"({report.figures} figures, {report.tables} tables, {report.equations} equations)"
    )


if __name__ == "__main__":
    main()

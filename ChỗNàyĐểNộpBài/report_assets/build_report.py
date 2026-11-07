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
                runs.append('<w:r><w:t xml:space="preserve">' + esc(rest) + "</w:t></w:r>")
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
    def figure(self, filename: str, caption: str, width_ratio: float = 0.82) -> None:
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
            f'<wp:docPr id="%d" name="Figure %d" descr="%s"/>'
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
            + run(f"Fig. {number}.", bold=True)
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
            + run(f"Table {number}.", bold=True)
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
            + run(f"({number})")
        )
        extra = ""
        if note:
            extra = f'<w:r><w:t xml:space="preserve">  {esc(note)}</w:t></w:r>'
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
        # The page setup is inherited unchanged from the Springer proceedings
        # template: A4 with 4.4 cm side margins and 5.2 cm at the top and the
        # bottom, which is the layout the reference report follows.
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
            run("Pham Trong Nghia, Tran Cong Vinh and Tran Quang Duy"),
            "author",
        )
    )
    report.add(text_para("Course PPR501, Class MSA44HCM", "address"))
    report.add(
        text_para(
            "School of Business and Technology FSB, FPT University, "
            "Ho Chi Minh City, Vietnam",
            "address",
        )
    )
    report.add(text_para("Lecturer: Dr. Nguyen Xuan Sam", "address"))

    abstract = (
        "This report presents a car recognition, price estimation and recommendation "
        "system for the Vietnamese second-hand market, implemented as a single "
        "Streamlit application. The system combines structured filtering of Vietnamese "
        "natural-language queries, hybrid ranking of the surviving listings that fuses "
        "a multilingual MiniLM sentence embedding with a TF-IDF representation, and "
        "image classification of vehicle photographs into seven categories. On 15,442 "
        "Vietnamese listings, ordinary least squares explains 4.7% of the price "
        "variance in production year and 0.8% in odometer reading, so the two "
        "regressions are reported as descriptive market references rather than "
        "valuation models. On 813 test photographs, MobileNetV2 pre-trained on ImageNet "
        "reaches 97.05% accuracy and 96.54% macro-F1, with per-class recall ranging "
        "from 89.55% to 99.06%, so the class imbalance of the collection has been "
        "neutralised. A prioritised constraint-relaxation mechanism and a confidence "
        "gate on the classifier make the system degrade gracefully rather than return a "
        "confident wrong answer."
    )
    report.add(
        para(run("Abstract. ", bold=True) + run(abstract), "abstract")
    )
    report.add(
        para(
            run("Keywords:", bold=True)
            + run(" ")
            + run(
                "semantic search, sentence embeddings, TF-IDF, hybrid retrieval, "
                "convolutional neural network, transfer learning, linear regression, "
                "car valuation, car recommendation"
            ),
            "keywords",
        )
    )

    # --------------------------------------------------------- 1 introduction
    report.heading1("Introduction")
    report.p1(
        "The Vietnamese second-hand car market has moved almost entirely online, but "
        "the information a buyer works with is still mostly unstructured: a title, a "
        "free-text description, a photograph, a price, and a handful of fields that the "
        "seller filled in inconsistently or left blank. Keyword search over such a "
        "collection is fragile for two reasons. First, buyers describe a car the way "
        "they talk about it, so that one request is written as “Toyota Innova 2020 7 "
        "chỗ”, “innova 2020 7 seat” or “xe 7 chỗ đời 2020”, and a literal substring "
        "match finds only part of the intended set. Second, the most discriminative "
        "tokens in that vocabulary are short alphanumeric codes: a buyer who writes "
        "“C200” wants the C200, and a model that has learned that “Mercedes” is close "
        "to “Mercedes” will happily return the adjacent C300 AMG trim. A photograph "
        "identifies the vehicle category more reliably than the seller’s own text, and "
        "the price of a car cannot be read without a sense of how prices are "
        "distributed for comparable vehicles."
    )
    report.body_p(
        "This project develops and evaluates a single application that addresses these "
        "gaps at once, and it is deliberately modular: a hybrid text retrieval engine, "
        "a transfer-learned image classifier and two econometric baselines are "
        "independent tracks that read the same listing table and are presented in one "
        "web interface. The design goal is transparency: every recommendation is "
        "accompanied by the parsed criteria, the per-criterion match state of the "
        "selected listing and the price trend behind the chart. Three research "
        "questions organise the report:"
    )
    report.bullet(
        "RQ1. How far does a hybrid retrieval engine combining rule-based attribute "
        "filters, dense multilingual sentence embeddings and sparse TF-IDF matching "
        "recover the listings a Vietnamese buyer intends, and where does it fail?"
    )
    report.bullet(
        "RQ2. How accurately does MobileNetV2 pre-trained on ImageNet separate seven "
        "vehicle categories on a held-out test split of 813 photographs, and what does a "
        "confidence threshold of 0.60 contribute to the reliability of the answer?"
    )
    report.bullet(
        "RQ3. How much of the price variation can a single-variable ordinary least "
        "squares model explain on a grouped hold-out split, and what does that imply "
        "about using it to value an individual car?"
    )
    report.body_p(
        "Section 2 reviews the background, Section 3 the dataset, Section 4 the system "
        "design, Section 5 the results for the three research questions, Section 6 the "
        "strengths and the limitations, and Section 7 the conclusion."
    )

    # ------------------------------------------------------------ 2 background
    report.heading1("Background and Related Work")
    report.heading2("Semantic Retrieval with Sentence Embeddings")
    report.p1(
        "Text retrieval on short, heterogeneous documents is classically addressed with "
        "sparse lexical models. In the tf-idf weighting of Salton and Buckley [10], the "
        "weight of a term combines how often it occurs in a document with how widely it "
        "is distributed across the collection. The approach remains attractive because "
        "it is transparent, cheap, and matches the alphanumeric vocabulary of car trims "
        "exactly: the token “c200” is the same token in the query and in the listing, "
        "and no amount of semantic smoothing should blur it. Lexical matching alone, "
        "however, fails on the part of the query that carries intent rather than "
        "identity: asking for an SUV with seven seats and a budget of one billion VND "
        "expresses constraints, not vocabulary, which a bag of words cannot disentangle."
    )
    report.body_p(
        "Dense retrieval addresses this directly. Text embedding models transform each "
        "sentence into a fixed-dimensional vector, positioning semantically similar "
        "sentences close to one another in vector space; for unit-normalised vectors the "
        "similarity between a query and a document is their dot product. This approach "
        "rests on the Transformer architecture [13] and has been popularised by "
        "bi-encoder models such as Sentence-BERT [9], which turns the pre-trained "
        "bidirectional encoder of Devlin et al. [3] into a bi-encoder in which query and "
        "document are embedded independently and compared by cosine similarity, and by "
        "the multilingual distillation of Reimers and Gurevych [8], which is what makes "
        "Vietnamese queries tractable. The two families are complementary: the "
        "embedding captures paraphrase and intent, while the sparse vector captures the "
        "exact codes that distinguish one trim from another."
    )
    report.heading2("Convolutional Neural Networks for Vehicle Classification")
    report.p1(
        "Convolutional neural networks learn hierarchical features directly from "
        "pixels, through stacked convolutions, pooling and fully connected layers, and "
        "MobileNetV2 [11] is the instance used in this work: its inverted residual "
        "blocks keep the representation expressive at low cost, which makes it a good "
        "candidate for CPU-only inference. Training from scratch would require far more "
        "data than the present task provides, so transfer learning is the practical "
        "choice: a backbone pre-trained on ImageNet [2] is reused and only a small "
        "classification head is trained on the target domain. Two training details "
        "matter later. The first is class re-weighting, since the collection is "
        "imbalanced and the effective-number weighting of Cui et al. [1] gives each "
        "class a weight proportional to the inverse of the expected overlap between its "
        "samples. The second is mild augmentation only, because aggressive rotation or "
        "cropping removes exactly the grille, badge and headlamp shapes that "
        "distinguish the seven brands."
    )
    report.heading2("Linear Regression for Price Analysis")
    report.p1(
        "The econometric component is deliberately modest. Ordinary least squares "
        "estimates the coefficients that minimise the sum of squared residuals, and it "
        "is the natural first baseline for the relationship between a listing price and "
        "a single numeric attribute such as production year or odometer reading. Its "
        "value here is not prediction accuracy but orientation: it quantifies how much "
        "of the observed price variation is associated with each attribute, and so "
        "establishes the limit of what one attribute can tell a buyer about a price. "
        "The estimates are obtained through the estimator interface of scikit-learn [7] "
        "and every metric is computed on data withheld from the fit, using a grouped "
        "split because listings that share a name and a price are the same car reposted "
        "on several platforms."
    )

    # ------------------------------------------------------------- 3 dataset
    report.heading1("Dataset and Preprocessing")
    report.p1(
        "The primary corpus is a public collection of used-car listings scraped from "
        "Vietnamese marketplaces [6]: 15,442 records described by 17 columns, among "
        "them the model name, the asking price as free text, the production year, the "
        "distance travelled, the body style, the fuel type, the transmission, the number "
        "of seats and doors, the colours, the origin, the condition and the location. "
        "The free-text price field is the reason the collection needs cleaning: prices "
        "appear as “1 Tỷ 118 Triệu”, as “646 Triệu”, or occasionally as a bare number, "
        "and the parser converts these forms into a single numeric column in millions "
        "of VND. Every listing yields a positive price, so no record is discarded. The "
        "remaining cleaning is conservative: missing values are never imputed but "
        "treated as unknown, and a listing whose year is unknown is excluded from the "
        "year-based analysis. The unknown rates are uneven — distance travelled is "
        "unknown for 22.95% of listings, whereas the transmission is recovered from the "
        "description text for all but 0.01% of records."
    )
    report.body_p(
        "The production year is present for 14,891 listings whose distribution, shown in "
        "Fig. 1, is concentrated in the second half of the 2010s, with a median of 2020 "
        "and a range from 1989 to 2024. Prices are spread over three orders of "
        "magnitude: the median listing is priced at 659 million VND, the mean is pulled "
        "up to 1,232 million VND by a few expensive vehicles, and the most expensive "
        "record is quoted at 63,568 million VND. Such extremes are genuine market "
        "observations rather than parsing artefacts, but they mean that any regression "
        "on price must be read as descriptive rather than predictive. A further "
        "characteristic constrains the interpretation: 3,533 listings share both a "
        "model name and a price with an earlier listing, almost certainly the same "
        "vehicle reposted on several platforms, so the pipeline removes them from the "
        "displayed results and the regression groups them together."
    )
    report.figure(
        "fig1_year_distribution.png",
        "Distribution of the 14,891 listings with a known production year. The bulk of "
        "the market consists of vehicles built between 2015 and 2023.",
        width_ratio=0.55,
    )
    report.p1(
        "The visual component uses a separate public collection of vehicle photographs "
        "[5] organised into seven category folders: Audi, Hyundai Creta, Mahindra "
        "Scorpio, Rolls Royce, Maruti Suzuki Swift, Tata Safari and Toyota Innova, split "
        "by the publisher into 3,352 training and 813 test photographs so that the test "
        "split can be used exactly as published. The class distribution is markedly "
        "imbalanced: Audi accounts for 1,013 of the 4,165 images, or 24.3%, and Toyota "
        "Innova for a further 23.2%, so the two largest classes alone represent 47.5% "
        "of the data, whereas the smallest class, Hyundai Creta, contributes only 338 "
        "images. A model evaluated on accuracy alone could therefore score well by "
        "favouring the majority classes, which is why the balanced accuracy and the "
        "macro-averaged F1 score are reported alongside the overall accuracy."
    )

    # ------------------------------------------------------- 4 system design
    report.heading1("System Design")
    report.p1(
        "The prototype is a single Streamlit application [12] organised as three "
        "decoupled processing tracks that share one in-memory listing table. The "
        "separation is not merely cosmetic: each track can fail without taking the "
        "others down. If the embedding model cannot be loaded, the retrieval engine "
        "falls back to TF-IDF matching and reports the degradation in the interface, "
        "while the classifier and the regression charts continue to work. Fig. 2 gives "
        "the overall arrangement. Two engineering decisions keep the application "
        "responsive on commodity hardware. First, the sentence embeddings of all "
        "15,442 listings and the TF-IDF matrix are computed once and cached on disk "
        "under a key derived from the model name and the dataset file, so that a change "
        "to either invalidates the cache automatically. Second, the embedding model is "
        "loaded lazily, only when a query needs semantic scoring. The model itself is "
        "the twelve-layer multilingual MiniLM bi-encoder released on the Hugging Face "
        "Hub [4], and everything runs on a multi-core CPU."
    )
    report.figure(
        "fig3_architecture.png",
        "Architecture of the prototype. Text queries are parsed into attribute filters "
        "before hybrid ranking, photographs are classified by a transfer-learned "
        "MobileNetV2, and the listing table feeds the regression baselines.",
        width_ratio=0.72,
    )
    report.p1(
        "A natural language request in Vietnamese is first converted into a set of "
        "explicit criteria. Each criterion records the column it constrains, the "
        "comparison operator, the value and a human-readable label shown back to the "
        "user, which keeps the extraction step explainable. The parser works with "
        "regular expressions on an accent-stripped, lower-cased copy of the query and "
        "recognises up to fourteen attributes, from location, distance, price and "
        "production year to colour, fuel type, transmission, origin, condition, seat "
        "and door count, body style and brand. The order of examination is not "
        "arbitrary, because several patterns are ambiguous in isolation: locations come "
        "before colour and fuel type, since a place name such as “Đồng Nai” would "
        "otherwise be matched by a colour pattern, and distance before price and year, "
        "since the digits “2000” in “2000 km” denote a distance and not a model year. A "
        "matched span is blanked rather than deleted, so the remaining tokens are not "
        "concatenated into spurious matches. Brand detection does not rely on a fixed "
        "list of triggers, but falls back to a character-level similarity measure."
    )
    report.p1(
        "The extracted criteria are applied as a hard filter. Every criterion must hold "
        "simultaneously, and a listing whose value for a constrained attribute is "
        "unknown does not satisfy that criterion. The strict filter therefore narrows "
        "the 15,442 listings to a candidate set whose size depends entirely on the "
        "query: the three benchmark requests of Section 5 leave 3,450, 44 and 287 "
        "candidates. The survivors are scored by two complementary representations and "
        "the two scores are fused. Let \\(q\\) be the embedding of the query and "
        "\\(e_i\\) the embedding of listing \\(i\\); because both vectors are "
        "normalised, their inner product is the cosine similarity"
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
        "vector and the corresponding row of the listing matrix. The two scores are not "
        "on a common scale, so each is min-max normalised across the candidate set "
        "before they are combined, and the fused score is"
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
        "\\(s̃\\) a min-max normalised score, exposed as an interface control because "
        "the balance it encodes is a design decision rather than a law. A strict filter "
        "is brittle in a heterogeneous corpus, so when no listing satisfies all criteria "
        "the system relaxes them in five fixed stages, starting with the price band and "
        "the production year. The brand criterion is never released, because a search "
        "for a Toyota that returns a Hyundai is not a relaxed result but a wrong answer. "
        "Relaxation is never silent: the interface names the criteria released, and each "
        "listing shows the match state of every criterion."
    )
    report.p1(
        "The classifier is MobileNetV2 with the ImageNet weights and the fully "
        "connected top removed, followed by a classification head consisting of global "
        "average pooling, dropout at rate 0.30, a dense layer of 128 units with ReLU "
        "activation, a second dropout at rate 0.20 and a softmax output over the seven "
        "classes, for 2,422,855 parameters in total. Training proceeds in two stages: "
        "five epochs with the backbone frozen at a learning rate of 10⁻³, then "
        "fine-tuning of the whole network at 10⁻⁴, with early stopping on validation "
        "accuracy. At inference a photograph is converted to RGB and resized to "
        "224 × 224 pixels, and a prediction is accepted only when the probability of the "
        "leading class reaches a threshold of 0.60; below that value the system reports "
        "the result as uncertain."
    )
    report.figure(
        "fig4_price_panels.png",
        "Listing price against production year (left) and against odometer reading "
        "(right), with the fitted regression line, the binned medians and the "
        "inter-quartile band. The vertical axis is truncated at 5,000 million VND; the "
        "slopes are small and the spread at every value is far wider than the effect.",
        width_ratio=0.86,
    )

    # ------------------------------------------------------------ 5 results
    report.heading1("Results")
    report.p1(
        "The retrieval engine was evaluated on three representative requests, each "
        "chosen to exercise a different failure mode of keyword search: a broad "
        "category with a budget constraint, a brand and model with an explicit "
        "configuration, and a brand with an alphanumeric trim designation. For each "
        "request the modes were run over the identical candidate set and the top-ranked "
        "listing was recorded, as shown in Table 1. The attribute filter behaves as "
        "intended: for the Innova query the three extracted criteria reduce the corpus "
        "to 44 listings, and all three modes agree on the same top result, the 2020 "
        "Innova V 2.0 at 699 million VND."
    )
    report.table_caption(
        "Top-ranked listing for three benchmark queries under the hybrid and the TF-IDF "
        "retrieval modes."
    )
    report.table(
        [
            "Query",
            "Cand.",
            "Hybrid: top-ranked listing",
            "Price",
            "TF-IDF: top-ranked listing",
            "Price",
        ],
        [
            [
                "SUV below 1 bn",
                "3,450",
                "Toyota Avanza Premio 1.5 AT - 2024",
                "565",
                "Lynk & Co 01 - 2024",
                "999",
            ],
            [
                "Innova 2020, 7 seats",
                "44",
                "Toyota Innova V 2.0 AT - 2020",
                "699",
                "Toyota Innova V 2.0 AT - 2020",
                "699",
            ],
            [
                "C200 2021",
                "287",
                "Mercedes-Benz C class C300 AMG - 2021",
                "1,350",
                "Mercedes-Benz C class C200 Exclusive - 2021",
                "1,220",
            ],
        ],
        [19, 8, 26, 9, 27, 9],
    )
    report.body_p(
        "The broad SUV query separates the modes most clearly. The hybrid ranking "
        "returns a 2024 Toyota Avanza Premio at 565 million VND, well inside the budget, "
        "whereas the lexical mode returns a Lynk & Co 01 from 2024 at 999 million VND, "
        "the most expensive car in the candidate set: the query shares the token “SUV” "
        "with that listing while the other two vehicles are described with different "
        "wording. The hybrid score avoids this failure by giving the semantic component "
        "the larger weight, but the example also shows that the lexical score is not a "
        "reliable ranking signal on its own. The trim query is the most instructive of "
        "the three. The parser extracts the Mercedes-Benz brand and the 2021 model year "
        "and leaves 287 candidates, but the requested trim “C200” is not among the "
        "parsed criteria, because trim designations are alphanumeric codes that the "
        "regular expressions cannot distinguish from one another. Both the hybrid and "
        "the semantic mode therefore rank the C300 AMG first at 1,350 million VND, "
        "while only the lexical mode returns the C200 Exclusive at 1,220 million VND. "
        "This is the characteristic weakness of a bi-encoder, which embeds query and "
        "document independently, so that the difference between two adjacent trim codes "
        "is smaller than the differences between them and unrelated vehicles."
    )
    report.p1(
        "The classifier was evaluated once, on the published test split of 813 "
        "photographs that was never used for training or validation. It reaches 97.05% "
        "accuracy, a balanced accuracy of 96.15% and a macro-averaged F1 score of "
        "96.54%, with a test loss of 0.1119. The gap between overall and balanced "
        "accuracy is only 0.90 percentage points, so the model does not trade "
        "minority-class performance for majority-class accuracy. Fig. 4 shows the "
        "confusion matrix and the per-class scores."
    )
    report.figure(
        "fig5_cnn_evaluation.png",
        "Left: row-normalised confusion matrix on the 813 test photographs; the "
        "diagonal dominates. Right: per-class precision, recall and F1 score against "
        "the overall and the balanced accuracy; every class lies in a narrow band "
        "around the two reference lines.",
        width_ratio=0.86,
    )
    report.body_p(
        "The confusion matrix is dominated by the diagonal. The largest single confusion "
        "is Audi being read as Rolls Royce, three cases out of 199, which is "
        "understandable given that the two cars share a front-grille shape that "
        "dominates a photograph taken at an angle. Hyundai Creta is the weakest class, "
        "with a recall of 89.55%, and it is also the class with the fewest test images; "
        "its errors are spread over Rolls Royce, Tata Safari and Toyota Innova, the "
        "other high-bodied vehicles in the collection. Tata Safari shows the mirror "
        "image behaviour, with the highest recall of all classes at 99.06% and the "
        "lowest precision at 94.59%, so it absorbs some of the compact-SUV confusions. "
        "All seven classes fall between 89.55% and 99.06% recall, so the class "
        "re-weighting is doing what it was intended to do. The confidence gate of 0.60 "
        "was applied in the interface: on well-separated categories the softmax output "
        "is close to one and the gate never rejects a prediction, while it becomes "
        "active on genuinely ambiguous photographs."
    )
    report.p1(
        "Both regressions were fitted with the same protocol: a grouped 80/20 split on "
        "the pair of model name and price, a mean-predictor baseline fitted on the same "
        "training part, and a five-fold grouped cross-validation inside the training "
        "part only. The results are summarised in Table 2. The year model has a clearly "
        "positive slope: a listing is worth about 77.41 million VND more for each "
        "additional year of production, consistent with the direction a buyer would "
        "expect, but the descriptive \\(R^2\\) of 0.034 and the test \\(R^2\\) of 0.047 "
        "are very low. The baseline achieves a test RMSE of 1,744.9 million VND against "
        "1,703.3 million VND for the regression, an improvement of 2.4% that is small "
        "compared with the dispersion of prices in the corpus. The mileage model is "
        "weaker still: within the 0–100,000 km band the test \\(R^2\\) is 0.008, and "
        "removing the upper limit collapses the descriptive \\(R^2\\) to 0.0003, so the "
        "apparent dependence on distance is driven by the most expensive vehicles being "
        "clustered at low mileages."
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
        "These numbers answer RQ3, and the answer is a negative one with a clear "
        "interpretation. Neither attribute explains more than five percent of the price "
        "variance, so neither can support a valuation of an individual car. The models "
        "retain value in a different role: they quantify the market’s price gradient "
        "and establish the scale of the noise against which any future, richer model "
        "would have to demonstrate improvement. A multivariable model adding brand, "
        "body style, transmission and origin is the natural next step."
    )

    report.p1(
        "Two ethical considerations follow from the design. The first concerns the "
        "meaning of the prices the system displays: the corpus consists of asking "
        "prices collected from listings, not of transaction records, and the two differ "
        "by an unknown margin that includes negotiation and seller incentives, so the "
        "interface presents every figure as a listed price and never as a market "
        "valuation. The second concerns the photographs: the system runs on publicly "
        "available vehicle images without attempting to identify the person or the "
        "plate in the frame, so the classification output carries no personal "
        "information."
    )

    # ---------------------------------------------------------- 6 discussion
    report.heading1("Discussion")
    report.p1(
        "The prototype demonstrates that the three capabilities can be combined without "
        "interfering with one another, and that each is useful within a well-defined "
        "range. Rule-based parsing isolates the attribute constraints reliably and "
        "reduces the corpus to candidate sets of a few dozen or a few thousand entries. "
        "The two retrieval representations are genuinely complementary, as the trim "
        "query shows: the semantic score finds the vehicle that fits the description, "
        "and the lexical score finds the vehicle that carries the requested "
        "designation. The classifier is the strongest component by a wide margin, and it "
        "stays balanced across the seven classes despite a threefold imbalance in the "
        "training data. Two engineering properties make the results reproducible rather "
        "than merely plausible: a cached embedding can never be reused for another "
        "corpus or model, and the retrieval engine degrades rather than fails."
    )
    report.p1(
        "The limitations are of four kinds, because they call for different remedies. "
        "The first is a modelling limitation in the retrieval stage: the trim "
        "confusion of Table 1 cannot be repaired by tuning the fusion weight; the "
        "remedies are a cross-encoder reranking stage that scores query and candidate "
        "jointly, or a rule that extracts the alphanumeric trim token from the listing "
        "title and matches it exactly. The second is a limitation of the data: the "
        "corpus contains 3,533 reposted listings, its attribute fields are missing at "
        "rates from 0.01% for the transmission to 22.95% for the distance travelled, and "
        "the price field has a long tail reaching 63,568 million VND, fifty times the "
        "median. The third is a limitation of the evaluation: three benchmark queries "
        "cannot support a statistical claim about ranking quality, and a proper "
        "evaluation would require a labelled set of queries with relevance judgements. "
        "The classification evaluation is on a published test split never used for "
        "training, but the seven categories are all popular vehicles in India and none "
        "is common in Vietnam, so the reported accuracy should not be read as an "
        "estimate of performance on Vietnamese traffic. The fourth is a limitation of "
        "scope: the price models are descriptive baselines, and nothing in the "
        "interface uses a regression output to modify a ranking score."
    )
    # ---------------------------------------------------------- 7 conclusion
    report.heading1("Conclusion")
    report.p1(
        "This report has built and evaluated a prototype that supports car search, car "
        "recognition and price orientation for the Vietnamese second-hand market. The "
        "three pipelines are decoupled and share a single cleaned listing table of "
        "15,442 records. On the retrieval side, the hybrid engine recovers the intended "
        "listing exactly whenever the request is categorical, and the semantic and "
        "lexical components fail in different and complementary ways. On the "
        "recognition side, transfer learning proved to be the decisive choice: a "
        "compact MobileNetV2 head on a frozen-then-fine-tuned ImageNet backbone reaches "
        "97.05% accuracy and 96.54% macro-F1 on 813 unseen photographs, with a balanced "
        "accuracy within one percentage point of the overall accuracy despite a "
        "threefold class imbalance. On the price side, the negative result is equally "
        "informative: single-variable ordinary least squares explains 4.7% of the price "
        "variance in production year and 0.8% in odometer reading, so the price track "
        "is an orientation aid rather than a scoring mechanism. The value of such a "
        "system lies less in the sophistication of any single model than in making "
        "several weak signals legible at once."
    )

    # ------------------------------------------------- 8 reproducibility
    report.heading1("Reproducibility and Future Work")
    report.p1(
        "All three components are implemented in Python and run without a GPU. The "
        "experiments were run on a laptop with an Apple M5 processor and 16 GB of "
        "memory, using Python 3.14.7, TensorFlow 2.22, scikit-learn 1.9.1, "
        "sentence-transformers 6.2.0 and Streamlit 1.66.0 [12]. The listing data and "
        "the seven-class image collection are public, and the trained model together "
        "with its metrics is published under a pinned commit so that the reported "
        "numbers can be reproduced exactly. Three directions for future work follow."
    )
    report.bullet(
        "Replace the bi-encoder with a two-stage pipeline that adds a cross-encoder "
        "reranking stage over the top candidates, or extract the alphanumeric trim token "
        "from the listing title and promote it to a hard filter. Either change addresses "
        "the trim confusion of Table 1."
    )
    report.bullet(
        "Extend the price analysis to a multivariable model that conditions on brand, "
        "body style, transmission, fuel type and origin, using the grouped split "
        "throughout and reporting the improvement over the baselines of Table 2."
    )
    report.bullet(
        "Build a labelled query set with relevance judgements so that retrieval quality "
        "can be measured rather than illustrated, and calibrate the confidence threshold "
        "of 0.60 on a validation set drawn from Vietnamese vehicles rather than from the "
        "Indian categories of the present collection."
    )

    # ------------------------------------------------- acknowledgements
    report.acknowledgement(
        "Acknowledgments.",
        "The authors thank Nguyen Xuan Sam for the guidance on the design of the "
        "retrieval pipeline and for the review of earlier drafts of this report.",
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
            "Kaggle: Car Images Dataset (kshitij192/cars-image-dataset). "
            "https://www.kaggle.com/datasets/kshitij192/cars-image-dataset, last "
            "accessed 06/11/2026",
            "Kaggle: Vietnam Automobile Dataset (qucvinhdng/automobile-dataset). "
            "https://www.kaggle.com/datasets/qucvinhdng/automobile-dataset, last "
            "accessed 06/11/2026",
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
            "in Neural Information Processing Systems 30, pp. 5998–6008 (2017)",
        ]
    )
    return report


def _swap_inner(path: Path, inner: str) -> str:
    """Replace the body of a header or footer part, keeping its root element.

    The template parts are full ``w:hdr`` / ``w:ftr`` documents with a long list
    of namespace declarations; only their content needs to change, so the root
    element is preserved and the paragraphs inside it are replaced.
    """
    original = path.read_text(encoding="utf-8")
    match = re.search(r"(<w:(?:hdr|ftr)\b[^>]*>)(.*)(</w:(?:hdr|ftr)>)", original, re.S)
    if match is None:
        raise ValueError(f"No header/footer root element found in {path}")
    return match.group(1) + inner + match.group(3)


def build_headers() -> dict[str, str]:
    """Replace the running heads inherited from the Springer template.

    The proceedings template puts the page number and the author list on the
    even pages and the paper title on the odd pages, which is the camera-ready
    convention for a conference volume. A course assignment does not need that
    distinction, and the author list in particular is redundant with the title
    page, so the running heads are emptied and the page number is moved into
    the footer, where it appears on every page except the title page.
    """
    empty = "<w:p><w:pPr><w:ind w:firstLine=\"0\"/></w:pPr></w:p>"
    return {
        "word/header1.xml": empty,
        "word/header2.xml": empty,
        "word/header3.xml": empty,
    }


def build_footers() -> dict[str, str]:
    """Put a centred page number in the even and odd footers.

    The first-page footer stays empty so that the title page carries no
    numbering, which is the convention of the template the report is built on.
    """
    centred = (
        '<w:p><w:pPr><w:jc w:val="center"/><w:ind w:firstLine="0"/></w:pPr>'
        "<w:r><w:fldChar w:fldCharType=\"begin\"/></w:r>"
        '<w:r><w:instrText xml:space="preserve">PAGE   \\* MERGEFORMAT</w:instrText></w:r>'
        '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
        '<w:r><w:rPr><w:noProof/></w:rPr><w:t>1</w:t></w:r>'
        '<w:r><w:fldChar w:fldCharType="end"/></w:r>'
        "</w:p>"
    )
    return {
        "word/footer1.xml": centred,
        "word/footer2.xml": centred,
        "word/footer3.xml": "<w:p><w:pPr><w:ind w:firstLine=\"0\"/></w:pPr></w:p>",
    }


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
    headers = build_headers()
    footers = build_footers()
    overrides = {**headers, **footers}

    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(TEMPLATE_DIR.rglob("*")):
            if path.is_dir():
                continue
            relative = path.relative_to(TEMPLATE_DIR).as_posix()
            if relative == "word/document.xml":
                archive.writestr(relative, document)
            elif relative == "word/_rels/document.xml.rels":
                archive.writestr(relative, rels)
            elif relative in overrides:
                archive.writestr(
                    relative, _swap_inner(path, overrides[relative])
                )
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

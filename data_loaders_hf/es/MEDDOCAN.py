# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MEDDOCAN: Spanish clinical case reports annotated with protected health information."""

from pathlib import Path

from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value
from syntok import segmenter
from syntok.tokenizer import Tokenizer

_CITATION = """\
@inproceedings{marimon2019automatic,
	title={Automatic De-identification of Medical Texts in Spanish: the MEDDOCAN Track,
	Corpus, Guidelines, Methods and Evaluation of Results},
	author={Marimon, Montserrat and Gonzalez-Agirre, Aitor and Intxaurrondo, Ander and
	Rodriguez, Heidy and Martin, Jose Lopez and Villegas, Marta and Krallinger, Martin},
	booktitle={Proceedings of the Iberian Languages Evaluation Forum (IberLEF 2019),
	CEUR Workshop Proceedings},
	pages={618--638},
	year={2019}
}
"""

_DESCRIPTION = """\
MEDDOCAN (Medical Document Anonymization) is a collection of 1000 Spanish clinical case reports
drawn from the Spanish Clinical Case Corpus (SPACCC) and enriched by expert annotators with
protected health information. It carries 22 of the 29 PHI types its guidelines define, and is
distributed as 500 clinical cases for training and 250 each for development and test.
"""

_HOMEPAGE = "https://temu.bsc.es/meddocan/"

_LICENSE = "CC_BY_4p0"

# Official BSC release, in Brat standoff: one .txt and one .ann per clinical case
_URL = "https://zenodo.org/records/4279323/files/meddocan.zip"

SUBSETS = ["source"]

_DIRECTORIES = {"train": "train", "validation": "dev", "test": "test"}

_TYPES = [
    "CALLE",
    "CENTRO_SALUD",
    "CORREO_ELECTRONICO",
    "EDAD_SUJETO_ASISTENCIA",
    "FAMILIARES_SUJETO_ASISTENCIA",
    "FECHAS",
    "HOSPITAL",
    "ID_ASEGURAMIENTO",
    "ID_CONTACTO_ASISTENCIAL",
    "ID_EMPLEO_PERSONAL_SANITARIO",
    "ID_SUJETO_ASISTENCIA",
    "ID_TITULACION_PERSONAL_SANITARIO",
    "INSTITUCION",
    "NOMBRE_PERSONAL_SANITARIO",
    "NOMBRE_SUJETO_ASISTENCIA",
    "NUMERO_FAX",
    "NUMERO_TELEFONO",
    "OTROS_SUJETO_ASISTENCIA",
    "PAIS",
    "PROFESION",
    "SEXO_SUJETO_ASISTENCIA",
    "TERRITORIO",
]

# all B- then all I-, as in the PlanTL CoNLL conversions the other Spanish loaders read
_LABELS = ["O"] + [f"B-{kind}" for kind in _TYPES] + [f"I-{kind}" for kind in _TYPES]

FEATURES = Features(
    {
        "id": Value("string"),
        "document_id": Value("string"),
        "tokens": [Value("string")],
        "ner_tags": Sequence(ClassLabel(names=_LABELS)),
    }
)

# beyond this a line is prose, not a header field, and long enough to reach the position limit
_MAX_LINE_TOKENS = 128

_tokenizer = Tokenizer()


def build(subset, data_dir, dl):
    root = Path(dl.download_and_extract(_URL)) / "meddocan"
    return DatasetDict(
        {
            split: Dataset.from_list(
                list(_read_brat(root / name / "brat")), features=FEATURES
            )
            for split, name in _DIRECTORIES.items()
        }
    )


def _read_brat(directory):
    identifier = 0
    for text_file in sorted(directory.glob("*.txt")):
        text = text_file.read_text(encoding="utf-8")
        spans = _read_annotations(text_file.with_suffix(".ann"))

        for sentence in _sentences(text):
            yield {
                "id": str(identifier),
                "document_id": text_file.stem,
                "tokens": [value for _, _, value in sentence],
                "ner_tags": _tag(sentence, spans),
            }
            identifier += 1


def _read_annotations(annotation_file):
    """Brat T lines: mention id, "TYPE start end", surface form."""
    spans = []

    with open(annotation_file, encoding="utf-8") as fd:

        for line in fd:
            _, mention, _ = line.rstrip("\n").split("\t")
            kind, start, end = mention.split(" ")
            spans.append((int(start), int(end), kind))

    return sorted(spans)


def _sentences(text):
    """The line is the unit: no annotation crosses a newline, whereas sentence-splitting the
    "Domicilio: Calle ..." header lines cuts one CALLE mention in six after "Calle."."""
    offset = 0

    for line in text.split("\n"):

        tokens = _offsets(_tokenizer.tokenize(line), offset)

        if len(tokens) > _MAX_LINE_TOKENS:
            for paragraph in segmenter.analyze(line):
                for sentence in paragraph:
                    yield _offsets(sentence, offset)
        elif tokens:
            yield tokens

        offset += len(line) + 1


def _offsets(tokens, offset):
    """Brat offsets are absolute; syntok reports them from the start of the string it was given."""
    return [
        (offset + token.offset, offset + token.offset + len(token.value), token.value)
        for token in tokens
    ]


def _tag(sentence, spans):
    tags = ["O"] * len(sentence)

    for start, end, kind in spans:
        # 455 of the 22795 mentions have a boundary inside a token, which then carries the tag;
        # 3 pairs share one token ("06006.Badajoz") and IOB has no way to keep them apart
        covered = [i for i, (a, b, _) in enumerate(sentence) if a < end and b > start]
        for position, index in enumerate(covered):
            tags[index] = f"{'I' if position else 'B'}-{kind}"

    return tags

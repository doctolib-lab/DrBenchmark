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

"""DisTEMIST: Spanish clinical cases annotated with disease mentions."""

import csv
import random
import re
from pathlib import Path

from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value
from syntok import segmenter

_CITATION = """\
@inproceedings{miranda2022overview,
	title={Overview of DisTEMIST at BioASQ: Automatic detection and normalization of diseases
	from clinical texts: results, methods, evaluation and multilingual resources},
	author={Miranda-Escalada, Antonio and Gasco, Luis and Lima-L{\'o}pez, Salvador and
	Farr{\'e}-Maduell, Eul{\`a}lia and Estrada, Darryl and Nentidis, Anastasios and
	Krithara, Anastasia and Katsimpras, Georgios and Paliouras, Georgios and Krallinger, Martin},
	booktitle={Working Notes of Conference and Labs of the Evaluation (CLEF) Forum,
	CEUR Workshop Proceedings},
	year={2022}
}
"""

_DESCRIPTION = """\
DisTEMIST is a collection of 1000 Spanish clinical case reports manually annotated with disease
mentions by clinical experts. It is the Spanish gold seed from which the disease layer of the
multilingual clinical corpora is projected, and its subtrack 1 is the mention detection task this
loader exposes.
"""

_HOMEPAGE = "https://temu.bsc.es/distemist/"

_LICENSE = "CC_BY_4p0"

_URL = "https://zenodo.org/records/7614764/files/distemist_zenodo.zip"

SUBSETS = ["source"]

_TRAIN = "training"
_TEST = "test_annotated"

# the shared task ships 750 training and 250 test cases and no development set
_SPLIT_SEED = 42
_VALIDATION_SHARE = 0.2

_LABELS = ["O", "B-ENFERMEDAD", "I-ENFERMEDAD"]

_SWALLOWED = re.compile(r"\S+")

FEATURES = Features(
    {
        "id": Value("string"),
        "document_id": Value("string"),
        "tokens": [Value("string")],
        "ner_tags": Sequence(ClassLabel(names=_LABELS)),
    }
)


def build(subset, data_dir, dl):
    root = Path(dl.download_and_extract(_URL)) / "distemist_zenodo"

    training = sorted((root / _TRAIN / "text_files").glob("*.txt"))
    random.Random(_SPLIT_SEED).shuffle(training)
    cut = int(len(training) * _VALIDATION_SHARE)

    cuts = {
        "train": (training[cut:], root / _TRAIN),
        "validation": (training[:cut], root / _TRAIN),
        "test": (sorted((root / _TEST / "brat").glob("*.txt")), root / _TEST),
    }
    return DatasetDict(
        {
            split: Dataset.from_list(
                list(_read_documents(paths, _read_mentions(directory))),
                features=FEATURES,
            )
            for split, (paths, directory) in cuts.items()
        }
    )


def _read_mentions(directory):
    """Subtrack 1 columns: filename, mention id, label, offsets, surface form."""
    mentions = {}

    for annotations in (directory / "subtrack1_entities").glob("*.tsv"):
        with open(annotations, encoding="utf-8") as fd:
            for row in csv.DictReader(fd, delimiter="\t"):
                mentions.setdefault(row["filename"], []).append(
                    (int(row["off0"]), int(row["off1"]), row["label"])
                )

    return mentions


def _read_documents(paths, mentions):
    identifier = 0

    for text_file in paths:
        text = text_file.read_text(encoding="utf-8")
        spans = sorted(mentions.get(text_file.stem, []))

        for sentence in _sentences(text):
            yield {
                "id": str(identifier),
                "document_id": text_file.stem,
                "tokens": [value for _, _, value in sentence],
                "ner_tags": _tag(sentence, spans),
            }
            identifier += 1


def _sentences(text):
    """The sentence is the unit: unlike MEDDOCAN these cases carry no header lines, so no mention
    is cut, and the line would otherwise reach 746 tokens against 242 for the sentence."""
    for paragraph in segmenter.analyze(text):
        for sentence in paragraph:
            spans = [
                (token.offset, token.offset + len(token.value), token.value)
                for token in sentence
            ]
            yield _restore(spans, text)


def _restore(spans, text):
    """syntok reports "-" and "_" as spacing rather than as tokens, which would drop them from the
    text; the official SPACCC tokenisation these clinical cases also ship under keeps them."""
    restored = []

    for index, span in enumerate(spans):

        if index:
            previous = spans[index - 1][1]
            for run in _SWALLOWED.finditer(text[previous : span[0]]):
                start = previous + run.start()
                restored.append((start, start + len(run.group()), run.group()))

        restored.append(span)

    return restored


def _tag(sentence, spans):
    tags = ["O"] * len(sentence)

    # 214 mention pairs are nested, which a flat IOB scheme cannot hold: the widest one wins,
    # so every span that survives is one the corpus annotates, at the cost of 223 inner ones
    for start, end, kind in sorted(spans, key=lambda span: (span[0], -span[1])):
        covered = [i for i, (a, b, _) in enumerate(sentence) if a < end and b > start]
        if any(tags[index] != "O" for index in covered):
            continue
        for position, index in enumerate(covered):
            tags[index] = f"{'I' if position else 'B'}-{kind}"

    return tags

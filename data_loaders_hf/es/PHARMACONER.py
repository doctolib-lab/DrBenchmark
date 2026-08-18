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

"""PharmaCoNER: Spanish clinical cases annotated with substance, compound and protein mentions."""

from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value

_CITATION = """\
@inproceedings{gonzalez-agirre-etal-2019-pharmaconer,
	title = "{P}harma{C}o{NER}: Pharmacological Substances, Compounds and proteins Named Entity
	Recognition track",
	author = {Gonzalez-Agirre, Aitor and Marimon, Montserrat and Intxaurrondo, Ander and
	Rabal, Obdulia and Villegas, Marta and Krallinger, Martin},
	booktitle = "Proceedings of The 5th Workshop on BioNLP Open Shared Tasks",
	month = nov,
	year = "2019",
	address = "Hong Kong, China",
	publisher = "Association for Computational Linguistics",
	url = "https://aclanthology.org/D19-5701",
	doi = "10.18653/v1/D19-5701",
	pages = "1--10"
}
"""

_DESCRIPTION = """\
PharmaCoNER is a collection of 1000 Spanish clinical cases drawn from the Spanish Clinical Case
Corpus (SPACCC), manually annotated by medicinal chemistry experts with four entity types:
NORMALIZABLES, NO_NORMALIZABLES, PROTEINAS and UNCLEAR. It holds 396,988 words, split into
500 clinical cases for training and 250 each for development and test.
"""

_HOMEPAGE = "https://temu.bsc.es/pharmaconer/"

_LICENSE = "CC_BY_4p0"

# PlanTL mirror of the official BSC distribution, converted from Brat to CoNLL splits
_URL = "https://huggingface.co/datasets/PlanTL-GOB-ES/pharmaconer/resolve/main/"

SUBSETS = ["source"]

_FILES = {
    "train": "train-set_1.1.conll",
    "validation": "dev-set_1.1.conll",
    "test": "test-set_1.1.conll",
}

_LABELS = [
    "O",
    "B-NO_NORMALIZABLES",
    "B-NORMALIZABLES",
    "B-PROTEINAS",
    "B-UNCLEAR",
    "I-NO_NORMALIZABLES",
    "I-NORMALIZABLES",
    "I-PROTEINAS",
    "I-UNCLEAR",
]

FEATURES = Features(
    {
        "id": Value("string"),
        "document_id": Value("string"),
        "tokens": [Value("string")],
        "ner_tags": Sequence(ClassLabel(names=_LABELS)),
    }
)


def build(subset, data_dir, dl):
    paths = dl.download({split: f"{_URL}{name}" for split, name in _FILES.items()})
    return DatasetDict(
        {
            split: Dataset.from_list(list(_read_conll(path)), features=FEATURES)
            for split, path in paths.items()
        }
    )


def _read_conll(data_file):
    # CoNLL columns: token, document id, character offsets, IOB tag
    with open(data_file, encoding="utf-8") as fd:

        identifier = 0
        document_id = ""
        tokens, ner_tags = [], []

        for line in fd:

            if not line.strip():

                if tokens:
                    yield {
                        "id": str(identifier),
                        "document_id": document_id,
                        "tokens": tokens,
                        "ner_tags": ner_tags,
                    }
                    identifier += 1
                    tokens, ner_tags = [], []

                continue

            columns = line.rstrip("\n").split("\t")
            document_id = columns[1]
            tokens.append(columns[0])
            ner_tags.append(columns[-1])

        if tokens:
            yield {
                "id": str(identifier),
                "document_id": document_id,
                "tokens": tokens,
                "ner_tags": ner_tags,
            }

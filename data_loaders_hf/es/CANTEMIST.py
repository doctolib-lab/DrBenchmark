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

"""CANTEMIST-NER: Spanish oncology clinical cases annotated with tumour morphology mentions."""

from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value

_CITATION = """\
@inproceedings{miranda2020named,
	title={Named entity recognition, concept normalization and clinical coding: Overview of the
	cantemist track for cancer text mining in spanish, corpus, guidelines, methods and results},
	author={Miranda-Escalada, A and Farr{\'e}, E and Krallinger, M},
	booktitle={Proceedings of the Iberian Languages Evaluation Forum (IberLEF 2020),
	CEUR Workshop Proceedings},
	year={2020}
}
"""

_DESCRIPTION = """\
CANTEMIST (CANcer TExt Mining Shared Task) is a corpus of 1301 Spanish oncology clinical cases,
annotated at the mention level with tumour morphology entities normalised to eCIE-O-3.1 codes.
This loader exposes the NER subtask (CANTEMIST-NER).
"""

_HOMEPAGE = "https://temu.bsc.es/cantemist/"

_LICENSE = "CC_BY_4p0"

# PlanTL mirror of the official BSC distribution, already segmented into CoNLL splits
_URL = "https://huggingface.co/datasets/PlanTL-GOB-ES/cantemist-ner/resolve/main/"

SUBSETS = ["source"]

_FILES = {"train": "train.conll", "validation": "dev.conll", "test": "test.conll"}

_LABELS = ["O", "B-MORFOLOGIA_NEOPLASIA", "I-MORFOLOGIA_NEOPLASIA"]

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

            if line.startswith("-DOCSTART-") or not line.strip():

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

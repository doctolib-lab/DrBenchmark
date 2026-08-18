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

"""Chilean Waiting List: referrals from Chilean public hospitals, annotated with clinical entities."""

from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value

_CITATION = """\
@inproceedings{baez2020chilean,
	title={The Chilean Waiting List Corpus: a new resource for clinical named entity recognition
	in Spanish},
	author={B{\'a}ez, Pablo and Villena, Felipe and Rojas, Mat{\'\i}as and Dur{\'a}n, Manuel and
	Dunstan, Jocelyn},
	booktitle={Proceedings of the 3rd Clinical Natural Language Processing Workshop},
	pages={291--300},
	year={2020}
}
"""

_DESCRIPTION = """\
The Chilean Waiting List corpus holds referrals to medical specialists from the waiting list of
Chilean public hospitals, annotated with eleven clinical entity types. It is the only open Spanish
clinical corpus written in Latin-American Spanish and in the telegraphic register of real referral
forms, with its own abbreviations and spelling variation.
"""

_HOMEPAGE = "https://zenodo.org/records/7072314"

_LICENSE = "CC_BY_NC_SA_4p0"

# the pooled packaging of the Zenodo release; the per-entity ones (wl-disease, wl-medication, ...)
# would split one corpus into seven recipes
_URL = "https://huggingface.co/datasets/plncmm/wl/resolve/main/"

SUBSETS = ["source"]

_FILES = {"train": "train.conll", "validation": "dev.conll", "test": "test.conll"}

# the order the plncmm distribution declares
_LABELS = [
    "O",
    "B-Disease",
    "I-Disease",
    "B-Medication",
    "I-Medication",
    "B-Abbreviation",
    "I-Abbreviation",
    "B-Body_Part",
    "I-Body_Part",
    "B-Family_Member",
    "I-Family_Member",
    "B-Sign_or_Symptom",
    "I-Sign_or_Symptom",
    "B-Laboratory_or_Test_Result",
    "I-Laboratory_or_Test_Result",
    "B-Clinical_Finding",
    "I-Clinical_Finding",
    "B-Diagnostic_Procedure",
    "I-Diagnostic_Procedure",
    "B-Laboratory_Procedure",
    "I-Laboratory_Procedure",
    "B-Therapeutic_Procedure",
    "I-Therapeutic_Procedure",
]

FEATURES = Features(
    {
        "id": Value("string"),
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
    # two columns, token and IOB tag; a referral carries no identifier of its own
    with open(data_file, encoding="utf-8") as fd:

        identifier = 0
        tokens, ner_tags = [], []

        for line in fd:

            if not line.strip():

                if tokens:
                    yield {
                        "id": str(identifier),
                        "tokens": tokens,
                        "ner_tags": ner_tags,
                    }
                    identifier += 1
                    tokens, ner_tags = [], []

                continue

            columns = line.rstrip("\n").split("\t")
            tokens.append(columns[0])
            ner_tags.append(columns[-1])

        if tokens:
            yield {"id": str(identifier), "tokens": tokens, "ner_tags": ner_tags}

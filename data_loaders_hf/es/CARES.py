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

"""CARES: Spanish radiology reports classified into ICD-10 chapters."""

import pyarrow.parquet as pq
from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value

_CITATION = """\
@article{chizhikova2023cares,
	title={CARES: A Corpus for classification of Spanish Radiological reports},
	author={Chizhikova, Mariia and L{\'o}pez-{\'U}beda, Pilar and Collado-Monta{\~n}ez, Jaime and
	Mart{\'\i}n-Noguerol, Teodoro and D{\'\i}az-Galiano, Manuel C and Luna, Antonio and
	Ure{\~n}a-L{\'o}pez, L Alfonso and Mart{\'\i}n-Valdivia, M Teresa},
	journal={Computers in Biology and Medicine},
	volume={154},
	pages={106581},
	year={2023}
}
"""

_DESCRIPTION = """\
CARES is a collection of 3219 anonymised radiology reports from a Spanish hospital, manually
annotated with ICD-10 codes. This loader exposes the chapter-level task: the set of ICD-10
chapters a report belongs to, between one and nine per report.
"""

_HOMEPAGE = "https://huggingface.co/datasets/chizhikchi/CARES"

_LICENSE = "AFL_3p0"

# the packaging the published Spanish clinical benchmark evaluates, and the only one served
# without an access agreement; the authors' own repository is gated
_URL = "https://huggingface.co/datasets/IIC/caresC/resolve/main/data/"

SUBSETS = ["chapters"]

_FILES = {
    "train": "train-00000-of-00001.parquet",
    "validation": "validation-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}

# the packaging ships a 16-wide multi-hot vector and no chapter names, so the position is the
# only identity a chapter has here
_LABELS = [f"chapter_{index:02d}" for index in range(16)]

FEATURES = Features(
    {
        "id": Value("string"),
        "text": Value("string"),
        "chapters": Sequence(ClassLabel(names=_LABELS)),
    }
)


def build(subset, data_dir, dl):
    paths = dl.download({split: f"{_URL}{name}" for split, name in _FILES.items()})
    return DatasetDict(
        {
            split: Dataset.from_list(list(_read_parquet(path)), features=FEATURES)
            for split, path in paths.items()
        }
    )


def _read_parquet(path):
    """The multi-hot vector becomes the list of chapter ids the family builds its target from."""
    for identifier, row in enumerate(pq.read_table(path).to_pylist()):
        yield {
            "id": str(identifier),
            "text": row["text"],
            "chapters": [index for index, flag in enumerate(row["label"]) if flag],
        }

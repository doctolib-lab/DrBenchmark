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

"""SPACCC-POS: Spanish clinical cases annotated with FreeLing part-of-speech tags."""

import random
from pathlib import Path

from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value

_CITATION = """\
@article{intxaurrondo2018spaccc,
	title={Spanish Clinical Case Corpus (SPACCC): a corpus of clinical cases with
	part-of-speech annotations},
	author={Intxaurrondo, Ander and Marimon, Montserrat and Gonzalez-Agirre, Aitor and
	Lopez-Martin, Jose Antonio and Betcheva, Heidy and Santamaria, Jesus and
	Villegas, Marta and Krallinger, Martin},
	journal={Zenodo},
	year={2018}
}
"""

_DESCRIPTION = """\
SPACCC-POS is the part-of-speech layer of the Spanish Clinical Case Corpus. This loader exposes
its manually annotated portion: 100 clinical cases corrected by two annotators and harmonised
into a gold standard, plus 100 annotated by a single annotator to validate the tagger.
"""

_HOMEPAGE = "https://github.com/PlanTL-GOB-ES/SPACCC_POS"

_LICENSE = "CC_BY_4p0"

_URL = "https://zenodo.org/records/2560344/files/SPACCC_POS.zip"

SUBSETS = ["source"]

# the two manually annotated tenths; the other 800 cases carry the tagger's own output
_MANUAL = ["corpus/development/armonizada", "corpus/validation/anotador1"]

# the corpus has no official splits, so 200 documents are shuffled and cut once
_SPLIT_SEED = 42
_SPLIT_SIZES = (0.6, 0.2)

# FreeLing tags encode morphology in positions 3 and beyond (NCMS000, NCFP000); the corpus
# reduces them to two characters in its own Brat release, which also lands on cas/essai's 31 tags
_TAG_LENGTH = 2

_TYPES = [
    "AO", "AQ", "CC", "CS", "DA", "DD", "DI", "DP", "Fc", "Fd", "Fe", "Fg", "Fh",
    "Fi", "Fp", "Fr", "Fs", "Fx", "Fz", "NC", "NP", "P0", "PD", "PI", "PP", "PR",
    "PT", "RG", "RN", "SP", "VA", "VM", "VS", "Z", "Zd",
]

_LABELS = [f"B-{kind}" for kind in _TYPES]

FEATURES = Features(
    {
        "id": Value("string"),
        "document_id": Value("string"),
        "tokens": [Value("string")],
        "pos_tags": Sequence(ClassLabel(names=_LABELS)),
    }
)


def build(subset, data_dir, dl):
    root = Path(dl.download_and_extract(_URL)) / "SPACCC_POS"
    documents = sorted(
        path for directory in _MANUAL for path in (root / directory).glob("*_tagged")
    )
    random.Random(_SPLIT_SEED).shuffle(documents)

    first = int(len(documents) * _SPLIT_SIZES[0])
    second = first + int(len(documents) * _SPLIT_SIZES[1])
    cuts = {
        "train": documents[:first],
        "validation": documents[first:second],
        "test": documents[second:],
    }
    return DatasetDict(
        {
            split: Dataset.from_list(list(_read_tagged(paths)), features=FEATURES)
            for split, paths in cuts.items()
        }
    )


def _read_tagged(paths):
    """Three space-separated columns, form, lemma and tag, with a blank line between sentences."""
    identifier = 0

    for path in paths:
        tokens, pos_tags = [], []

        for line in path.read_text(encoding="utf-8").splitlines():

            if not line.strip():

                if tokens:
                    yield {
                        "id": str(identifier),
                        "document_id": path.name.replace(".txt_tagged", ""),
                        "tokens": tokens,
                        "pos_tags": pos_tags,
                    }
                    identifier += 1
                    tokens, pos_tags = [], []

                continue

            columns = line.split()
            tokens.append(columns[0])
            pos_tags.append(f"B-{columns[-1][:_TAG_LENGTH]}")

        if tokens:
            yield {
                "id": str(identifier),
                "document_id": path.name.replace(".txt_tagged", ""),
                "tokens": tokens,
                "pos_tags": pos_tags,
            }

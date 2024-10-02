# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#        Copyright (c) -2023 - Mtumbuka F.                                                       #
#        All rights reserved.                                                                       #
#                                                                                                   #
#        Redistribution and use in source and binary forms, with or without modification, are       #
#        permitted provided that the following conditions are met:                                  #    
#        1. Redistributions of source code must retain the above copyright notice, this list of     #
#           conditions and the following disclaimer.                                                #
#        2. Redistributions in binary form must reproduce the above copyright notice, this list of  #
#           conditions and the following disclaimer in the documentation and/or other materials     #
#           provided with the distribution.                                                         #
#                                                                                                   #
#        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS \"AS IS\" AND ANY      #
#        EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF    #
#        MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE #
#        COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,   #
#        EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF         #
#        SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)     #
#        HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR   #
#        TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS         #
#        SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                               #
#                                                                                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


__license__ = "BSD-2-Clause"
__version__ = "2023.1"
__date__ = "18 Sept 2023"
__author__ = "Frank M. Mtumbuka"
__maintainer__ = "Frank M. Mtumbuka"
__email__ = "" ""
__status__ = "Development"

import datapipeline.base_data_loader as base_data_loader
import datapipeline.relation_input_example as relation_input_example
import experiment
import json
import os
import typing


class WikidataDataLoader(base_data_loader.BaseDataLoader):
    """This implements a loader that loads the Wikidata dataset"""

    DEV_REL_FILE = "dev.json"
    """str: The file holding the dev partition data."""

    TEST_REL_FILE = "test.json"
    """str: The file holding the test partition data."""

    TRAIN_REL_FILE = "train.json"
    """str: The file holding the train partition data."""

    def __init__(self, *args, dev: bool = False, test: bool = False, **kwargs):
        """

        Args:
            *args: See `base_data_loader.BaseDataLoader.__init__.
            dev (bool): Specifies whether to load data from the dev partition or not.
            test (bool): Specifies whether to load data from the test partition or not.
            train (bool): Specifies whether to load data from the train partition or not.
            **kwargs: See `base_data_loader.BaseDataLoader.__init__.
        """
        super().__init__(*args, **kwargs)

        rel_data_file = None
        if dev:
            rel_data_file = self.DEV_REL_FILE

        elif test:
            rel_data_file = self.TEST_REL_FILE

        else:
            rel_data_file = self.TRAIN_REL_FILE

        self._rel_data_file = rel_data_file

    def load(self) -> typing.List[relation_input_example.RelationInputExample]:
        samples = []
        data = None
        with open(os.path.join(self._data_path, self._rel_data_file), "r") as f:
            data = json.load(f)
            for idx, sample in enumerate(data):
                try:
                    for item in sample["edgeSet"]:
                        subject_mask = [0] * len(sample["tokens"])
                        object_mask = [0] * len(sample["tokens"])
                        left = item["left"]
                        right = item["right"]
                        left_s = 0
                        left_e = 0
                        right_s = 0
                        right_e = 0
                        if len(left) > 1:
                            left_s = left[0]
                            left_e = left[-1] + 1
                            subj = sample["tokens"][left_s:left_e]
                            subj = " ".join(subj)
                            subj_doc = self._nlp(subj)
                            sub_deps = [t.dep_ for t in subj_doc]
                            sub_root = sub_deps.index("ROOT")
                            subj_mask = [0] * len(sub_deps)
                            subj_mask[sub_root] = 1
                            subject_mask[left_s: left_e] = subj_mask
                        else:
                            subject_mask[left[0]] = 1

                        if len(right) > 1:
                            right_s = right[0]
                            right_e = right[-1] + 1

                            obj = sample["tokens"][right_s:right_e]
                            obj = " ".join(obj)
                            obj_doc = self._nlp(obj)
                            obj_deps = [t.dep_ for t in obj_doc]
                            obj_root = obj_deps.index("ROOT")
                            obj_mask = [0] * len(obj_deps)
                            obj_mask[obj_root] = 1
                            object_mask[right_s: right_e] = obj_mask
                        else:
                            object_mask[right[0]] = 1

                        subj_index = subject_mask.index(1)
                        obj_index = object_mask.index(1)
                        rel_type = item["kbID"]
                        if rel_type in experiment.WIKIDATA_RELATION_TYPES_MAP.all_values():
                            input_example = relation_input_example.RelationInputExample(
                                input_tokens=sample["tokens"],
                                head_span=[left_s, left_e],
                                head_span_root=[subj_index],
                                tail_span=[right_s, right_e],
                                tail_span_root=[obj_index],
                                relation_type=rel_type
                            )
                            samples.append(input_example)
                        else:
                            continue
                except Exception as e:
                    continue
        return samples



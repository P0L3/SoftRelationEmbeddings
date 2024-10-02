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
__date__ = "10 May 2023"
__author__ = "Frank M. Mtumbuka"
__maintainer__ = "Frank M. Mtumbuka"
__email__ = "" ""
__status__ = "Development"

import datapipeline.base_data_loader as base_data_loader
import datapipeline.relation_input_example as relation_input_example
import json
import os
import typing


class TACREDDataLoader(base_data_loader.BaseDataLoader):
    """This implements a loader that loads the TACRED dataset"""

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
        data = json.load(open(os.path.join(self._data_path, self._rel_data_file), "r"))

        for idx, sample in enumerate(data):
            tokens = sample["token"]
            subj_start = sample["subj_start"]
            subj_end = sample["subj_end"]
            obj_start = sample["obj_start"]
            obj_end = sample["obj_end"]

            # Specify the start and end of head and tail entity spans
            head_span = [subj_start, subj_end + 1]
            tail_span = [obj_start, obj_end + 1]

            # Find the roots of head and tail entity spans
            head_doc = self._nlp(" ".join(tokens[head_span[0]: head_span[1]]))
            tail_doc = self._nlp(" ".join(tokens[tail_span[0]: tail_span[1]]))
            head_deps = [t.dep_ for t in head_doc]
            head_root = head_deps.index("ROOT")
            tail_deps = [t.dep_ for t in tail_doc]
            tail_root = tail_deps.index("ROOT")

            relation_type = sample["relation"]
            try:
                head_span_root_index = subj_start + head_root
                tail_span_root_index = obj_start + tail_root
                head = tokens[head_span_root_index]
                tail = tokens[tail_span_root_index]
                input_example = relation_input_example.RelationInputExample(
                    input_tokens=tokens,
                    head_span=head_span,
                    head_span_root=[head_span_root_index],
                    tail_span=tail_span,
                    tail_span_root=[tail_span_root_index],
                    relation_type=relation_type
                )
                samples.append(input_example)
            except Exception as e:
                continue
        return samples


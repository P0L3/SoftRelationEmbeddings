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
__date__ = "29 Aug 2023"
__author__ = "Frank M. Mtumbuka"
__maintainer__ = "Frank M. Mtumbuka"
__email__ = "" ""
__status__ = "Development"

import datapipeline.base_data_loader as base_data_loader
import datapipeline.relation_input_example as relation_input_example
import json
import os
import typing
import experiment


class NYTDataLoader(base_data_loader.BaseDataLoader):
    """This implements a loader that loads the NYT dataset"""

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
        data = json.load(open(os.path.join(self._data_path, self._rel_data_file), "r"))["data"]

        for idx, sample in enumerate(data):
            try:
                # Retrieve the subject and object from the sample
                sub = " ".join(sample["sub"].split("_")).lower()
                obj = " ".join(sample["obj"].split("_")).lower()
                sent = sample["sent"].rstrip().lower()

                # Replace the sub and obj with special tokens
                sent = sent.replace(sub, "subjh")
                sent = sent.replace(obj, "objt")

                # Tokenize the sentence using Spacy
                sent_doc = self._nlp(sent)
                sent_tokens = [token.text for token in sent_doc]

                subjh_index = sent_tokens.index("subjh")
                objt_index = sent_tokens.index("objt")
                # Tokenize sub and obj
                sub_doc = self._nlp(sub)
                obj_doc = self._nlp(obj)

                sub_tokens = [token.text for token in sub_doc]
                sub_deps = [t.dep_ for t in sub_doc]
                sub_root = sub_deps.index("ROOT")
                obj_tokens = [token.text for token in obj_doc]
                obj_deps = [t.dep_ for t in obj_doc]
                obj_root = obj_deps.index("ROOT")

                sent_tokens_sub_obj = ""
                sub_head_token_position = 0
                obj_head_token_position = 0
                # replace subj in sent tokens with original tokens
                if objt_index > subjh_index:
                    subj_index = sent_tokens.index("subjh")
                    new_sent_tokens_sub = sent_tokens[:subj_index] + sub_tokens + sent_tokens[subj_index + 1:]
                    sub_head_token_position = subj_index + sub_root

                    obj_index = new_sent_tokens_sub.index("objt")
                    sent_tokens_sub_obj = new_sent_tokens_sub[:obj_index] + obj_tokens + new_sent_tokens_sub[obj_index + 1:]
                    obj_head_token_position = obj_index + obj_root
                else:
                    obj_index = sent_tokens.index("objt")
                    sent_tokens_obj = sent_tokens[:obj_index] + obj_tokens + sent_tokens[obj_index + 1:]
                    obj_head_token_position = obj_index + obj_root

                    subj_index = sent_tokens.index("subjh")
                    sent_tokens_sub_obj = sent_tokens_obj[:subj_index] + sub_tokens + sent_tokens_obj[subj_index + 1:]
                    sub_head_token_position = subj_index + sub_root

                rel = sample["rel"]
                if rel in experiment.NYT_RELATION_TYPES_MAP.all_values():
                    input_example = relation_input_example.RelationInputExample(
                        input_tokens=sent_tokens_sub_obj,
                        head_span=[sub_head_token_position, sub_head_token_position],
                        head_span_root=[sub_head_token_position],
                        tail_span=[obj_head_token_position, obj_head_token_position],
                        tail_span_root=[obj_head_token_position],
                        relation_type=rel
                    )
                    samples.append(input_example)
                else:
                    continue
            except Exception as e:
                continue
        return samples

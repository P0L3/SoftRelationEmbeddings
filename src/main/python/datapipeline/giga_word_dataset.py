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
__date__ = "20 Mar 2023"
__author__ = "Frank M. Mtumbuka"
__maintainer__ = "Frank M. Mtumbuka"
__email__ = "" ""
__status__ = "Development"

import datapipeline.base_data_loader as base_data_loader
import torch.utils.data as data
import typing
from transformers import AutoTokenizer


class GigaWordDataset(data.Dataset):

    def __init__(self, data_loader: base_data_loader.BaseDataLoader, tokenizer: AutoTokenizer, sentences: bool = False):
        """

        Args:
            data_loader:
        """
        super().__init__()
        # Sanitize args.

        # Store args.
        self._tokenizer = tokenizer

        self._data = data_loader.load()
        print("Pre-processing data...")
        if sentences:
            sents = []
            for story in self._data:
                for cluster in story.clusters:
                    sents += cluster.sentences
            self._data = sents

            for sample_idx, item in enumerate(self._data):
                new_text_arr = []
                # Wrap the mentions between [MASK] and [PAD] tokens.
                for token_idx, token in enumerate(item.text_arr):
                    if token_idx in item.mention_positions:
                        new_text_arr.append(self._tokenizer.mask_token)
                        new_text_arr.append(token)
                        new_text_arr.append(self._tokenizer.pad_token)
                    else:
                        new_text_arr.append(token)

                # Tokenize tokens
                new_text = " ".join(new_text_arr)
                tokenized_text = self._tokenizer(new_text)["input_ids"]

                # Find positions of entity mentions
                entity_start = [
                    index for index, token in enumerate(tokenized_text) if token == self._tokenizer.mask_token_id
                ]
                entity_end = [
                    index for index, token in enumerate(tokenized_text) if token == self._tokenizer.pad_token_id
                ]

                entity_mask = [0] * len(tokenized_text)
                for s, e in zip(entity_start, entity_end):
                    for token_idx, token in enumerate(tokenized_text):
                        if s < token_idx < e:
                            entity_mask[token_idx] = 1

                # Remove [MASK] and [PAD] tokens around mentions.
                new_tokenized_text = []
                new_entity_mask = []

                for token_idx, (token, mask) in enumerate(zip(tokenized_text, entity_mask)):
                    if token_idx in entity_start or token_idx in entity_end:
                        pass
                    else:
                        new_entity_mask.append(mask)
                        new_tokenized_text.append(token)

                # Set tokenized text idx and mask to sentence
                self._data[sample_idx].tokenized_text_mask = new_entity_mask
                self._data[sample_idx].tokenizer_text_idx = new_tokenized_text
        print("OK")
        print()

    def __getitem__(self, index) -> typing.Tuple[typing.List, typing.List]:
        """
        This return the input sequence with a corresponding entity mask.
        Args:
            index (int): The index for the sequence and entity mask to be retrieved.

        Returns:
            input_seq (list): A list of token ids for the tokens in the input sentence.
            input_mask (list): An entity mask with `1` indicating the position of an entity, and `0` for non-entities.
        """
        return self._data[index].tokenizer_text_idx, self._data[index].tokenized_text_mask

    def __len__(self) -> int:
        """int: Specifies the size of the dataset."""
        return len(self._data)

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Specifies the tokenizer used."""
        return self._tokenizer

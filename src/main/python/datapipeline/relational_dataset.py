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
__date__ = "09 Apr 2023"
__author__ = "Frank M. Mtumbuka"
__maintainer__ = "Frank M. Mtumbuka"
__email__ = "" ""
__status__ = "Development"

import datapipeline.base_data_loader as base_data_loader
import datapipeline.relation_input_example as relation_input_example
import experiment
import torch.utils.data as data
import typing
import utilpackage.index_map as index_map
from transformers import AutoTokenizer


class RelationalDataset(data.Dataset):
    """This class processes and represents relational data."""

    RELATION_PROMPT = ["The", "relation", "is"]
    """list: The list of tokens at the beginning of the relation prompt to be followed by [MASK]?"""

    def __init__(
            self,
            data_loader: base_data_loader.BaseDataLoader,
            relation_types: index_map.IndexMap,
            tokenizer: AutoTokenizer,
            relation_prompt: bool = False
    ):
        """
        This creates an instance of `RelationalDataset`.
        Args:
            data_loader (::class:`base_data_loader.BaseDataLoader`): The specified data loader.
            relation_types (::class:`index_map.IndexMap`): A class that maps relation types to the corresponding ids.
            relation_prompt (bool): Specifies whether to use the prompt with a MASK for the relation vector or not.
            tokenizer (::class:`AutoTokenizer`): The tokenizer for the model being used.
        """
        super().__init__()
        # Sanitize args.

        # Store args.
        self._tokenizer = tokenizer
        self._relation_types = relation_types
        self._relation_prompt = relation_prompt
        self._cls_token = tokenizer.cls_token
        self._sep_token = tokenizer.sep_token
        self._mask_token = tokenizer.mask_token

        self._data = data_loader.load()
        print("Pre-processing data...")
        if self._relation_prompt:
            # Create prompt
            # rel_prompt = [SEP] The relation is [MASK] ? [SEP]
            rel_prompt = [self._sep_token] + self.RELATION_PROMPT + [self._mask_token, "?", self._sep_token]
            rel_prompt_mask = [0] * len(rel_prompt)
            rel_mask_index = rel_prompt.index(self._mask_token)
            # rel_prompt_mask = [0, 0, 0, 0, 1, 0, 0]
            rel_prompt_mask[rel_mask_index] = 1

            # Loop through samples
            for sample_idx, sample in enumerate(self._data):
                # Pre append [CLS] token to input tokens
                input_tokens = sample.input_tokens

                # Create entity mask and mark positions of roots for the head and tail entity spans
                head_entity_mask = [0] * len(input_tokens)
                tail_entity_mask = [0] * len(input_tokens)
                head_entity_mask[sample.head_span_root[0]] = 1
                tail_entity_mask[sample.tail_span_root[0]] = 1

                # Add CLS token and corresponding masks

                # Create relation mask to mark the position of the mask representing the relation vector
                current_relation_mask = [0] * len(input_tokens) + rel_prompt_mask

                # Add input tokens to relation prompt
                input_with_prompt = input_tokens + rel_prompt

                # Create entity mask
                current_head_entity_mask = head_entity_mask + [0] * len(rel_prompt)
                current_tail_entity_mask = tail_entity_mask + [0] * len(rel_prompt)

                # Convert tokens to idx.
                input_idx = []
                new_tokens = []
                new_head_entity_mask = []
                new_tail_entity_mask = []
                new_relation_mask = []
                for token, h_mask, t_mask, r_mask in zip(
                        input_with_prompt,
                        current_head_entity_mask,
                        current_tail_entity_mask,
                        current_relation_mask
                ):
                    # compute the number of pieces that the word was split into
                    word_pieces = self._tokenizer.tokenize(token)
                    token_idx = self._tokenizer.convert_tokens_to_ids(word_pieces)
                    # check if the word was split into pieces
                    if len(word_pieces) > 1:  # -> the word was split
                        new_tokens.extend(word_pieces)
                        input_idx.extend(token_idx)
                        h_mask_tensor = None
                        t_maks_tensor = None
                        if h_mask == 1:
                            h_mask_tensor = [0] * len(token_idx)
                            h_mask_tensor[0] = 1
                        else:
                            h_mask_tensor = [h_mask] * len(token_idx)

                        if t_mask == 1:
                            t_mask_tensor = [0] * len(token_idx)
                            t_mask_tensor[0] = 1
                        else:
                            t_mask_tensor = [t_mask] * len(token_idx)
                        new_tail_entity_mask.extend(t_mask_tensor)
                        new_head_entity_mask.extend(h_mask_tensor)
                        new_relation_mask.extend([r_mask] * len(token_idx))
                    else:
                        new_tokens.extend(word_pieces)
                        input_idx.extend(token_idx)
                        new_head_entity_mask.append(h_mask)
                        new_tail_entity_mask.append(t_mask)
                        new_relation_mask.append(r_mask)

                # Update sample
                self._data[sample_idx].input_tokens_prompt = new_tokens
                self._data[sample_idx].input_tokens_prompt_idx = input_idx
                self._data[sample_idx].head_entity_mask = new_head_entity_mask
                self._data[sample_idx].tail_entity_mask = new_tail_entity_mask
                self._data[sample_idx].relation_mask = new_relation_mask
                self._data[sample_idx].relation_type_idx = self._relation_types.index(sample.relation_type)
        else:
            print("To be implemented")
        print("OK")

    @property
    def relation_types(self) -> index_map.IndexMap:
        return self._relation_types

    def __getitem__(self, index) -> typing.List[relation_input_example.RelationInputExample]:
        """
        This returns the input example at the specified index.
        Args:
            index (int): The index for the input example to be retrieved.

        Returns:
            list[::class:`relation_input_example.RelationInputExample`]: The retrieved relation input example
        """
        return [self._data[index]]

    def __len__(self) -> int:
        """int: Specifies the size of the dataset."""
        return len(self._data)

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Specifies the tokenizer used."""
        return self._tokenizer

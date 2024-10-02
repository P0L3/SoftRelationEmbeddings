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
import experiment
import random
import torch.utils.data as data
import typing
import utilpackage.index_map as index_map
from transformers import AutoTokenizer


class ACEDataset(data.Dataset):

    def __init__(
            self,
            data_loader: base_data_loader.BaseDataLoader,
            tokenizer: AutoTokenizer,
            rel_types: index_map.IndexMap,
            rel_mask: bool = False
    ):
        """
        This creates an instance of `ACEDataset`.
        Args:
            data_loader (::class:`base_data_loader.BaseDataLoader`): The specified data loader.
            tokenizer (::class:`AutoTokinizer`): The pre-trained tokenizer being used.
            rel_types (::class:`index_map.IndexMap`): A class that maps relation types to the corresponding ids.
            rel_mask (bool): Specifies whether to get the relation vector for the relation expressed in the input
                sequence using the `[MASK]` token in the prompt.
        """
        super().__init__()
        # Sanitize args.

        # Store args.
        self._tokenizer = tokenizer
        self._rel_types = rel_types
        self._rel_mask = rel_mask
        self._data = []
        self._cls_token = self._tokenizer.cls_token
        self._cls_token_id = self._tokenizer.cls_token_id
        self._mask_token = self._tokenizer.mask_token
        self._mask_token_id = self._tokenizer.mask_token_id
        self._pad_token = self._tokenizer.pad_token
        self._pad_token_id = self._tokenizer.pad_token_id
        self._sep_token = self._tokenizer.sep_token
        self._sep_token_id = self._tokenizer.sep_token_id
        loaded_data = data_loader.load()
        print("Pre-processing data...")
        for sample_idx, item in enumerate(loaded_data):
            input_tokens = item.input_text_tokens
            arg_1 = item.relation_arg_1
            arg_2 = item.relation_arg_2
            relation_entity_role_mask = item.relation_entity_role_mask

            # Re-label the relation roles to using the BIO system
            new_input_tokens = []
            new_input_tokens_idx = []
            new_arg_one = []
            new_arg_two = []
            new_relation_roles = []

            for idx, (token, a_one, a_two, role) in enumerate(
                zip(input_tokens, arg_1, arg_2, relation_entity_role_mask)
            ):
                # compute the number of pieces that the word was split into
                new_tokens = self._tokenizer.tokenize(token)
                new_token_ids = self._tokenizer.convert_tokens_to_ids(new_tokens)
                num_pieces = len(new_tokens)

                # check if the word was split into pieces
                if num_pieces > 1:  # -> the word was split

                    if role == experiment.REL_ARG_ONE_BEGIN:  # -> beginning of arg 1
                        new_relation_roles.extend([experiment.REL_ARG_ONE_INTERMEDIATE] * num_pieces)

                    elif role == experiment.REL_ARG_ONE_INTERMEDIATE:  # ->
                        new_relation_roles.extend([experiment.REL_ARG_ONE_INTERMEDIATE] * num_pieces)

                    elif role == experiment.REL_ARG_TWO_BEGIN:  # -> beginning of arg 2
                        new_relation_roles.extend([experiment.REL_ARG_TWO_INTERMEDIATE] * num_pieces)

                    elif role == experiment.REL_ARG_TWO_INTERMEDIATE:
                        new_relation_roles.extend([experiment.REL_ARG_TWO_INTERMEDIATE] * num_pieces)
                    else:  # -> not a beginning -> we can simply replicate the label
                        new_relation_roles.extend([role] * num_pieces)

                    # Extend the other sequences with the according number of tokens.
                    new_input_tokens.extend(new_tokens)
                    new_input_tokens_idx.extend(new_token_ids)
                    new_arg_one.extend([a_one] * num_pieces)
                    new_arg_two.extend([a_one] * num_pieces)

                else:
                    """
                    If the token is not split into multiple tokens, just append the current token, and the according
                    tokens to respective lists.
                    """
                    new_input_tokens.append(token)
                    new_input_tokens_idx.extend(new_token_ids)
                    new_arg_one.append(a_one)
                    new_arg_two.append(a_two)
                    new_relation_roles.append(role)

            """
            Correct malformed label sequences to make sure I-tags follow B-tags. 
            e.g ["I-Arg1","I-Arg1"] => ["B-Arg1", "I-Arg1"]
            """
            new_relation_roles = self.re_label_seq(new_relation_roles)

            # check if length of args
            arg_1_len = sum(1 for i in new_arg_one if i != "O")
            arg_2_len = sum(1 for i in new_arg_two if i != "O")

            if arg_1_len == 0 or arg_2_len == 0:
                continue
            else:
                # Generate the head and tail masks
                input_head, head_mask = self.generate_mask(new_input_tokens_idx, new_arg_one)
                _, tail_mask = self.generate_mask(new_input_tokens_idx, new_arg_two)

                # Pre-append and post-append according special tokens to the sequences
                loaded_data[sample_idx].input_token_idx = [self._cls_token_id] + input_head + [self._sep_token_id]
                loaded_data[sample_idx].relation_arg_1_mask = [0] + head_mask + [0]
                loaded_data[sample_idx].relation_arg_2_mask = [0] + tail_mask + [0]
                loaded_data[sample_idx].relation_type_id = self._rel_types.index(item.relation_type)
                loaded_data[sample_idx].relation_entity_role_mask = ["O"] + new_relation_roles + ["O"]

                # If relation mask is specified, generate the prompt.
                if self._rel_mask:
                    prompt_idx = self._tokenizer.convert_tokens_to_ids(["The", "relation", "is", self._mask_token, "?"])
                    rel_prompt = loaded_data[sample_idx].input_token_idx + prompt_idx + [self._sep_token_id]

                    # Create the mask with `0`s at the end indicating tokens that belong to the prompt.
                    prompt_mask = [1] * len(loaded_data[sample_idx].input_token_idx) + [0] * (len(prompt_idx) + 1)
                    rel_prompt_idx, rel_prompt_mask = self.generate_mask(rel_prompt, rel_mask=True)

                    # Update the input example
                    loaded_data[sample_idx].rel_prompt_idx = rel_prompt_idx
                    loaded_data[sample_idx].rel_prompt_mask = rel_prompt_mask
                    loaded_data[sample_idx].prompt_mask = prompt_mask
                self._data.append(loaded_data[sample_idx])

        print("OK")

    @property
    def rel_types(self) -> index_map.IndexMap:
        return self._rel_types

    def __getitem__(self, index) -> typing.Tuple[
        typing.List, typing.List, typing.List, typing.List, typing.List, typing.List, typing.List
    ]:
        """
        This return the input sequence with a corresponding entity mask.
        Args:
            index (int): The index for the sequence and entity mask to be retrieved.

        Returns:
            input_seq (list): A list of token ids for the tokens in the input sentence.
            input_mask (list): An entity mask with `1` indicating the position of an entity, and `0` for non-entities.
        """
        input_idx = self._data[index].input_token_idx
        head_entity_mask = self._data[index].relation_arg_1_mask
        tail_entity_mask = self._data[index].relation_arg_2_mask
        relation_type_id = [self._data[index].relation_type_id]
        rel_prompt_idx = self._data[index].rel_prompt_idx if self._data[index].rel_prompt_idx is not None else [
            self._data[index].rel_prompt_idx]
        rel_prompt_mask = self._data[index].rel_prompt_mask if self._data[index].rel_prompt_mask is not None else [
            self._data[index].rel_prompt_mask]
        if self._data[index].rel_prompt_idx is not None:
            head_entity_mask += [0] * (len(rel_prompt_idx) - len(head_entity_mask))
            tail_entity_mask += [0] * (len(rel_prompt_idx) - len(tail_entity_mask))

        prompt_mask = self._data[index].prompt_mask if self._data[index].prompt_mask is not None else [
            self._data[index].prompt_mask]

        return (
            input_idx,
            head_entity_mask,
            tail_entity_mask,
            relation_type_id,
            rel_prompt_idx,
            rel_prompt_mask,
            prompt_mask
        )

    def __len__(self) -> int:
        """int: Specifies the size of the dataset."""
        return len(self._data)

    def generate_mask(
            self,
            input_tokens: list,
            input_mask: list = None,
            rel_mask: bool = False
    ) -> typing.Tuple[typing.List, typing.List]:
        """
        This generates a mask indicating the positions for specific tokens.
        Args:
            input_tokens (list): The input tokens from which to mark special tokens.
            input_mask (list): The mask indicating the positions of tokens of interest in the input tokens.
            rel_mask (bool): Whether to relation prompt is used or not.
        Returns:
            input_sequence (list):  The input tokens from which to mark special tokens.
            mask (list): The mask sequence with `1`s indicating the position of special tokens, `0`s otherwise.
        """
        new_tokens = []
        processed_input_mask = []
        processed_input_tokens = []
        if rel_mask:
            rel_mask_position = input_tokens.index(self._mask_token_id)
            processed_input_mask = [0] * len(input_tokens)
            processed_input_mask[rel_mask_position] = 1
            processed_input_tokens = input_tokens
        if input_mask is None:
            # processed_input_tokens = self._tokenizer(" ".join(input_tokens))["input_ids"]

            processed_input_mask = [0] * len(input_tokens)
            processed_input_mask[0] = 1
            processed_input_tokens = input_tokens
        else:
            # Find the length of the argument in terms of the number of tokens.
            length_arg_span = sum(1 for i in input_mask if i != 'O')
            span_count = 0
            for token, mask in zip(input_tokens, input_mask):
                if mask != "O":
                    if span_count == 0:
                        new_tokens.append(self._mask_token_id)
                        new_tokens.append(token)
                    elif 0 < span_count < length_arg_span:
                        new_tokens.append(token)

                    if length_arg_span == 1 or span_count == (length_arg_span - 1):
                        new_tokens.append(self._pad_token_id)
                    span_count += 1
                else:
                    new_tokens.append(token)

            # Tokenize tokens
            # new_text_tokens = " ".join(new_tokens)
            # tokenized_text = self._tokenizer(new_text_tokens)["input_ids"]
            tokenized_text = new_tokens

            # Find positions of entity mentions
            entity_start = tokenized_text.index(self._mask_token_id)
            entity_end = tokenized_text.index(self._pad_token_id)
            entity_mask = [0] * len(tokenized_text)
            if length_arg_span > 1:
                random_index = random.randint((entity_start + 1), (entity_end - 1))
                entity_mask[random_index] = 1
            else:
                for i in range(entity_start, entity_end):
                    entity_mask[i] = 1

            for token, mask in zip(tokenized_text, entity_mask):
                if token == self._mask_token_id or token == self._pad_token_id:
                    pass
                else:
                    processed_input_mask.append(mask)
                    processed_input_tokens.append(token)

            # Check the number of ones
            num_ones = sum(1 for i in processed_input_mask if i == 1)
            counter = 0
            if num_ones > 0:
                for idx, mask in enumerate(processed_input_mask):
                    if mask == 1 and counter == 0:
                        counter += 1
                    elif mask == 1 and counter > 0:
                        processed_input_mask[idx] = 0
                        counter += 1
                    else:
                        pass

        return processed_input_tokens, processed_input_mask

    def re_label_seq(self, relation_entity_role_mask):
        """
        Correct malformed label sequences to make sure I-tags follow B-tags.
        e.g ["I-Arg1","I-Arg1"] => ["B-Arg1", "I-Arg1"]
        Args:
            relation_entity_role_mask:

        Returns:

        """
        new_relation_entity_role_mask = [experiment.OTHER_TAG] * len(relation_entity_role_mask)
        for idx, tag in enumerate(relation_entity_role_mask):
            if tag == "Arg-1" and (idx - 1) >= 0 and relation_entity_role_mask[idx - 1] != "Arg-1":
                # Assign beginning of argument one tag if the current tag is ARG1, it's appearing for the 1st time.
                new_relation_entity_role_mask[idx] = experiment.REL_ARG_ONE_BEGIN

            elif tag == "Arg-1" and (idx - 1) >= 0 and relation_entity_role_mask[idx - 1] == "Arg-1":
                # Assign arg one intermediate tag if the current tag is ARG1, it isn't appearing for the 1st time.
                new_relation_entity_role_mask[idx] = experiment.REL_ARG_ONE_INTERMEDIATE

            elif tag == "Arg-2" and (idx - 1) >= 0 and relation_entity_role_mask[idx - 1] != "Arg-2":
                # Assign beginning of argument two tag if the current tag is ARG1, it's appearing for the 1st time.
                new_relation_entity_role_mask[idx] = experiment.REL_ARG_TWO_BEGIN

            elif tag == "Arg-2" and (idx - 1) >= 0 and relation_entity_role_mask[idx - 1] == "Arg-2":
                # Assign arg two intermediate tag if the current tag is ARG1, it isn't appearing for the 1st time.
                new_relation_entity_role_mask[idx] = experiment.REL_ARG_TWO_INTERMEDIATE

            elif tag == "Arg-1" and idx == 0:
                # Assign arg one tag if the current tag is ARG1, is at the beginning of the sequence.
                new_relation_entity_role_mask[idx] = experiment.REL_ARG_ONE_BEGIN

            elif tag == "Arg-2" and idx == 0:
                # Assign arg two tag if the current tag is ARG1, is at the beginning of the sequence.
                new_relation_entity_role_mask[idx] = experiment.REL_ARG_TWO_BEGIN
            else:
                pass

        return new_relation_entity_role_mask

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Specifies the tokenizer used."""
        return self._tokenizer

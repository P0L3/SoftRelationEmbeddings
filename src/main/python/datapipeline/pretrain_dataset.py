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
__date__ = "08 Aug 2023"
__author__ = "Frank M. Mtumbuka"
__maintainer__ = "Frank M. Mtumbuka"
__email__ = "" ""
__status__ = "Development"

import datapipeline.base_data_loader as base_data_loader
import datapipeline.pretraining_pairs as pretraining_pairs
import insanity
import spacy
import torch.utils.data as data
import typing
from transformers import AutoTokenizer


class PretrainDataset(data.Dataset):
    """This creates the pretraining dataset."""

    def __init__(self, data_loader: base_data_loader.BaseDataLoader, tokenizer: AutoTokenizer):
        """
        This creates an instance of `PretrainDataset`.
        Args:
            data_loader (::class:`base_data_loader.BaseDataLoader`): The specified data loader.
            tokenizer (::class:`AutoTokenizer`): The specified tokenizer.
        """

        # Super class call
        super().__init__()

        # Sanitize args
        insanity.sanitize_type("data_loader", data_loader, base_data_loader.BaseDataLoader)

        # Create dataset
        self._data = []
        self._nlp = spacy.load('en_core_web_sm')
        self._tokenizer = tokenizer
        loaded_data = data_loader.load()

        # Pre-process the data
        print("Pre-processing data")
        for sample_idx, sample in enumerate(loaded_data):
            try:
                # Replace the entity spans with ent_a, and ent_b
                first_sent = sample.first_sent
                second_sent = sample.second_sent

                # The start and end of entities A and B in both the first and second sentences.
                first_sent_ent_a_start = sample.first_sent_ent_a[0]
                first_sent_ent_a_end = sample.first_sent_ent_a[1]
                first_sent_ent_b_start = sample.first_sent_ent_b[0]
                first_sent_ent_b_end = sample.first_sent_ent_b[1]
                second_sent_ent_a_start = sample.second_sent_ent_a[0]
                second_sent_ent_a_end = sample.second_sent_ent_a[1]
                second_sent_ent_b_start = sample.second_sent_ent_b[0]
                second_sent_ent_b_end = sample.second_sent_ent_b[1]

                # Replace ent_a and ent_b spans in the first sentence
                first_sent_entity_a = first_sent[first_sent_ent_a_start:first_sent_ent_a_end]
                first_sent_entity_b = first_sent[first_sent_ent_b_start:first_sent_ent_b_end]

                first_sent = first_sent.replace(first_sent_entity_a, "ent_a")
                first_sent = first_sent.replace(first_sent_entity_b, "ent_b")

                # Replace ent_a and ent_b spans in the second sentence
                second_sent_entity_a = second_sent[second_sent_ent_a_start:second_sent_ent_a_end]
                second_sent_entity_b = second_sent[second_sent_ent_b_start:second_sent_ent_b_end]

                second_sent = second_sent.replace(second_sent_entity_a, "ent_a")
                second_sent = second_sent.replace(second_sent_entity_b, "ent_b")

                # Tokenize the text before using spacy
                first_sent_doc = self._nlp(first_sent)
                second_sent_doc = self._nlp(second_sent)
                first_sent_tokens = [token.text for token in first_sent_doc]
                second_sent_tokens = [token.text for token in second_sent_doc]

                # Tokenize entity spans
                first_sent_ent_a_doc = self._nlp(first_sent_entity_a)
                first_sent_ent_b_doc = self._nlp(first_sent_entity_b)
                second_sent_ent_a_doc = self._nlp(second_sent_entity_a)
                second_sent_ent_b_doc = self._nlp(second_sent_entity_b)

                first_sent_ent_a_tokens = [token.text for token in first_sent_ent_a_doc]
                first_sent_ent_b_tokens = [token.text for token in first_sent_ent_b_doc]
                second_sent_ent_a_tokens = [token.text for token in second_sent_ent_a_doc]
                second_sent_ent_b_tokens = [token.text for token in second_sent_ent_b_doc]

                # Locate the heads of the entity spans
                first_sent_ent_a_head = [1 if token.dep_ == "ROOT" else 0 for token in first_sent_ent_a_doc]
                first_sent_ent_b_head = [1 if token.dep_ == "ROOT" else 0 for token in first_sent_ent_b_doc]
                second_sent_ent_a_head = [1 if token.dep_ == "ROOT" else 0 for token in second_sent_ent_a_doc]
                second_sent_ent_b_head = [1 if token.dep_ == "ROOT" else 0 for token in second_sent_ent_b_doc]

                # Update sentences and positions
                first_sent_ent_a_mask = [0] * len(first_sent_tokens)
                first_sent_ent_b_mask = [0] * len(first_sent_tokens)
                first_sent_ent_a_index = first_sent_tokens.index("ent_a")
                # Update masks
                first_sent_ent_a_mask = first_sent_ent_a_mask[:first_sent_ent_a_index] + \
                                        first_sent_ent_a_head + \
                                        first_sent_ent_a_mask[first_sent_ent_a_index + 1:]
                # Update ent_b mask
                first_sent_ent_b_mask = first_sent_ent_b_mask[:first_sent_ent_a_index] + \
                                        [0] * len(first_sent_ent_a_head) + \
                                        first_sent_ent_b_mask[first_sent_ent_a_index + 1:]
                first_sent_tokens = first_sent_tokens[:first_sent_ent_a_index] + \
                                    first_sent_ent_a_tokens + \
                                    first_sent_tokens[first_sent_ent_a_index + 1:]

                first_sent_ent_b_index = first_sent_tokens.index("ent_b")
                first_sent_ent_b_mask = first_sent_ent_b_mask[:first_sent_ent_b_index] + \
                                        first_sent_ent_b_head + \
                                        first_sent_ent_b_mask[first_sent_ent_b_index + 1:]
                # Update ent_a mask
                first_sent_ent_a_mask = first_sent_ent_a_mask[:first_sent_ent_b_index] + \
                                        [0] * len(first_sent_ent_b_head) + \
                                        first_sent_ent_a_mask[first_sent_ent_b_index + 1:]
                first_sent_tokens = first_sent_tokens[:first_sent_ent_b_index] + \
                                    first_sent_ent_b_tokens + \
                                    first_sent_tokens[first_sent_ent_b_index + 1:]

                # second_sent_ent_mask = [0] * len(second_sent_tokens)
                # second_sent_ent_a_index = second_sent_tokens.index("ent_a")
                # # Update masks
                # second_sent_ent_mask = second_sent_ent_mask[:second_sent_ent_a_index] + \
                #                        second_sent_ent_a_head + \
                #                        second_sent_ent_mask[second_sent_ent_a_index + 1:]
                # second_sent_tokens = second_sent_tokens[:second_sent_ent_a_index] + \
                #                      second_sent_ent_a_tokens + \
                #                      second_sent_tokens[second_sent_ent_a_index + 1:]
                #
                # second_sent_ent_b_index = second_sent_tokens.index("ent_b")
                # second_sent_ent_mask = second_sent_ent_mask[:second_sent_ent_b_index] + \
                #                        second_sent_ent_b_head + \
                #                        second_sent_ent_mask[second_sent_ent_b_index + 1:]
                # second_sent_tokens = second_sent_tokens[:second_sent_ent_b_index] + \
                #                      second_sent_ent_b_tokens + \
                #                      second_sent_tokens[second_sent_ent_b_index + 1:]

                second_sent_ent_a_mask = [0] * len(second_sent_tokens)
                second_sent_ent_b_mask = [0] * len(second_sent_tokens)
                second_sent_ent_a_index = second_sent_tokens.index("ent_a")
                # Update masks
                second_sent_ent_a_mask = second_sent_ent_a_mask[:second_sent_ent_a_index] + \
                                         second_sent_ent_a_head + \
                                         second_sent_ent_a_mask[second_sent_ent_a_index + 1:]
                # Update ent_b mask
                second_sent_ent_b_mask = second_sent_ent_b_mask[:second_sent_ent_a_index] + \
                                         [0] * len(second_sent_ent_a_head) + \
                                         second_sent_ent_b_mask[second_sent_ent_a_index + 1:]
                second_sent_tokens = second_sent_tokens[:second_sent_ent_a_index] + \
                                     second_sent_ent_a_tokens + \
                                     second_sent_tokens[second_sent_ent_a_index + 1:]

                second_sent_ent_b_index = second_sent_tokens.index("ent_b")
                second_sent_ent_b_mask = second_sent_ent_b_mask[:second_sent_ent_b_index] + \
                                         second_sent_ent_b_head + \
                                         second_sent_ent_b_mask[second_sent_ent_b_index + 1:]
                # Update ent_a mask
                second_sent_ent_a_mask = second_sent_ent_a_mask[:second_sent_ent_b_index] + \
                                         [0] * len(second_sent_ent_b_head) + \
                                         second_sent_ent_a_mask[second_sent_ent_b_index + 1:]
                second_sent_tokens = second_sent_tokens[:second_sent_ent_b_index] + \
                                     second_sent_ent_b_tokens + \
                                     second_sent_tokens[second_sent_ent_b_index + 1:]

                # Update token sequences using the supplied tokenizer
                new_first_sent_token_idx = []
                new_first_sent_ent_a_mask = []
                new_first_sent_ent_b_mask = []
                new_second_sent_token_idx = []
                new_second_sent_ent_a_mask = []
                new_second_sent_ent_b_mask = []
                for token, mask_a, mask_b in zip(first_sent_tokens, first_sent_ent_a_mask, first_sent_ent_a_mask):
                    # compute the number of pieces that the word was split into
                    new_tokens = self._tokenizer.tokenize(token)
                    new_token_ids = self._tokenizer.convert_tokens_to_ids(new_tokens)
                    num_pieces = len(new_token_ids)

                    # check if the word was split into pieces
                    if num_pieces > 1:  # -> the word was split
                        new_mask_a = [0] * num_pieces
                        new_mask_b = [0] * num_pieces
                        if mask_a == 1:
                            new_mask_a[0] = 1

                        if mask_b == 1:
                            new_mask_b[0] = 1
                        new_first_sent_token_idx.extend(new_token_ids)
                        new_first_sent_ent_a_mask.extend(new_mask_a)
                        new_first_sent_ent_b_mask.extend(new_mask_b)
                    else:
                        new_first_sent_token_idx.extend(new_token_ids)
                        new_first_sent_ent_a_mask.append(mask_a)
                        new_first_sent_ent_b_mask.append(mask_b)

                # for token, mask in zip(second_sent_tokens, second_sent_ent_mask):
                #     # compute the number of pieces that the word was split into
                #     new_tokens = self._tokenizer.tokenize(token)
                #     new_token_ids = self._tokenizer.convert_tokens_to_ids(new_tokens)
                #     num_pieces = len(new_token_ids)
                #
                #     # check if the word was split into pieces
                #     if num_pieces > 1:  # -> the word was split
                #         new_mask = [0] * num_pieces
                #         if mask == 1:
                #             new_mask[0] = 1
                #         new_second_sent_token_idx.extend(new_token_ids)
                #         new_second_sent_ent_mask.extend(new_mask)
                #     else:
                #         new_second_sent_token_idx.extend(new_token_ids)
                #         new_second_sent_ent_mask.append(mask)

                for token, mask_a, mask_b in zip(second_sent_tokens, second_sent_ent_a_mask, second_sent_ent_a_mask):
                    # compute the number of pieces that the word was split into
                    new_tokens = self._tokenizer.tokenize(token)
                    new_token_ids = self._tokenizer.convert_tokens_to_ids(new_tokens)
                    num_pieces = len(new_token_ids)

                    # check if the word was split into pieces
                    if num_pieces > 1:  # -> the word was split
                        new_mask_a = [0] * num_pieces
                        new_mask_b = [0] * num_pieces
                        if mask_a == 1:
                            new_mask_a[0] = 1

                        if mask_b == 1:
                            new_mask_b[0] = 1
                        new_second_sent_token_idx.extend(new_token_ids)
                        new_second_sent_ent_a_mask.extend(new_mask_a)
                        new_second_sent_ent_b_mask.extend(new_mask_b)
                    else:
                        new_second_sent_token_idx.extend(new_token_ids)
                        new_second_sent_ent_a_mask.append(mask_a)
                        new_second_sent_ent_b_mask.append(mask_b)

                # Add the relation prompt mask
                prompt = "The relation is {}?".format(self._tokenizer.mask_token)
                prompt_tokens = prompt.split(" ")
                prompt_token_idx = []
                for token in prompt_tokens:
                    new_tokens = self._tokenizer.tokenize(token)
                    new_token_ids = self._tokenizer.convert_tokens_to_ids(new_tokens)
                    prompt_token_idx.extend(new_token_ids)

                mask_token_index = prompt_token_idx.index(self._tokenizer.mask_token_id)
                prompt_mask = [0] * len(prompt_token_idx)
                prompt_mask[mask_token_index] = 1

                # Update input tokens
                loaded_data[sample_idx].first_sent_tokens = [self._tokenizer.cls_token_id] + \
                                                           new_first_sent_token_idx + \
                                                           [self._tokenizer.sep_token_id] + \
                                                           prompt_token_idx

                loaded_data[sample_idx].first_sent_ent_a_mask = [0] + \
                                                             new_first_sent_ent_a_mask + \
                                                             [0] + \
                                                             [0] * len(prompt_token_idx)
                loaded_data[sample_idx].first_sent_ent_b_mask = [0] + \
                                                               new_first_sent_ent_b_mask + \
                                                               [0] + \
                                                               [0] * len(prompt_token_idx)

                loaded_data[sample_idx].first_sent_prompt_mask = [0] + [0] * len(new_first_sent_ent_a_mask) + [0] + prompt_mask

                loaded_data[sample_idx].second_sent_tokens = [self._tokenizer.cls_token_id] + \
                                                            new_second_sent_token_idx + \
                                                            [self._tokenizer.sep_token_id] + \
                                                            prompt_token_idx

                loaded_data[sample_idx].second_sent_ent_a_mask = [0] + \
                                                              new_second_sent_ent_a_mask + \
                                                              [0] + \
                                                              [0] * len(prompt_token_idx)
                loaded_data[sample_idx].second_sent_ent_b_mask = [0] + \
                                                                new_second_sent_ent_b_mask + \
                                                                [0] + \
                                                                [0] * len(prompt_token_idx)

                loaded_data[sample_idx].second_sent_prompt_mask = [0] + \
                                                                 [0] * len(new_second_sent_ent_a_mask) + \
                                                                 [0] + \
                                                                 prompt_mask

                self._data.append(loaded_data[sample_idx])
                if (sample_idx + 1) % 1000 == 0:
                    print("Processed sample: {}/{}...".format(sample_idx+1, len(loaded_data)))

            except Exception as e:
                continue

        print("Ok")
        print()

    def __len__(self) -> int:
        """int: Specifies the size of the dataset."""
        return len(self._data)

    def __getitem__(self, index: int) -> typing.List[pretraining_pairs.SentencePairs]:
        """
        Retrieves the pretraining pair at a given index
        Args:
            index (int): The index for the pair of sentences to be retrieved.

        Returns:
            pair (list[::class:`pretraining_pairs.SentencePairs`]): The pair at the given index.
        """
        return [self._data[index]]

    @property
    def tokenizer(self) -> AutoTokenizer:
        """::class:`AutoTokenizer`: Specifies the tokenizer being used."""
        return self._tokenizer

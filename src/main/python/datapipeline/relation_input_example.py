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
__date__ = "02 Apr 2023"
__author__ = "Frank M. Mtumbuka"
__maintainer__ = "Frank M. Mtumbuka"
__email__ = "" ""
__status__ = "Development"

import insanity
import typing


class RelationInputExample(object):
    """This encapsulates a single input example for single sentence relation extraction."""

    def __init__(
            self,
            input_tokens: typing.List[str],
            head_span: typing.List[int],
            head_span_root: typing.List[int],
            tail_span: typing.List[int],
            tail_span_root: typing.List[int],
            relation_type: str
    ):
        """
        This creates an instance on `RelationInputExample`.
        Args:
            input_tokens (list[str]): The list of tokens in the input sentence.
            head_span (list[int]): The list containing the indices of the start and end of the head entity span.
            head_span_root (list[int]): The list containing the index of the root of the head entity span.
            tail_span (list[int]): The list containing the indices of the start and end of the tail entity span.
            tail_span_root (list[int]): The list containing the index of the root of the tail entity span.
            relation_type (str): The relation type being expressed in the input text.

        """

        # Sanitize args.
        insanity.sanitize_type("text_tokens", input_tokens, list)
        insanity.sanitize_type("head_span", head_span, list)
        insanity.sanitize_type("head_span_root", head_span_root, list)
        insanity.sanitize_type("tail_span", tail_span, list)
        insanity.sanitize_type("tail_span_root", tail_span_root, list)
        insanity.sanitize_type("relation_type", relation_type, str)

        # Store args.
        self._input_tokens = input_tokens
        self._head_span = head_span
        self._head_span_root = head_span_root
        self._tail_span = tail_span
        self._tail_span_root = tail_span_root
        self._relation_type = relation_type
        self._input_tokens_prompt = None    # Input tokens + Prompt
        self._input_tokens_prompt_idx = None    # Input idx
        self._head_entity_mask = None  # To indicate head entity position in the input with 1s, and 0s for non entity.
        self._tail_entity_mask = None  # To indicate tail entity position in the input with 1s, and 0s for non entity.
        self._relation_mask = None  # To indicate the position of the MASK token representing the token in the prompt.
        self._relation_type_idx = None  # To indicate the relation type id

    # Properties
    @property
    def head_entity_mask(self) -> typing.Union[typing.List[int], None]:
        """(list[int], None): Specifies the positions of entity position in the input text with 1s."""
        return self._head_entity_mask

    @head_entity_mask.setter
    def head_entity_mask(self, head_entity_mask: typing.List[int]) -> None:
        self._head_entity_mask = head_entity_mask


    @property
    def input_tokens(self) -> typing.List:
        """list[str]: Specifies a list of tokens in the input sentence."""
        return self._input_tokens

    @input_tokens.setter
    def input_tokens(self, input_tokens: typing.List[str]) -> None:
        self._input_tokens = input_tokens

    @property
    def input_tokens_prompt(self) -> typing.Union[typing.List[str], None]:
        """(list[str] or None): Specifies the input tokens + prompt"""
        return self._input_tokens_prompt

    @input_tokens_prompt.setter
    def input_tokens_prompt(self, input_tokens_prompt: typing.List[str]) -> None:
        self._input_tokens_prompt = input_tokens_prompt

    @property
    def input_tokens_prompt_idx(self) -> typing.Union[typing.List[int], None]:
        """(list[int], None): Specifies the vocabulary ids of the tokens in input + prompt."""
        return self._input_tokens_prompt_idx

    @input_tokens_prompt_idx.setter
    def input_tokens_prompt_idx(self, input_tokens_prompt_idx: typing.List[int]) -> None:
        self._input_tokens_prompt_idx = input_tokens_prompt_idx

    @property
    def head_span(self) -> typing.List[int]:
        """list[int]: Specifies the indices of the start and end positions of the head entity span in input tokens."""
        return self._head_span

    @head_span.setter
    def head_span(self, head_span: typing.List[int]) -> None:
        self._head_span = head_span

    @property
    def head_span_root(self) -> typing.List[int]:
        """list[int]: Specifies the index of the root/head of the head entity span."""
        return self._head_span_root

    @head_span_root.setter
    def head_span_root(self, head_span_root: typing.List[int]) -> None:
        self._head_span_root = head_span_root

    @property
    def relation_mask(self) -> typing.Union[typing.List[int], None]:
        """(list[int] or None): Specifies the position of the MASK that represents the relation vector in the input."""
        return self._relation_mask

    @relation_mask.setter
    def relation_mask(self, relation_mask: typing.List[int]) -> None:
        self._relation_mask = relation_mask

    @property
    def tail_span(self) -> typing.List[int]:
        """list[int]: Specifies the indices for the start and end positions of the tail entity span in input tokens."""
        return self._tail_span

    @tail_span.setter
    def tail_span(self, tail_span: typing.List[int]) -> None:
        self._tail_span = tail_span

    @property
    def tail_span_root(self) -> typing.List[int]:
        """list[int]: Specifies the index of the head/root of the tail entity span."""
        return self._tail_span_root

    @tail_span_root.setter
    def tail_span_root(self, tail_span_root: typing.List[int]) -> None:
        self._tail_span_root = tail_span_root

    @property
    def relation_type(self) -> str:
        """str: Specifies the relation type between the head and tail entities."""
        return self._relation_type

    @relation_type.setter
    def relation_type(self, relation_type: str) -> None:
        self._relation_type = relation_type

    @property
    def relation_type_idx(self) -> int:
        """int: Specifies the relation type id."""
        return self._relation_type_idx

    @relation_type_idx.setter
    def relation_type_idx(self, relation_type_idx: int) -> None:
        self._relation_type_idx = relation_type_idx

    @property
    def tail_entity_mask(self) -> typing.Union[typing.List[int], None]:
        """(list[int], None): Specifies the positions of entity position in the input text with 1s."""
        return self._tail_entity_mask

    @tail_entity_mask.setter
    def tail_entity_mask(self, tail_entity_mask: typing.List[int]) -> None:
        self._tail_entity_mask = tail_entity_mask



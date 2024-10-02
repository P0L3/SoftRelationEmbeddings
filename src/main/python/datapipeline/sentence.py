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
__date__ = "02 Mar 2023"
__author__ = "Frank M. Mtumbuka"
__maintainer__ = "Frank M. Mtumbuka"
__email__ = "" ""
__status__ = "Development"

import insanity
import typing


class Sentence(object):
    """This class represents a single sentence that serves as input."""

    def __init__(
            self,
            end: int,
            start: int,
            text: str,
            mention_position: int = 0,
            mention_positions: list = None,
            text_arr: list = None,
            text_augmented: str = None,
            tokenized_text_mask: list = None,
            tokenizer_text_idx: list = None
    ):
        """
        This creates an instance of `Sentence`.
        Args:
            end (int): The end index of the sentence in a given document.
            start (int): The starting index of the sentence in a given document.
            text (str): The actual text of the sentence in a given document.
            mention_position (int): The position of the current mention in the sentence.
            mention_positions (list): The positions of co-referring entities in a sentence.Default value is None.
            text_arr (list): A list of tokens in the input sentence.
            text_augmented (str): The text with special tokens around co-referring entities in a sentence.
            tokenized_text_mask (list): A mask indicating the positions of entity mentions in tokenized text. `1`
                indicates the position of the entity mention, and `0` indicates otherwise.
            tokenizer_text_idx (list): A list of token ids for the input text as given by the specified tokenizer.
        """

        # Sanitize args.
        insanity.sanitize_type("end", end, int)
        insanity.sanitize_type("start", start, int)
        insanity.sanitize_type("text", text, str)
        if mention_position is not None:
            insanity.sanitize_type("mention_position", mention_position, int)
        if mention_positions is not None:
            insanity.sanitize_type("mention_positions", mention_positions, list)
        if text_arr is not None:
            insanity.sanitize_type("text_arr", text_arr, list)
        if text_augmented is not None:
            insanity.sanitize_type("text_augmented", text_augmented, str)
        if tokenized_text_mask is not None:
            insanity.sanitize_type("tokenized_text_mask", tokenized_text_mask, list)
        if tokenizer_text_idx is not None:
            insanity.sanitize_type("tokenizer_text_idx", tokenizer_text_idx, list)

        # Store args
        self._end = end
        self._start = start
        self._text = text
        self._mention_position = mention_position
        self._mention_positions = mention_positions
        self._text_arr = text_arr
        self._text_augmented = text_augmented
        self._tokenized_text_mask = tokenized_text_mask
        self._tokenizer_text_idx = tokenizer_text_idx
        self._coreferent = []
        self._noun_chunks = []

    @property
    def coreferent(self) -> typing.List:
        """Specifies the co-referent."""
        return self._coreferent

    @coreferent.setter
    def coreferent(self, correferent: list) -> None:
        self._coreferent = correferent

    @property
    def end(self) -> int:
        """int: Specifies the end index of a given sentence in the given document."""
        return self._end

    @property
    def mention_position(self) -> typing.Union[int, None]:
        return self._mention_position

    @mention_position.setter
    def mention_position(self, mention_position: int) -> None:
        self._mention_position = mention_position

    @property
    def mention_positions(self) -> list:
        """list: A list of positions of co-referring entities."""
        return self._mention_positions

    @mention_positions.setter
    def mention_positions(self, mention_positions: list) -> None:
        self._mention_positions = mention_positions

    @property
    def noun_chunks(self) -> typing.List:
        return self._noun_chunks

    @noun_chunks.setter
    def noun_chunks(self, noun_chunks: list) -> None:
        self._noun_chunks = noun_chunks

    @property
    def start(self) -> int:
        """int: Specifies the start index of a given sentence in the given document."""
        return self._start

    @property
    def text(self) -> str:
        """str: Specifies the text of the given sentence."""
        return self._text

    @property
    def text_arr(self) -> list:
        """list: Specifies the list of tokens in the sentence as given by spacy."""
        return self._text_arr

    @text_arr.setter
    def text_arr(self, text_arr: list) -> None:
        self._text_arr = text_arr

    @property
    def text_augmented(self) -> typing.Union[str, None]:
        """str: Specifies the text of a given sentence with special token around co-referring entities"""
        return self._text_augmented

    @text_augmented.setter
    def text_augmented(self, text_augmented: str) -> None:
        self._text_augmented = text_augmented

    @property
    def tokenized_text_mask(self) -> list:
        """list: Specifies the entity mask for a given piece tokenized text."""
        return self._tokenized_text_mask

    @tokenized_text_mask.setter
    def tokenized_text_mask(self, tokenized_text_mask: list) -> None:
        self._tokenized_text_mask = tokenized_text_mask

    @property
    def tokenizer_text_idx(self) -> list:
        """list: Specifies the list of token idx given by the tokenizer."""
        return self._tokenizer_text_idx

    @tokenizer_text_idx.setter
    def tokenizer_text_idx(self, tokenizer_text_idx: list) -> None:
        self._tokenizer_text_idx = tokenizer_text_idx

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

import typing


class SentencePairs(object):
    """This encapsulates pretraining sentence pairs"""

    def __init__(self, first_sent: str, second_text: str):
        """
        This creates an instance of the `SentencePairs`
        Args:
            first_sent (str): The actual text of the first sentence.
            second_text (str): The actual text of the second sentence.
        """

        self._first_sent = first_sent
        self._second_sent = second_text
        self._cluster_idx = 0
        self._first_sent_tokens = []
        self._first_sent_ent_a = []
        self._first_sent_ent_b = []
        self._second_sent_tokens = []
        self._second_sent_ent_a = []
        self._second_sent_ent_b = []
        self._first_sent_ent_a_mask = []
        self._first_sent_ent_b_mask = []
        self._second_sent_ent_a_mask = []
        self._second_sent_ent_b_mask = []
        self._second_sent_prompt_mask = []
        self._first_sent_prompt_mask = []

    @property
    def cluster_idx(self) -> int:
        """int: Specifies the cluster idx"""
        return self._cluster_idx

    @cluster_idx.setter
    def cluster_idx(self, cluster_idx: int) -> None:
        self._cluster_idx = int(cluster_idx)

    @property
    def first_sent(self) -> str:
        """str: Specifies the text of the first sentence."""
        return self._first_sent

    @first_sent.setter
    def first_sent(self, first_sent: str) -> None:
        self._first_sent = first_sent

    @property
    def first_sent_ent_a(self) -> typing.List[int]:
        """list[int]: Specifies the start and end positions of entity A in the first sentence."""
        return self._first_sent_ent_a

    @first_sent_ent_a.setter
    def first_sent_ent_a(self, first_sent_ent_a: typing.List[int]) -> None:
        self._first_sent_ent_a = first_sent_ent_a

    @property
    def first_sent_ent_b(self) -> typing.List[int]:
        """list[int]: Specifies the start and end positions of entity B in the first sentence."""
        return self._first_sent_ent_b

    @first_sent_ent_b.setter
    def first_sent_ent_b(self, first_sent_ent_b: typing.List[int]) -> None:
        self._first_sent_ent_b = first_sent_ent_b

    @property
    def first_sent_ent_a_mask(self) -> typing.List[int]:
        """list[int]: Specifies the positions of entity A in the sentence."""
        return self._first_sent_ent_a_mask

    @first_sent_ent_a_mask.setter
    def first_sent_ent_a_mask(self, first_sent_ent_a_mask: typing.List[int]) -> None:
        self._first_sent_ent_a_mask = first_sent_ent_a_mask

    @property
    def first_sent_ent_b_mask(self) -> typing.List[int]:
        """list[int]: Specifies the positions of entity B in the sentence."""
        return self._first_sent_ent_b_mask

    @first_sent_ent_b_mask.setter
    def first_sent_ent_b_mask(self, first_sent_ent_b_mask: typing.List[int]) -> None:
        self._first_sent_ent_b_mask = first_sent_ent_b_mask

    @property
    def first_sent_tokens(self) -> typing.List[str]:
        """list[str]: Specifies the tokens in the first sentence."""
        return self._first_sent_tokens

    @first_sent_tokens.setter
    def first_sent_tokens(self, first_sent_tokens: typing.List[str]) -> None:
        self._first_sent_tokens = first_sent_tokens

    @property
    def first_sent_prompt_mask(self) -> typing.List[int]:
        """list[int]: Specifies the position of the token representing the relation vector."""
        return self._first_sent_prompt_mask

    @first_sent_prompt_mask.setter
    def first_sent_prompt_mask(self, first_sent_prompt_mask: typing.List[int]) -> None:
        self._first_sent_prompt_mask = first_sent_prompt_mask

    @property
    def second_sent_prompt_mask(self) -> typing.List[int]:
        """list[int]: Specifies the position of the token representing the relation vector."""
        return self._second_sent_prompt_mask

    @second_sent_prompt_mask.setter
    def second_sent_prompt_mask(self, second_sent_prompt_mask: typing.List[int]) -> None:
        self._second_sent_prompt_mask = second_sent_prompt_mask

    @property
    def second_sent(self) -> str:
        """str: Specifies the text of the second sentence."""
        return self._second_sent

    @second_sent.setter
    def second_sent(self, second_sent: str) -> None:
        self._second_sent = second_sent

    @property
    def second_sent_ent_a(self) -> typing.List[int]:
        """list[int]: Specifies the start and end positions of entity A in the second sentence."""
        return self._second_sent_ent_a

    @second_sent_ent_a.setter
    def second_sent_ent_a(self, second_sent_ent_a: typing.List[int]) -> None:
        self._second_sent_ent_a = second_sent_ent_a

    @property
    def second_sent_ent_b(self) -> typing.List[int]:
        """list[int]: Specifies the start and end positions of entity B in the second sentence."""
        return self._second_sent_ent_b

    @second_sent_ent_b.setter
    def second_sent_ent_b(self, second_sent_ent_b: typing.List[int]) -> None:
        self._second_sent_ent_b = second_sent_ent_b

    @property
    def second_sent_ent_a_mask(self) -> typing.List[int]:
        """list[int]: Specifies the positions of entities A."""
        return self._second_sent_ent_a_mask

    @second_sent_ent_a_mask.setter
    def second_sent_ent_a_mask(self, second_sent_ent_a_mask: typing.List[int]) -> None:
        self._second_sent_ent_a_mask = second_sent_ent_a_mask

    @property
    def second_sent_ent_b_mask(self) -> typing.List[int]:
        """list[int]: Specifies the positions of entities B."""
        return self._second_sent_ent_b_mask

    @second_sent_ent_b_mask.setter
    def second_sent_ent_b_mask(self, second_sent_ent_b_mask: typing.List[int]) -> None:
        self._second_sent_ent_b_mask = second_sent_ent_b_mask

    @property
    def second_sent_tokens(self) -> typing.List[str]:
        """list[str]: Specifies the tokens in the second sentence."""
        return self._second_sent_tokens

    @second_sent_tokens.setter
    def second_sent_tokens(self, second_sent_tokens: typing.List[str]) -> None:
        self._second_sent_tokens = second_sent_tokens



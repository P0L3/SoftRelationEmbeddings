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
__date__ = "09 Aug 2023"
__author__ = "Frank M. Mtumbuka"
__maintainer__ = "Frank M. Mtumbuka"
__email__ = "" ""
__status__ = "Development"

import abc
import torch
import torch.nn as nn
import typing


class BasePretrainLoss(nn.Module, metaclass=abc.ABCMeta):
    """This class is the base class for all pretraining loss layers."""

    @abc.abstractmethod
    def forward(
            self,
            first_sent_input: torch.FloatTensor,
            first_sent_ent_a_mask: torch.LongTensor,
            first_sent_ent_b_mask: torch.LongTensor,
            first_sent_prompt_mask: torch.LongTensor,
            input_labels: torch.LongTensor,
            second_sent_input: torch.FloatTensor,
            second_sent_ent_a_mask: torch.LongTensor,
            second_sent_ent_b_mask: torch.LongTensor,
            second_sent_prompt_mask: torch.LongTensor
    ) -> typing.Dict:
        """
        This computes the losses as defined in the user configuration.
        Args:
            first_sent_input (::class:`torch.FloatTensor`): The embeddings of the first sentence,
                (batch-size x seq-len x hidden size) tensor.
            first_sent_ent_a_mask (::class:`torch.LongTensor`): The mask indicating the positions of entity A in the
                first sentence with 1s, (batch-size x seq-len) tensor.
            first_sent_ent_b_mask (::class:`torch.LongTensor`): The mask indicating the positions of entity B in the
                first sentence with 1s, (batch-size x seq-len) tensor.
            first_sent_prompt_mask (::class:`torch.LongTensor`): The mask indicating the positions of mask representing
                the relation vector in the first sentence with 1s, (batch-size x seq-len) tensor.
            second_sent_input (::class:`torch.FloatTensor`): The embeddings of the second sentence,
                (batch-size x seq-len x hidden size) tensor.
            second_sent_ent_a_mask (::class:`torch.LongTensor`): The mask indicating the positions of entity A in the
                second sentence with 1s, (batch-size x seq-len) tensor.
            second_sent_ent_b_mask (::class:`torch.LongTensor`): The mask indicating the positions of entity B in the
                second sentence with 1s, (batch-size x seq-len) tensor.
            second_sent_prompt_mask (::class:`torch.LongTensor`): The mask indicating the positions of mask representing
                the relation vector in the second sentence with 1s, (batch-size x seq-len) tensor.
            input_labels (::class:`torch.LongTensor`): The contrastive loss labels, (batch-size x 1) tensor.

        Returns:
            loss (dict): The dictionary of losses.
        """

        # Sanitize args
        assert first_sent_input.shape[0] == first_sent_ent_a_mask.shape[0] == first_sent_prompt_mask.shape[0]
        if second_sent_input is not None:
            assert second_sent_input.shape[0] == second_sent_ent_b_mask.shape[0] == second_sent_prompt_mask.shape[0]
            assert first_sent_input.shape[2] == second_sent_input.shape[2]
            assert second_sent_ent_b_mask.shape[1] == second_sent_prompt_mask.shape[1]
            assert first_sent_input.shape[0] == second_sent_input.shape[0] == input_labels.shape[0]
        assert first_sent_ent_a_mask.shape[1] == first_sent_prompt_mask.shape[1]

    def retrieve_representations(
            self,
            input_seq: torch.FloatTensor,
            mask: torch.Tensor = None
    ) -> torch.FloatTensor:
        """
        This retrieves vectors at specific indices in a tensor.
        If the mask tensor is not specified, the method retrieves representations of the first
        element of each sequence, [CLS] token.
        Args:
            input_seq(:class::`torch.FloatTensor`): A tensor of vector representations as computed by
                the specified encoder. (batch-size x seq-length x hidden-size).
            mask (:class::`torch.Tensor`): This could either be a mask indicating the position of
                named entities in the input sequence, or masked out elements. (batch-size x seq-length)
        Returns:
            representations (:class::`torch.FloatTensor`): A tensor of vectors from specific indices in
                the input tensor. Usually of batch-size x 1 x hidden-size.
        """
        representations = None
        # Kill(zero) unnecessary representations
        representations = input_seq * mask.unsqueeze(2)

        # Sum the representations along the first dimension
        representations = torch.sum(representations, 1)

        return representations

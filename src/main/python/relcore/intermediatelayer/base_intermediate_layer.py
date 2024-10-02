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
__date__ = "08 Mar 2023"
__author__ = "Frank M. Mtumbuka"
__maintainer__ = "Frank M. Mtumbuka"
__email__ = "" ""
__status__ = "Development"

import abc

import insanity
import torch
import torch.nn as nn
import typing


class BaseIntermediateLayer(nn.Module, metaclass=abc.ABCMeta):
    """This class is the base class for all intermediate operations between the encoder and classifier."""

    @abc.abstractmethod
    def forward(
            self,
            input_seq: torch.FloatTensor,
            mask: torch.Tensor = None,
            labels: torch.Tensor = None,
            rel_embed: bool = False
    ) -> typing.Tuple[torch.FloatTensor, typing.Any]:
        """
        This computes the intermediate operation as specified in the experiment configuration.
        Args:
            input_seq (:class::`torch.FloatTensor`): A tensor of vector representations as computed by
                the specified encoder. (batch-size x seq-length x hidden-size).
            mask (:class::`torch.Tensor`): This could either be a mask indicating the position of
                named entities in the input sequence, or masked out elements. (batch-size x seq-length).
            labels (:class::`torch.Tensor`): This could either be  masked language modelling losses or
                labels for contrastive loss.
            rel_embed (bool): This specifies whether the input embeddings are relation embeddings or not.

        Returns:
            (logits, loss)
        """
        # Sanitize args.
        insanity.sanitize_type("input_seq", input_seq, torch.FloatTensor)
        if mask is not None:
            insanity.sanitize_type("mask", mask, torch.Tensor)
            assert (input_seq.shape[0], input_seq.shape[1]) == (mask.shape[0], mask.shape[1])

        if labels is not None:
            insanity.sanitize_type("labels", labels, torch.Tensor)
            assert (input_seq.shape[0], input_seq.shape[1]) == (labels.shape[0], labels.shape[1])

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

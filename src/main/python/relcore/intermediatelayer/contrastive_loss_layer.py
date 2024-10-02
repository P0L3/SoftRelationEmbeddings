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
__date__ = "09 Mar 2023"
__author__ = "Frank M. Mtumbuka"
__maintainer__ = "Frank M. Mtumbuka"
__email__ = "" ""
__status__ = "Development"

import relcore.intermediatelayer.base_intermediate_layer as base_intermediate_layer
import insanity
import torch
import typing
from pytorch_metric_learning import losses


class ContrastiveLossLayer(base_intermediate_layer.BaseIntermediateLayer):
    """This computes the contrastive loss."""

    def __init__(self, temperature: float = 0.07):
        """
        This creates an instance of the `ContrastiveLossLayer`.
        Args:
            temperature (float): This is tau in the contrastive loss equation.
        """

        super().__init__()

        # Sanitize args.
        insanity.sanitize_type("temperature", temperature, float)
        insanity.sanitize_range("temperature", temperature, minimum=0.00)

        # Create loss function.
        self._contrastive_loss = losses.NTXentLoss(temperature=temperature)

    def forward(
            self,
            input_seq: torch.FloatTensor,
            mask: torch.Tensor = None,
            labels: torch.Tensor = None
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

        Returns:
            (logits, loss)
        """
        insanity.sanitize_type("input_seq", input_seq, torch.FloatTensor)
        if mask is not None:
            insanity.sanitize_type("mask", mask, torch.Tensor)

        if labels is not None:
            insanity.sanitize_type("labels", labels, torch.Tensor)

        entity_representations = self.retrieve_representations(input_seq, mask)
        cp_loss = self._contrastive_loss(entity_representations.squeeze(1), labels.view(-1))

        return entity_representations, cp_loss


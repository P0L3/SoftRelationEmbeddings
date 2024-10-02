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
__date__ = "13 Mar 2023"
__author__ = "Frank M. Mtumbuka"
__maintainer__ = "Frank M. Mtumbuka"
__email__ = "" ""
__status__ = "Development"

import typing
import relcore.encoder.base_encoder as base_encoder
import relcore.pretraininglosses.base_pretrain_losses as base_pretraining_losses
import insanity
import torch
import torch.nn as nn


class RelEncoder(nn.Module):
    """
    This encapsulates all modules for encoding as specified in the experiment configuration for pretraining purposes.
    """

    def __init__(
            self,
            encoder: base_encoder.BaseEncoder,
            pretraining_loss: base_pretraining_losses.BasePretrainLoss
    ):
        """
        This creates an instance of `RelEncoder`
        Args:
            encoder (::class:`base_encoder.BaseEncoder`): The encoder as specified in the
                experiment configuration.
            pretraining_loss (::class:`base_pretraining_losses.BasePretrainingLoss`): The pretraining loss specified.
        """

        super().__init__()

        # Sanitize args.
        insanity.sanitize_type("encoder", encoder, base_encoder.BaseEncoder)
        insanity.sanitize_type("pretraining_loss", pretraining_loss, base_pretraining_losses.BasePretrainLoss)

        # Create components
        self._encoder = encoder
        self._pretraining_loss_layer = pretraining_loss

    @property
    def encoder(self) -> base_encoder.BaseEncoder:
        """Specifies the encoder being used."""
        return self._encoder

    @property
    def pretraining_loss_layer(self) -> base_pretraining_losses.BasePretrainLoss:
        """Specifies the pretraining loss layer."""
        return self._pretraining_loss_layer

    def forward(
            self,
            sent_a_input: torch.LongTensor,
            sent_a_ent_a_mask: torch.LongTensor,
            sent_a_ent_b_mask: torch.LongTensor,
            sent_a_prompt_mask: torch.LongTensor,
            sent_a_mlm_labels: torch.LongTensor,
            sent_a_attention_mask: torch.LongTensor,
            contrastive_labels: torch.LongTensor,
            sent_b_input: torch.LongTensor = None,
            sent_b_ent_a_mask: torch.LongTensor = None,
            sent_b_ent_b_mask: torch.LongTensor = None,
            sent_b_prompt_mask: torch.LongTensor = None,
            sent_b_mlm_labels: torch.LongTensor = None,
            sent_b_attention_mask: torch.LongTensor = None

    ) -> typing.Dict:
        """
        Args:
            sent_a_input (::class:`torch.LongTensor`): A tensor of input examples, (batch-size x seq-len).
            sent_a_ent_a_mask (::class:`torch.LongTensor`): A tensor of entity masks with 1s representing the positions
                of entity A and 0s non-entity tokens in the first sentence, (batch-size x seq-len).
            sent_a_ent_b_mask (::class:`torch.LongTensor`): A tensor of entity masks with 1s representing the positions
                of entity B and 0s non-entity tokens in the first sentence, (batch-size x seq-len).
            sent_a_prompt_mask (::class:`torch.LongTensor`): A tensor of prompt masks with 1s representing the positions
                of mask representing a relation and 0s otherwise in the first sentence, (batch-size x seq-len).
            sent_a_mlm_labels (::class:`torch.LongTensor`): A tensor of mlm labels for the first sentence,
                (batch-size x seq-len).
            sent_a_attention_mask (::class:`torch.LongTensor`): A tensor of attention masks with 1s tokens to be
                attended to, the actual tokens in first sentences, and 0s for padding tokens,(batch-size x seq-len).
            sent_b_input (::class:`torch.LongTensor`): A tensor of input examples, (batch-size x seq-len).
            sent_b_ent_a_mask (::class:`torch.LongTensor`): A tensor of entity masks with 1s representing the
                positions of entity A and 0s non-entity tokens in the second sentence, (batch-size x seq-len).
            sent_b_ent_b_mask (::class:`torch.LongTensor`): A tensor of entity masks with 1s representing the positions
                of entity B and 0s non-entity tokens in the second sentence, (batch-size x seq-len).
            sent_b_prompt_mask (::class:`torch.LongTensor`): A tensor of prompt masks with 1s representing the positions
                of mask representing a relation and 0s otherwise in the second sentence, (batch-size x seq-len).
            sent_b_mlm_labels (::class:`torch.LongTensor`): A tensor of mlm labels for the first sentence,
                (batch-size x seq-len).
            sent_b_attention_mask (::class:`torch.LongTensor`): A tensor of attention masks with 1s tokens to be
                attended to, the actual tokens in second sentences, and 0s for padding tokens,(batch-size x seq-len).
            contrastive_labels (::class:`torch.LongTensor`): A tensor of contrastive labels, (batch-size x 1)
        Returns:
            loss (dict) : The dictionary of losses.
        """
        first_sent_input = None
        second_sent_input = None
        # Compute embeddings and mlm loss using the encoder.
        first_sent_input, _ = self._encoder(sent_a_input, sent_a_mlm_labels, sent_a_attention_mask)
        if sent_b_input is not None:
            second_sent_input, _ = self._encoder(sent_b_input, sent_b_mlm_labels, sent_b_attention_mask)

        # Compute losses.
        losses = self._pretraining_loss_layer(
            first_sent_input,
            sent_a_ent_a_mask,
            sent_a_ent_b_mask,
            sent_a_prompt_mask,
            contrastive_labels,
            second_sent_input,
            sent_b_ent_a_mask,
            sent_b_ent_b_mask,
            sent_b_prompt_mask
        )

        # losses["sent_a_mlm_loss"] = first_sent_mlm_loss
        # losses["sent_b_mlm_loss"] = second_sent_mlm_loss
        return losses

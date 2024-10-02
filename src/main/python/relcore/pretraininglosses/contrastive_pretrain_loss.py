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

import insanity
import relcore.pretraininglosses.base_pretrain_losses as base_pretrain_losses
import torch
import torch.nn as nn
import typing
from pytorch_metric_learning import losses


class ContrastivePretrainLoss(base_pretrain_losses.BasePretrainLoss):
    """This implements various contrastive losses for our pretraining."""

    def __init__(self, hidden_size: int, h_and_t: bool = False, rel_mask: bool = False, temperature: float = 0.07):
        """
        This creates an instance of the `ContrastivePretrainLoss`.
        Args:
            hidden_size (int): The hidden dimension of the representations.
            h_and_t (bool): Specifies whether to consider the head and tail entities in the loss.
            rel_mask (bool): Specifies whether to consider the representation of the relation mask in the loss.
            temperature (float): This is tau in the contrastive loss equation.
        """

        # Super class call.
        super().__init__()

        # Sanitize args.
        insanity.sanitize_type("hidden_size", hidden_size, int)
        insanity.sanitize_type("h_and_t", h_and_t, bool)
        insanity.sanitize_type("temperature", temperature, float)
        insanity.sanitize_range("temperature", temperature, minimum=0.00)
        insanity.sanitize_type("rel_mask", rel_mask, bool)
        insanity.sanitize_range("hidden_size", hidden_size, minimum=1)

        # Attributes
        self._hidden_size = hidden_size
        self._h_and_t = h_and_t
        self._rel_mask = rel_mask

        # Create reducer to be used in the case when vectors are concatenated
        self._double_reducer = nn.Linear(self._hidden_size * 2, self._hidden_size)  # When 2 vectors are concatenated.
        self._triple_reducer = nn.Linear(self._hidden_size * 3, self._hidden_size)  # When 3 vectors are concatenated.

        # Create loss function.
        self._contrastive_loss = losses.NTXentLoss(temperature=temperature)

    def forward(
            self,
            first_sent_input: torch.FloatTensor,
            first_sent_ent_a_mask: torch.LongTensor,
            first_sent_ent_b_mask: torch.LongTensor,
            first_sent_prompt_mask: torch.LongTensor,
            input_labels: torch.LongTensor,
            second_sent_input: torch.FloatTensor = None,
            second_sent_ent_a_mask: torch.LongTensor = None,
            second_sent_ent_b_mask: torch.LongTensor = None,
            second_sent_prompt_mask: torch.LongTensor = None
    ) -> typing.Dict:
        """See base class."""
        loss_dict = {
            "h_and_t": torch.FloatTensor([0.0]),
            "rel_mask": torch.FloatTensor([0.0]),
            "all": torch.FloatTensor([0.0])
        }

        if self._h_and_t and self._rel_mask:
            ent_a_sent_a = None
            ent_b_sent_a = None
            rel_mask_sent_a = None
            ent_a_sent_b = None
            ent_b_sent_b = None
            rel_mask_sent_b = None
            sent_a_h_t = None
            sent_b_h_t = None
            h_t_representations = None
            rel_representations = None

            # Retrieve first_sent_ent_a representations
            ent_a_sent_a = self.retrieve_representations(first_sent_input, first_sent_ent_a_mask)

            # Retrieve first_sent_ent_b representations
            ent_b_sent_a = self.retrieve_representations(first_sent_input, first_sent_ent_b_mask)

            # Retrieve first_sent_rel_mask representations
            rel_mask_sent_a = self.retrieve_representations(first_sent_input, first_sent_prompt_mask)

            # Reduce representations: (batch-size x hidden-size * 2) -> (batch-size x hidden-size)
            sent_a_h_t = self._double_reducer(torch.cat((ent_a_sent_a, ent_b_sent_a), dim=1))

            if second_sent_input is not None:
                # Retrieve second_sent_ent_a representations
                ent_a_sent_b = self.retrieve_representations(second_sent_input, second_sent_ent_a_mask)

                # Retrieve second_sent_ent_b representations
                ent_b_sent_b = self.retrieve_representations(second_sent_input, second_sent_ent_b_mask)

                # Retrieve second_sent_rel_mask representations
                rel_mask_sent_b = self.retrieve_representations(second_sent_input, second_sent_prompt_mask)

                # Reduce representations: (batch-size x hidden-size * 2) -> (batch-size x hidden-size)
                sent_b_h_t = self._double_reducer(torch.cat((ent_a_sent_b, ent_b_sent_b), dim=1))

                # Stack representations: (batch-size x hidden-size) -> (batch-size * 2 x hidden-size)
                h_t_representations = torch.cat((sent_a_h_t, sent_b_h_t), dim=0)
                rel_representations = torch.cat((rel_mask_sent_a, rel_mask_sent_b), dim=0)

                # Stack labels
                labels = torch.cat((input_labels, input_labels), dim=0)
                # Compute losses
                h_t_loss = self._contrastive_loss(h_t_representations, labels)
                rel_mask_loss = self._contrastive_loss(rel_representations, labels)

                # Update loss dict
                loss_dict["h_and_t"] = h_t_loss
                loss_dict["rel_mask"] = rel_mask_loss
            else:
                # Compute losses
                h_t_loss = self._contrastive_loss(sent_a_h_t, input_labels)
                rel_mask_loss = self._contrastive_loss(rel_mask_sent_a, input_labels)

                # Update loss dict
                loss_dict["h_and_t"] = h_t_loss
                loss_dict["rel_mask"] = rel_mask_loss

        elif self._h_and_t:
            ent_a_sent_a = None
            ent_b_sent_a = None
            rel_mask_sent_a = None
            ent_a_sent_b = None
            ent_b_sent_b = None
            rel_mask_sent_b = None
            sent_a_h_t = None
            sent_b_h_t = None
            h_t_representations = None
            rel_representations = None

            # Retrieve first_sent_ent_a representations
            ent_a_sent_a = self.retrieve_representations(first_sent_input, first_sent_ent_a_mask)

            # Retrieve first_sent_ent_b representations
            ent_b_sent_a = self.retrieve_representations(first_sent_input, first_sent_ent_b_mask)

            # Reduce representations: (batch-size x hidden-size * 2) -> (batch-size x hidden-size)
            sent_a_h_t = self._double_reducer(torch.cat((ent_a_sent_a, ent_b_sent_a), dim=1))

            if second_sent_input is not None:
                # Retrieve second_sent_ent_a representations
                ent_a_sent_b = self.retrieve_representations(second_sent_input, second_sent_ent_a_mask)

                # Retrieve second_sent_ent_b representations
                ent_b_sent_b = self.retrieve_representations(second_sent_input, second_sent_ent_b_mask)

                # Reduce representations: (batch-size x hidden-size * 2) -> (batch-size x hidden-size)
                sent_b_h_t = self._double_reducer(torch.cat((ent_a_sent_b, ent_b_sent_b), dim=1))

                # Stack representations: (batch-size x hidden-size) -> (batch-size * 2 x hidden-size)
                h_t_representations = torch.cat((sent_a_h_t, sent_b_h_t), dim=0)

                # Stack labels
                labels = torch.cat((input_labels, input_labels), dim=0)

                # Compute losses
                h_t_loss = self._contrastive_loss(h_t_representations, labels)

                # Update loss dict
                loss_dict["h_and_t"] = h_t_loss
            else:

                # Compute losses
                h_t_loss = self._contrastive_loss(sent_a_h_t, input_labels)

                # Update loss dict
                loss_dict["h_and_t"] = h_t_loss

        elif self._rel_mask:

            rel_mask_sent_a = None
            rel_mask_sent_b = None

            # Retrieve first_sent_rel_mask representations
            rel_mask_sent_a = self.retrieve_representations(first_sent_input, first_sent_prompt_mask)

            if second_sent_input is not None:
                # Retrieve second_sent_rel_mask representations
                rel_mask_sent_b = self.retrieve_representations(second_sent_input, second_sent_prompt_mask)

                # Stack representations: (batch-size x hidden-size) -> (batch-size * 2 x hidden-size)
                rel_representations = torch.cat((rel_mask_sent_a, rel_mask_sent_b), dim=0)

                # Stack labels
                labels = torch.cat((input_labels, input_labels), dim=0)

                # Compute loss
                rel_mask_loss = self._contrastive_loss(rel_representations, labels)

                # Update loss dict
                loss_dict["rel_mask"] = rel_mask_loss
            else:

                # Compute loss
                rel_mask_loss = self._contrastive_loss(rel_mask_sent_a, input_labels)

                # Update loss dict
                loss_dict["rel_mask"] = rel_mask_loss
        else:
            ent_a_sent_a = None
            ent_b_sent_a = None
            rel_mask_sent_a = None
            ent_a_sent_b = None
            ent_b_sent_b = None
            rel_mask_sent_b = None
            sent_a_h_t = None
            sent_b_h_t = None
            h_t_representations = None
            rel_representations = None

            # Retrieve first_sent_ent_a representations
            ent_a_sent_a = self.retrieve_representations(first_sent_input, first_sent_ent_a_mask)

            # Retrieve first_sent_ent_b representations
            ent_b_sent_a = self.retrieve_representations(first_sent_input, first_sent_ent_b_mask)

            # Retrieve first_sent_rel_mask representations
            rel_mask_sent_a = self.retrieve_representations(first_sent_input, first_sent_prompt_mask)

            # Reduce representations: (batch-size x hidden-size * 3) -> (batch-size x hidden-size)
            sent_a_rep = self._triple_reducer(torch.cat((ent_a_sent_a, ent_b_sent_a, rel_mask_sent_a), dim=1))

            if second_sent_input is not None:
                # Retrieve second_sent_ent_a representations
                ent_a_sent_b = self.retrieve_representations(second_sent_input, second_sent_ent_a_mask)

                # Retrieve second_sent_ent_b representations
                ent_b_sent_b = self.retrieve_representations(second_sent_input, second_sent_ent_b_mask)

                # Retrieve second_sent_rel_mask representations
                rel_mask_sent_b = self.retrieve_representations(second_sent_input, second_sent_prompt_mask)

                # Reduce representations: (batch-size x hidden-size * 3) -> (batch-size x hidden-size)
                sent_b_rep = self._triple_reducer(torch.cat((ent_a_sent_b, ent_b_sent_b, rel_mask_sent_b), dim=1))

                # Stack representations: (batch-size x hidden-size) -> (batch-size * 2 x hidden-size)
                representations = torch.cat((sent_a_rep, sent_b_rep), dim=0)

                # Stack labels
                labels = torch.cat((input_labels, input_labels), dim=0)

                # Compute losses
                all_loss = self._contrastive_loss(representations, labels)

                # Update loss dict
                loss_dict["all"] = all_loss

            else:

                # Compute losses
                all_loss = self._contrastive_loss(sent_a_rep, input_labels)

                # Update loss dict
                loss_dict["all"] = all_loss

        return loss_dict


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
__date__ = "06 Feb 2023"
__author__ = "Frank M. Mtumbuka"
__maintainer__ = "Frank M. Mtumbuka"
__email__ = "" ""
__status__ = "Development"

import relcore.classifier.base_classifier as base_classifier
import relcore.encoder.base_encoder as base_encoder
import relcore.intermediatelayer.base_intermediate_layer as base_intermediate_layer
import insanity
import torch
import torch.nn as nn
import typing


class RelCoreModel(nn.Module):
    """This model encapsulates all modules for the EnCoreModel as specified in the experiment configuration."""

    # Constructor.
    def __init__(
            self,
            classifier: base_classifier.BaseClassifier,
            encoder: base_encoder.BaseEncoder,
            intermediate_layer: base_intermediate_layer.BaseIntermediateLayer,
            ht: bool = False,
            rel_mask: bool = False
    ):
        """
        This creates a new instance of `RelCoreModel`.
        Args:
            classifier (:class::`base_classifier.BaseClassifier`): The classifier to use.
            encoder (:class::`base_encoder.BaseEncoder`): The encoder to use for relations.
            intermediate_layer (:class::`base_intermediate_layer.BaseIntermediateLayer`): The intermediate layer
                to use.
            ht (bool): Specifies whether to use the head and tail representation to predict the relation type.
            rel_mask (bool): Specifies to use the relation mask to predict the relation type.
        """

        super().__init__()

        # Sanitize args.
        insanity.sanitize_type("classifier", classifier, base_classifier.BaseClassifier)
        insanity.sanitize_type("encoder", encoder, base_encoder.BaseEncoder)
        insanity.sanitize_type("intermediate_layer", intermediate_layer, base_intermediate_layer.BaseIntermediateLayer)

        # Store args.
        self._classifier = classifier
        self._encoder = encoder
        self._intermediate_layer = intermediate_layer
        self._entity_encoder = None
        self._ht = ht
        self._rel_mask = rel_mask

    # Properties.
    @property
    def classifier(self) -> base_classifier.BaseClassifier:
        """(:class::`base_classifier.BaseClassifier`): Specifies the classifier being used."""
        return self._classifier

    @classifier.setter
    def classifier(self, classifier: base_classifier.BaseClassifier) -> None:
        insanity.sanitize_type("classifier", classifier, base_classifier.BaseClassifier)
        self._classifier = classifier

    @property
    def encoder(self) -> typing.Union[base_encoder.BaseEncoder, None]:
        """(:class::`base_encoder.BaseEncoder`): Specifies the encoder being used for relations."""
        return self._encoder

    @encoder.setter
    def encoder(self, encoder: base_encoder.BaseEncoder) -> None:
        insanity.sanitize_type("encoder", encoder, base_encoder.BaseEncoder)
        self._encoder = encoder

    @property
    def entity_encoder(self) -> typing.Union[base_encoder.BaseEncoder, None]:
        """(:class::`base_encoder.BaseEncoder`): Specifies the additional entity encoder  if specified."""
        return self._entity_encoder

    @entity_encoder.setter
    def entity_encoder(self, entity_encoder: base_encoder.BaseEncoder) -> None:
        insanity.sanitize_type("entity_encoder", entity_encoder, base_encoder.BaseEncoder)
        self._entity_encoder = entity_encoder

    @property
    def intermediate_layer(self) -> base_intermediate_layer.BaseIntermediateLayer:
        """:class::`base_intermediate_layer.BaseIntermediateLayer`: Specifies the intermediate layer being used."""
        return self._intermediate_layer

    @intermediate_layer.setter
    def intermediate_layer(self, intermediate_layer: base_intermediate_layer.BaseIntermediateLayer) -> None:
        insanity.sanitize_type("intermediate_layer", intermediate_layer, base_intermediate_layer.BaseIntermediateLayer)
        self._intermediate_layer = intermediate_layer

    def predict(
            self,
            input_seq: torch.LongTensor,
            head_entity_mask: torch.LongTensor,
            tail_entity_mask: torch.LongTensor,
            relation_mask: torch.LongTensor
    ) -> torch.FloatTensor:
        """
        This computes the predictions.
        Args:
            input_seq (::class:`torch.LongTensor`): A tensor of token ids for input tokens, (batch-size x seq-len).
            entity_mask (::class:`torch.LongTensor`): A tensor of 0s and 1s of size (batch-size x seq-len). 1s indicate
                the positions of the entities in the input sequence.
            relation_mask (::class:`torch.LongTensor`): A tensor of 0s and 1s of size (batch-size x seq-len). 1s indicate
                the positions of the mask token that will represent the relation.

        Returns:
            predictions (::class:`torch.FloatTensor`): The computed predictions
        """

        # Check dimensions
        assert input_seq.shape[0] == head_entity_mask.shape[0] == relation_mask.shape[0]
        assert input_seq.shape[1] == head_entity_mask.shape[1] == relation_mask.shape[1]

        # Encode sequences
        input_encoding, _ = self._encoder(input_seq)

        # Important characteristics
        batch_size = input_encoding.shape[0]
        embedding_dim = input_encoding.shape[2]

        # Extract H+T representations
        h_representations, _ = self._intermediate_layer(input_encoding, head_entity_mask)

        t_representations, _ = self._intermediate_layer(input_encoding, tail_entity_mask)

        # Extract relation mask representations
        relation_mask_representations, _ = self._intermediate_layer(input_encoding, relation_mask)

        representations = None
        if self._ht and self._rel_mask:
            # Concatenate h_t and relation representations
            representations = torch.cat(
                (h_representations, t_representations, relation_mask_representations),
                dim=1
            )
        elif self._ht:
            representations = torch.cat(
                (h_representations, t_representations),
                dim=1
            )
        elif self._rel_mask:
            representations = relation_mask_representations
        else:
            # TODO! Better this condition
            representations = h_representations

        # (batch-size x hidden-dim * 3) -> (batch-size x 1 x hidden-dim * 3)
        representations = representations.unsqueeze(1)

        # Compute classification preds
        classification_preds = self._classifier.predict(representations)

        return classification_preds

    def forward(
            self,
            input_seq: torch.LongTensor,
            head_entity_mask: torch.LongTensor,
            tail_entity_mask: torch.LongTensor,
            relation_mask: torch.LongTensor,
            mlm_labels: torch.Tensor = None
    ) -> typing.Dict:
        """
        This computes the predictions.
        Args:
            input_seq (::class:`torch.LongTensor`): A tensor of token ids for input tokens, (batch-size x seq-len).
            entity_mask (::class:`torch.LongTensor`): A tensor of 0s and 1s of size (batch-size x seq-len). 1s indicate
                the positions of the entities in the input sequence.
            relation_mask (::class:`torch.LongTensor`): A tensor of 0s and 1s of size (batch-size x seq-len). 1s indicate
                the positions of the mask token that will represent the relation.
            mlm_labels (::class:`torch.LongTensor`): A tensor of mlm labels, (batch-size x seq-len).

        Returns:
            {
                "classification_preds": The relation classification predictions.,
                "mlm_loss": The masked language modelling loss.
            }
        """

        # Check dimensions
        assert input_seq.shape[0] == head_entity_mask.shape[0] == relation_mask.shape[0]
        assert input_seq.shape[1] == head_entity_mask.shape[1] == relation_mask.shape[1]

        # Encode sequences
        input_encoding, mlm_loss = self._encoder(input_seq, mlm_labels)

        # Important characteristics
        batch_size = input_encoding.shape[0]
        embedding_dim = input_encoding.shape[2]

        # Extract H+T representations
        h_representations, _ = self._intermediate_layer(input_encoding, head_entity_mask)
        t_representations, _ = self._intermediate_layer(input_encoding, tail_entity_mask)

        # Extract relation mask representations
        relation_mask_representations, _ = self._intermediate_layer(input_encoding, relation_mask)

        representations = None
        if self._ht and self._rel_mask:
            # Concatenate h_t and relation representations
            representations = torch.cat(
                (h_representations, t_representations, relation_mask_representations),
                dim=1
            )
        elif self._ht:
            representations = torch.cat(
                (h_representations, t_representations),
                dim=1
            )
        elif self._rel_mask:
            representations = relation_mask_representations
        else:
            # TODO! Better this condition
            representations = h_representations

        # (batch-size x hidden-dim * 3) -> (batch-size x 1 x hidden-dim * 3)
        representations = representations.unsqueeze(1)

        # Compute classification preds
        classification_logits = self._classifier(representations)

        return {
            "classfication_preds": classification_logits,
            "mlm_loss": mlm_loss
        }

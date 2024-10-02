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
__date__ = "01 Apr 2023"
__author__ = "Frank M. Mtumbuka"
__maintainer__ = "Frank M. Mtumbuka"
__email__ = "" ""
__status__ = "Development"

import relcore.classifier.base_classifier as base_classifier
import insanity
import torch
import torch.nn as nn


class MLPClassifier(base_classifier.BaseClassifier):
    """This implements an MLP based classifier."""
    def __init__(
            self,
            *args,
            hidden_layers: int = 1,
            dropout_rate: float = 0.1,
            ht: bool = False,
            rel_mask: bool = False,
            **kwargs
    ):
        """
        This creates an instance of `MLPRelClassifier`.
        Args:
            *args: See meth:`base_relation_classifier.BaseRelationClassifier.__init__`
            hidden_layers (int): The number of layers to use.
            dropout_rate (float): The dropout rate used for both attention and residual dropout.
            ht (bool): Specifies whether to use the head and tail representations.
            rel_mask (bool): Specifies whether to use the relation mask representations.
            **kwargs: See meth:`base_relation_classifier.BaseRelationClassifier.__init__`
        """
        super().__init__(*args, **kwargs)

        # Sanitize args.
        insanity.sanitize_type("hidden_layers", hidden_layers, int)
        insanity.sanitize_type("dropout_rate", dropout_rate, float)
        insanity.sanitize_range("hidden_layers", hidden_layers, minimum=1)
        insanity.sanitize_range("dropout_rate", dropout_rate, minimum=0.00)

        # store args that are needed later on
        # self._hidden_layers = hidden_layers

        # create the MLP that is used to process input sequences
        # layers = []
        # last_size = self._input_size
        # decay = (self._input_size - self._num_classes) // (self._hidden_layers + 1)
        # for idx in range(self._hidden_layers):
        #     layers.append(nn.Linear(last_size, last_size - decay))
        #     layers.append(nn.ReLU())
        #     layers.append(nn.Dropout(p=dropout_rate))
        #     last_size -= decay
        # layers.append(nn.Linear(last_size, self._num_classes))
        # self._mlp = nn.Sequential(*layers)
        hidden_size = 0
        if ht and rel_mask:
            hidden_size = self.input_size * 3
        elif ht:
            hidden_size = self.input_size * 2
        else:
            hidden_size = self.input_size

        self._mlp = nn.Sequential(
            nn.Linear(hidden_size, self.input_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.input_size, self.num_classes)
        )

        self.reset_parameters()

    @property
    def hidden_layers(self) -> int:
        """int: Specifies the number of hidden layers."""
        return self._hidden_layers

    def classify(
            self,
            input_seq: torch.FloatTensor
    ) -> torch.FloatTensor:
        return self._mlp(input_seq)

    def reset_parameters(self) -> None:
        """Resets all tunable parameters of the module."""
        for layer in self._mlp:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

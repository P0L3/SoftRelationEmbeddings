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

import abc
import insanity
import torch
import torch.nn as nn
import typing


class BaseClassifier(nn.Module, metaclass=abc.ABCMeta):
    """This is the base class that all classifiers will extend."""

    # Constructor
    def __init__(self, input_size: int, num_classes: int):
        """
        This creates an instance of `BaseRelationClassifier`.
        Args:
            input_size (int): This is the dimension of the input representations.
            num_classes (int): This is the total number of relation classes under consideration.
        """

        super().__init__()

        # Sanitize args.
        insanity.sanitize_type("input_size", input_size, int)
        insanity.sanitize_type("num_classes", num_classes, int)
        insanity.sanitize_range("input_size", input_size, minimum=1)
        insanity.sanitize_range("num_classes", num_classes, minimum=1)

        # Store args.
        self._input_size = input_size
        self._num_classes = num_classes

    # Properties
    @property
    def input_size(self) -> int:
        """int: Specifies the dimension of the computed representations."""
        return self._input_size

    @property
    def num_classes(self) -> int:
        """int: Specifies the number of entity classes under consideration."""
        return self._num_classes

    # Methods
    @abc.abstractmethod
    def classify(
            self,
            input_seq: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Given a representations of a sequence, this method predicts the class of the relation embedded in the
        representations.
        Args:
            input_seq (:class::`torch.FloatTensor`): A tensor of representations of the tokens to be classified,
                (batch-size x 1 x hidden-size).

        Returns:
            predictions (:class::`torch.FloatTensor`): A tensor of relation class predictions,
                (batch-size x 1 x num-classes).
        """

    def predict(self, input_seq: torch.FloatTensor) -> torch.FloatTensor:
        """
        This computes predictions over the label space.
        Args:
            input_seq (::class:`torch.FloatTensor`): Relation representations.

        Returns:
            predictions (::class:`torch.FloatTensor`): The softmax of logits.
        """
        logits = self.classify(input_seq)
        return torch.softmax(logits, dim=2).squeeze(1)

    def forward(
            self,
            input_seq: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Given a representations of a sequence, this method predicts the class of the relation embedded in the
        representations.
        Args:
            input_seq (:class::`torch.FloatTensor`): A tensor of representations of the tokens to be classified,
                (batch-size x 1 x hidden-size).

        Returns:
            predictions (:class::`torch.FloatTensor`): A tensor of relation class predictions,
                (batch-size x 1 x num-classes).
        """

        return self.classify(input_seq)

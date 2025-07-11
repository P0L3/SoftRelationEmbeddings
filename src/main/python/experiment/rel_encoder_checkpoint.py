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
__date__ = "27 Mar 2023"
__author__ = "Frank M. Mtumbuka"
__maintainer__ = "Frank M. Mtumbuka"
__email__ = "" ""
__status__ = "Development"

import expbase as xb
import insanity
import torch
import typing


class RelEncoderCheckpoint(object):
    """This represents a saved Entity Encoder Checkpoint."""

    def __init__(self, epoch: int, encoder_state: dict, loss_layer_state: dict, optimizer_state: dict):
        """
        This creates an instance of the `EntityEncoderCheckpoint`.
        Args:
            epoch (int): The training epoch at which the encoder state was saved.
            encoder_state (dict): The state that describes the encoder at the end of epoch when the checkpoint was
                being saved.
            loss_layer_state (dict): The state that describes the loss layer state at the end of the epoch when the
                checkpoint was being saved.
            optimizer_state (dict): The state that describes the optimizer at the point when the checkpoint was being
                saved.
        """

        super().__init__()

        # Define and store attributes.
        self._epoch = epoch
        self._encoder_state = encoder_state
        self._loss_layer_state = loss_layer_state
        self._optimizer_state = optimizer_state
        self._average_loss = 0.00

    @property
    def average_loss(self) -> float:
        """float: Specifies the average loss achieved when using this checkpoint."""
        return self._average_loss

    @average_loss.setter
    def average_loss(self, average_loss: float) -> None:
        self._average_loss = float(average_loss)

    @property
    def encoder_state(self) -> dict:
        """dict: Specifies the state of the encoder when the checkpoint was being created."""
        return self._encoder_state

    @encoder_state.setter
    def encoder_state(self, encoder_state: dict) -> None:
        self._encoder_state = encoder_state

    @property
    def epoch(self) -> int:
        """int: Specifies the training epoch after which the checkpoint was created."""
        return self._epoch

    @epoch.setter
    def epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    @property
    def loss_layer_state(self) -> dict:
        """dict: Specifies the state of the loss layer when the checkpoint was being created."""
        return self._loss_layer_state

    @loss_layer_state.setter
    def loss_layer_state(self, loss_layer_state: dict) -> None:
        self._loss_layer_state = loss_layer_state

    @property
    def optimizer_state(self) -> dict:
        """dict: Specifies the state of the optimizer when the checkpoint was being created."""
        return self._optimizer_state

    @optimizer_state.setter
    def optimizer_state(self, optimizer_state: dict) -> None:
        self._optimizer_state = optimizer_state

    def dump(self, path: str) -> typing.Any:
        torch.save(self, path)

    @staticmethod
    def load(path: str) -> "RelEncoderCheckpoint":
        """
        This loads a checkpoint from a specific path.
        Args:
            path (str): The path of a checkpoint to load.

        Returns:
            Checkpoint: The loaded checkpoint.
        """
        return torch.load(path, map_location=torch.device("cpu"))

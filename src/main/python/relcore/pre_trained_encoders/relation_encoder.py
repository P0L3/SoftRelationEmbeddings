# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#        Copyright (c) -2024 - Mtumbuka F.                                                    #
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
__version__ = "2024.1"
__date__ = "19 Dec, 2024."
__author__ = "Frank M. Mtumbuka"
__maintainer__ = "Frank M. Mtumbuka"
__email__ = ""
__status__ = "Development"


import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoTokenizer, AutoModel


class RelationEncoder(nn.Module, PyTorchModelHubMixin):
    """This creates a relation encoder based on the specified experiment configurations"""

    SPECIAL_TOKENS = ["[E1]", "[/E1]", "[E2]", "[/E2]"]
    """list: The special tokens to mark the start and end of the head and tail entities."""

    def __init__(self, base_model: str):
        super().__init__()
        # Create tokenizer.
        self._tokenizer = AutoTokenizer.from_pretrained(base_model)

        #Add special tokens
        self._tokenizer.add_special_tokens({"additional_special_tokens": self.SPECIAL_TOKENS})

        # Create encoder
        self._encoder = AutoModel.from_pretrained(base_model)

        # Resize the embedding size of the model to accommodate special tokens.
        self._encoder.resize_token_embeddings(len(self._tokenizer))

    @property
    def end_of_head_entity(self) -> str:
        """str: The special token indicating the end of the head entity."""

        return "[/E1]"

    @property
    def end_of_tail_entity(self) -> str:
        """str: The special token indicating the end of the tail entity."""

        return "[/E2]"

    @property
    def tokenizer(self):
        """Specifies the tokenizer being used together with the pre-trained relation encoder."""

        return self._tokenizer

    @property
    def start_of_head_entity(self) -> str:
        """str: The special token indicating the start of the head entity."""

        return "[E1]"

    @property
    def start_of_tail_entity(self) -> str:
        """str: The special token indicating the start of the tail entity."""

        return "[E2]"

    @property
    def mask_token(self) -> str:
        """str: The special token indicating the mask token in the input sentence."""

        return self._tokenizer.mask_token

    def forward(self, input_seq):
        encoder_outputs = self._encoder(**input_seq, output_hidden_states=True)
        return encoder_outputs.hidden_states[12]

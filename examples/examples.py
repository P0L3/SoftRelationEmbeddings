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

"""
This file demonstrates on how to use the pre-trained relation encoders.
"""

# Import the necessary libraries and classes

"""These are local classes in src/main/python/relcore/pre_trained_encoders/relation_encoder"""
import relcore.pre_trained_encoders.relation_encoder as relation_enc



"""To load the pre-trained relation encoder based on bert"""
model = relation_enc.RelationEncoder.from_pretrained("fmmka/rel-emb-bert-b-uncased")

"""To load the pre-trained relation encoder based on albert."""
model = relation_enc.RelationEncoder.from_pretrained("fmmka/rel-emb-albert")

"""To load the pre-trained encore model based on roberta."""
model = relation_enc.RelationEncoder.from_pretrained("fmmka/rel-emb-roberta-large")


"""Get the tokenizer from the model."""
tokenizer = model.tokenizer

"""Having loaded the tokenizer and pretrained model, below are the steps on how to encode text."""
sentence = ("The Olympics will take place in {} Paris {}, the capital of {} France {}. The relation between Paris "
                "and France is{}.").format(
        model.start_of_head_entity, model.end_of_head_entity, model.start_of_tail_entity, model.end_of_tail_entity, model.mask_token
)
inputs = tokenizer(sentence, return_tensors="pt")

# Tokenized input
tokens = tokenizer.tokenize(sentence)
# Get the token indices for [E1] and [E2]
e1_index = tokens.index(model.start_of_head_entity)  # Index of the [E1] token
e2_index = tokens.index(model.start_of_tail_entity)  # Index of the [E2] token
mask_index = tokens.index(model.mask_token) # Index of the "[MASK]" token

outputs = model(inputs)
# Extract embeddings for [E1] and [/E2]
e1_embedding = outputs[:, e1_index, :]  # Embedding for [E1]
e2_embedding = outputs[:, e2_index, :]  # Embedding for [/E2]
mask_embedding = outputs[:, mask_index, :] # Embedding for [MASK]



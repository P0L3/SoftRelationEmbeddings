# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#        Copyright (c) -2022 - Mtumbuka F.                                                    #
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
__version__ = "2022.1"
__date__ = "28 Jul 2022"
__author__ = "Frank M. Mtumbuka"
__maintainer__ = "Frank M. Mtumbuka"
__email__ = "" ""
__status__ = "Development"

import argmagic.decorators as decorators
import expbase as xb
import insanity
import os
import typing
import torch


class Config(xb.BaseConfig):
    """This class has all the configurations for an experiment."""

    DEFAULT_ACE_DATASET = False
    """bool: The default setting for loading the ACE2005 dataset."""

    DEFAULT_MY_CUSTOM_DATASET = False
    """bool: The default setting for using the custom weighted dataset."""
    
    DEFAULT_ACE_FINE_GRAINED = False
    """bool: The default setting for using fine-grained relation types or  not."""

    DEFAULT_ALBERT_ENC_REL = False
    """bool: The default setting for using an ALBERT based encoder for relations."""

    DEFAULT_BATCH_SIZE = 256
    """int: The default value for batch size."""

    DEFAULT_BERT_ENC_REL = False
    """bool: The default setting for using the BERT based encoder for relations."""

    DEFAULT_CLASSIFIER_LAYERS = 1
    """int: The default number of layers in the classifiers."""

    DEFAULT_CP_LAYER = False
    """bool: The default setting for using the contrastive loss intermediate layer."""

    DEFAULT_CP_LOSS_TAU = 0.07
    """float: The default value for the contrastive loss temperature."""

    DEFAULT_DATA_DIR = "/home/frankmtumbuka/Projects/RelCore/data/"
    """str: The default value for the data directory."""

    DEFAULT_DEV_PART = False
    """bool: The default setting for loading data from the dev partition."""

    DEFAULT_DROPOUT_RATE = 0.1
    """float: The dropout rate to use (across all components that use dropout)."""

    DEFAULT_EVAL_BATCH_SIZE = 1
    """int: The default batch size during evaluation."""

    DEFAULT_EVAL_GPU = False
    """bool: The default setting for running model evaluation on gpu."""

    DEFAULT_FINETUNE = False
    """bool: The default setting for fine-tuning the model."""

    DEFAULT_FREEZE_REL_ENC = False
    """bool: The default setting for freezing the relation encoder during the experiment."""

    DEFAULT_GPU = False
    """bool: The default setting for using GPU when training."""

    DEFAULT_GRAD_ACC_ITERS = 1
    """int: The default number of iterations to accumulate gradients over."""

    DEFAULT_HT = False
    """bool: The default setting for using head and tail embeddings to predict relation type."""

    DEFAULT_LEARNING_RATE = 0.00005
    """float: The default value for the model learning rate."""

    DEFAULT_MAX_GRAD_NORM = 2.0
    """float: The default maximum norm that any gradients are clipped to."""

    DEFAULT_MLM_PERCENTAGE = 0.15
    """float: The default value for masking percentage when performing MLM."""

    DEFAULT_MLP_CLASSIFIER = False
    """bool: The default setting for using the mlp classifier."""

    DEFAULT_NUM_EPOCHS = 1
    """int: The default value for the number of training epochs."""

    DEFAULT_NYT_DATASET = False
    """bool: The default setting for using the NYT dataset."""

    DEFAULT_PICKLE_DATA = False
    """bool: The default setting for pickling data or not."""

    DEFAULT_PRETRAIN = False
    """bool: The default setting for pre-training the encoder."""

    DEFAULT_PRINT_INT = 1
    """int: The default value for logging details when running experiments."""

    DEFAULT_PRINT_OUTPUT = False
    """bool: The default value of saving outputs during evaluation."""

    DEFAULT_RELATION_MASK = False
    """bool: The default setting for using the mask token to encode relations or not."""

    DEFAULT_RELATION_PROMPT = False
    """bool: The default setting for using a relation prompt to get the relation representation."""

    DEFAULT_REL_MASK = False
    """bool: The default setting for using the relation mask representation."""

    DEFAULT_RETACRED_DATASET = False
    """bool: The default setting for loading the RE-TACRED dataset."""

    DEFAULT_ROBERTA_ENC_REL = False
    """bool: The default setting for using a RoBERTa based encoder for relations."""

    DEFAULT_SENTENCES = False
    """bool: The default setting for loading individual sentences rather than stories."""

    DEFAULT_SORTED_PK = False
    """bool: The default for sorting predictions by their probability when computing precision at k."""

    DEFAULT_TACRED_DATASET = False
    """bool: The default setting for loading the TACRED dataset."""

    DEFAULT_TACREV_DATASET = False
    """bool: The default setting for loading the TACREV dataset."""

    DEFAULT_WIKIDATA_DATASET = False
    """bool: The default setting for loading the Wikidata-Wikipedia dataset."""

    def __init__(self):
        super().__init__()
        self._ace_dataset = self.DEFAULT_ACE_DATASET
        self._ace_fine_grained = self.DEFAULT_ACE_FINE_GRAINED
        self._albert_enc_rel = self.DEFAULT_ALBERT_ENC_REL
        self._batch_size = self.DEFAULT_BATCH_SIZE
        self._bert_enc_rel = self.DEFAULT_BERT_ENC_REL
        self._classifier_layers = self.DEFAULT_CLASSIFIER_LAYERS
        self._cp_layer = self.DEFAULT_CP_LAYER
        self._cp_loss_tau = self.DEFAULT_CP_LOSS_TAU
        self._data_dir = self.DEFAULT_DATA_DIR
        self._dev_part = self.DEFAULT_DEV_PART
        self._dropout_rate = self.DEFAULT_DROPOUT_RATE
        self._eval_batch_size = self.DEFAULT_EVAL_BATCH_SIZE
        self._eval_gpu = self.DEFAULT_EVAL_GPU
        self._finetune = self.DEFAULT_FINETUNE
        self._freeze_rel_enc = self.DEFAULT_FREEZE_REL_ENC
        self._gpu = self.DEFAULT_GPU
        self._grad_acc_iters = self.DEFAULT_GRAD_ACC_ITERS
        self._ht = self.DEFAULT_HT
        self._learning_rate = self.DEFAULT_LEARNING_RATE
        self._num_epochs = self.DEFAULT_NUM_EPOCHS
        self._max_grad_norm = self.DEFAULT_MAX_GRAD_NORM
        self._mlm_percentage = self.DEFAULT_MLM_PERCENTAGE
        self._mlp_classifier = self.DEFAULT_MLP_CLASSIFIER
        self._nyt_dataset = self.DEFAULT_NYT_DATASET
        self._pickle_data = self.DEFAULT_PICKLE_DATA
        self._pretrain = self.DEFAULT_PRETRAIN
        self._print_int = self.DEFAULT_PRINT_INT
        self._print_output = self.DEFAULT_PRINT_OUTPUT
        self._relation_mask = self.DEFAULT_RELATION_MASK
        self._relation_prompt = self.DEFAULT_RELATION_PROMPT
        self._rel_mask = self.DEFAULT_REL_MASK
        self._retacred_dataset = self.DEFAULT_RETACRED_DATASET
        self._roberta_enc_rel = self.DEFAULT_ROBERTA_ENC_REL
        self._sentences = self.DEFAULT_SENTENCES
        self._sorted_pk = self.DEFAULT_SORTED_PK
        self._tacred_dataset = self.DEFAULT_TACRED_DATASET
        self._tacrev_dataset = self.DEFAULT_TACREV_DATASET
        self._wikidata_dataset = self.DEFAULT_WIKIDATA_DATASET
        self._checkpoint = None
        self._checkpoints_dir = None
        self._num_classes = 1
        
        self._my_custom_dataset = self.DEFAULT_MY_CUSTOM_DATASET

    # Properties
    @decorators.optional
    @property
    def ace_dataset(self) -> bool:
        """bool: Specifies whether to load the ACE2005 dataset or not."""
        return self._ace_dataset

    @ace_dataset.setter
    def ace_dataset(self, ace_dataset: bool) -> None:
        self._ace_dataset = bool(ace_dataset)

    @property
    def ace_fine_grained(self) -> bool:
        """bool: Specifies whether to load fine-grained ACE or not."""
        return self._ace_fine_grained

    @ace_fine_grained.setter
    def ace_fine_grained(self, ace_fine_grained: bool) -> None:
        self._ace_fine_grained = bool(ace_fine_grained)

    @property
    def albert_enc_rel(self) -> bool:
        """bool: Specifies whether to use an ALBERT based encoder or not for relations."""
        return self._albert_enc_rel

    @albert_enc_rel.setter
    def albert_enc_rel(self, albert_enc_rel: bool) -> None:
        self._albert_enc_rel = bool(albert_enc_rel)

    @property
    def batch_size(self) -> int:
        """int: Specifies the batch size used during training."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int) -> None:
        insanity.sanitize_type("batch_size", batch_size, int)
        insanity.sanitize_range("batch_size", batch_size, minimum=1)
        self._batch_size = batch_size

    @property
    def bert_enc_rel(self) -> bool:
        """bool: Specifies whether to use BERT based encoder or not for relations."""
        return self._bert_enc_rel

    @bert_enc_rel.setter
    def bert_enc_rel(self, bert_enc_rel: bool) -> None:
        self._bert_enc_rel = bool(bert_enc_rel)

    @decorators.optional
    @property
    def checkpoint(self) -> typing.Optional[str]:
        """str: The path of a checkpoint to load the model state from."""
        return self._checkpoint

    @checkpoint.setter
    def checkpoint(self, checkpoint: str) -> None:
        checkpoint = str(checkpoint)
        if not os.path.isfile(checkpoint):
            raise ValueError("The provided <checkpoint> does not refer to an existing file: '{}'".format(checkpoint))
        self._checkpoint = checkpoint

    @decorators.optional
    @property
    def checkpoints_dir(self) -> typing.Union[None, str]:
        """Specifies the directory for checkpoints to be evaluated."""
        return self._checkpoints_dir

    @checkpoints_dir.setter
    def checkpoints_dir(self, checkpoints_dir: str) -> None:
        checkpoints_dir = str(checkpoints_dir)
        if not os.path.isdir(checkpoints_dir):
            raise ValueError(
                "The specified <checkpoint_dir> is not a directory on the system: {}".format(checkpoints_dir)
            )
        self._checkpoints_dir = checkpoints_dir

    @property
    def classifier_layers(self) -> int:
        """int: Specifies the number of layers in the classifier."""
        return self._classifier_layers

    @classifier_layers.setter
    def classifier_layers(self, classifier_layers: int) -> None:
        insanity.sanitize_type("classifier_layers", classifier_layers, int)
        insanity.sanitize_range("classifier_layers", classifier_layers, minimum=1)
        self._classifier_layers = int(classifier_layers)

    @property
    def cp_layer(self) -> bool:
        """bool: Specifies whether to use the contrastive loss intermediate layer or not."""
        return self._cp_layer

    @cp_layer.setter
    def cp_layer(self, cp_layer: bool) -> None:
        self._cp_layer = bool(cp_layer)

    @property
    def cp_loss_tau(self) -> float:
        """float: Specifies the tau in the contrastive loss equation."""
        return self._cp_loss_tau

    @cp_loss_tau.setter
    def cp_loss_tau(self, cp_loss_tau: float) -> None:
        insanity.sanitize_type("cp_loss_tau", cp_loss_tau, float)
        insanity.sanitize_range("cp_loss_tau", cp_loss_tau, minimum=0.0)
        self._cp_loss_tau = cp_loss_tau

    @property
    def data_dir(self) -> str:
        """str: Specifies the path to the data directory."""
        return self._data_dir

    @data_dir.setter
    def data_dir(self, data_dir: str) -> None:
        # TODO sanitize
        self._data_dir = str(data_dir)

    @property
    def dev_part(self) -> bool:
        """bool: Specifies whether to load data from the dev partition or not."""
        return self._dev_part

    @dev_part.setter
    def dev_part(self, dev_part: bool) -> None:
        self._dev_part = bool(dev_part)

    @property
    def dropout_rate(self) -> float:
        """float: The dropout rate to use (across all components that use dropout)."""
        return self._dropout_rate

    @dropout_rate.setter
    def dropout_rate(self, dropout_rate: float) -> None:
        insanity.sanitize_type("dropout_rate", dropout_rate, float)
        dropout_rate = float(dropout_rate)
        insanity.sanitize_range("dropout_rate", dropout_rate, minimum=0, maximum=1, max_inclusive=False)
        self._dropout_rate = dropout_rate

    @property
    def eval_batch_size(self) -> int:
        """int: Specifies the batch size used during evaluation."""
        return self._eval_batch_size

    @eval_batch_size.setter
    def eval_batch_size(self, eval_batch_size: int) -> None:
        insanity.sanitize_type("eval_batch_size", eval_batch_size, int)
        insanity.sanitize_range("eval_batch_size", eval_batch_size, minimum=1)
        self._eval_batch_size = eval_batch_size

    @property
    def eval_gpu(self) -> bool:
        """bool: Specifies whether to use GPU during model evaluation or not."""
        return self._eval_gpu

    @eval_gpu.setter
    def eval_gpu(self, eval_gpu: bool) -> None:
        if eval_gpu and not torch.cuda.is_available():
            raise ValueError("There is no GPU on the local machine.")
        self._eval_gpu = bool(eval_gpu)

    @property
    def finetune(self) -> bool:
        """bool: Specifies whether to finetune the model or not."""
        return self._finetune

    @finetune.setter
    def finetune(self, finetune: bool) -> None:
        self._finetune = bool(finetune)

    @property
    def freeze_rel_enc(self) -> bool:
        """bool: Specifies whether to freeze the relation encoder during the experiment or not."""
        return self._freeze_rel_enc

    @freeze_rel_enc.setter
    def freeze_rel_enc(self, freeze_rel_enc: bool) -> None:
        self._freeze_rel_enc = bool(freeze_rel_enc)

    @property
    def gpu(self) -> bool:
        """bool: Specifies whether to use GPU during model training or not."""
        return self._gpu

    @gpu.setter
    def gpu(self, gpu: bool) -> None:
        if gpu and not torch.cuda.is_available():
            raise ValueError("There is no GPU on the local machine.")
        self._gpu = bool(gpu)

    @property
    def grad_acc_iters(self) -> int:
        """int: The number of iterations to accumulate gradients over."""
        return self._grad_acc_iters

    @grad_acc_iters.setter
    def grad_acc_iters(self, grad_acc_iters: int) -> None:
        insanity.sanitize_type("grad_acc_iters", grad_acc_iters, int)
        insanity.sanitize_range("grad_acc_iters", grad_acc_iters, minimum=1)
        self._grad_acc_iters = int(grad_acc_iters)

    @property
    def ht(self) -> bool:
        """bool: Specifies whether to consider head and tail entity embeddings of relation classification or not"""
        return self._ht

    @ht.setter
    def ht(self, ht: bool) -> None:
        self._ht = bool(ht)

    @property
    def learning_rate(self) -> float:
        """float: Specifies the learning rate for the model during training."""
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate: float) -> None:
        self._learning_rate = float(learning_rate)

    @decorators.optional
    @property
    def max_grad_norm(self) -> float:
        """float: The maximum norm that any gradients are clipped to."""
        return self._max_grad_norm

    @max_grad_norm.setter
    def max_grad_norm(self, max_grad_norm: float) -> None:
        self._max_grad_norm = float(max_grad_norm)

    @property
    def mlm_percentage(self) -> float:
        """float: Specifies the percentage of tokens to be masked out in a given batch."""
        return self._mlm_percentage

    @mlm_percentage.setter
    def mlm_percentage(self, mlm_percentage: float) -> None:
        insanity.sanitize_type("mlm_percentage", mlm_percentage, float)
        insanity.sanitize_range("mlm_percentage", mlm_percentage, minimum=0.00)
        insanity.sanitize_range("mlm_percentage", mlm_percentage, maximum=1.0)
        self._mlm_percentage = float(mlm_percentage)

    @property
    def mlp_classifier(self) -> bool:
        """bool: Specifies whether to use an MLP classifier or not"""
        return self._mlp_classifier
    
    @property
    def my_custom_dataset(self) -> bool:
        """bool: Specifies whether to use my custom weighted dataset."""
        return self._my_custom_dataset

    @my_custom_dataset.setter
    def my_custom_dataset(self, my_custom_dataset: bool) -> None:
        self._my_custom_dataset = bool(my_custom_dataset)

    @mlp_classifier.setter
    def mlp_classifier(self, mlp_classifier: bool) -> None:
        self._mlp_classifier = bool(mlp_classifier)

    @decorators.optional
    @property
    def num_classes(self) -> int:
        """int: Specifies the number of classification classes."""
        return self._num_classes

    @num_classes.setter
    def num_classes(self, num_classes: int) -> None:
        insanity.sanitize_type("num_classes", num_classes, int)
        insanity.sanitize_range("num_classes", num_classes, minimum=1)
        self._num_classes = int(num_classes)

    @property
    def num_epochs(self) -> int:
        """int: Specifies the number of training epochs."""
        return self._num_epochs

    @num_epochs.setter
    def num_epochs(self, num_epochs: int) -> None:
        insanity.sanitize_type("num_epochs", num_epochs, int)
        insanity.sanitize_range("num_epochs", num_epochs, minimum=1)
        self._num_epochs = int(num_epochs)

    @property
    def nyt_dataset(self) -> bool:
        """bool: Specifies whether to use the NYT dataset or not."""
        return self._nyt_dataset

    @nyt_dataset.setter
    def nyt_dataset(self, nyt_dataset: bool) -> None:
        self._nyt_dataset = bool(nyt_dataset)

    @property
    def pickle_data(self) -> bool:
        """bool: Specifies whether to pickle the dataset or not."""
        return self._pickle_data

    @pickle_data.setter
    def pickle_data(self, pickle_data: bool) -> None:
        self._pickle_data = bool(pickle_data)

    @property
    def pretrain(self) -> bool:
        """bool: Specifies whether to pretrain the encoder or not."""
        return self._pretrain

    @pretrain.setter
    def pretrain(self, pretrain: bool) -> None:
        self._pretrain = bool(pretrain)

    @property
    def print_int(self) -> int:
        """int: Specifies the logging interval for experiment details during the experiment."""
        return self._print_int

    @print_int.setter
    def print_int(self, print_int) -> None:
        self._print_int = print_int

    @property
    def print_output(self) -> bool:
        """bool: The default setting for printing out the output."""
        return self._print_output

    @print_output.setter
    def print_output(self, print_output: bool) -> None:
        self._print_output = bool(print_output)

    @property
    def relation_mask(self) -> bool:
        """bool: Specifies whether to use the mask token to encode the relation or not."""
        return self._relation_mask

    @relation_mask.setter
    def relation_mask(self, relation_mask: bool) -> None:
        self._relation_mask = bool(relation_mask)

    @property
    def relation_prompt(self) -> bool:
        """bool: Specifies whether to use the relation prompt to represent the relation representation."""
        return self._relation_prompt

    @relation_prompt.setter
    def relation_prompt(self, relation_prompt: bool) -> None:
        self._relation_prompt = bool(relation_prompt)

    @property
    def rel_mask(self) -> bool:
        """bool: Specifies whether to use the relation prompt mask for relation prediction or not."""
        return self._rel_mask

    @rel_mask.setter
    def rel_mask(self, rel_mask: bool) -> None:
        self._rel_mask = bool(rel_mask)

    @property
    def retacred_dataset(self) -> bool:
        """bool: Specifies whether to load the Re-Tacred dataset or not."""
        return self._retacred_dataset

    @retacred_dataset.setter
    def retacred_dataset(self, retacred_dataset: bool) -> None:
        self._retacred_dataset = bool(retacred_dataset)

    @property
    def roberta_enc_rel(self) -> bool:
        """bool: Specifies whether to use a RoBERTa based encoder or not for relations."""
        return self._roberta_enc_rel

    @roberta_enc_rel.setter
    def roberta_enc_rel(self, roberta_enc_rel: bool) -> None:
        self._roberta_enc_rel = bool(roberta_enc_rel)

    @property
    def sentences(self) -> bool:
        """bool: Specifies whether to load data as individual sentences or not."""
        return self._sentences

    @sentences.setter
    def sentences(self, sentences: bool) -> None:
        self._sentences = bool(sentences)

    @property
    def sorted_pk(self) -> bool:
        """bool: Specifies whether to sort predictions by their probability when computing p@k."""
        return self._sorted_pk

    @sorted_pk.setter
    def sorted_pk(self, sorted_pk: bool) -> None:
        self._sorted_pk = bool(sorted_pk)

    @property
    def tacred_dataset(self) -> bool:
        """bool: Specifies whether to load the TACRED dataset or not."""
        return self._tacred_dataset

    @tacred_dataset.setter
    def tacred_dataset(self, tacred_dataset: bool) -> None:
        self._tacred_dataset = bool(tacred_dataset)

    @property
    def tacrev_dataset(self) -> bool:
        """bool: Specifies whether to load the Tacrev dataset or not."""
        return self._tacrev_dataset

    @tacrev_dataset.setter
    def tacrev_dataset(self, tacrev_dataset: bool) -> None:
        self._tacrev_dataset = bool(tacrev_dataset)

    @property
    def wikidata_dataset(self) -> bool:
        """bool: Specifies whether to use the wikidata-wikipedia dataset."""
        return self._wikidata_dataset

    @wikidata_dataset.setter
    def wikidata_dataset(self, wikidata_dataset: bool) -> None:
        self._wikidata_dataset = bool(wikidata_dataset)

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Encoder model wrappers based on HuggingFace code
"""

import logging
import typing as t
from typing import Tuple, List

import torch
from torch import nn
from torch.optim import AdamW
from transformers import BertTokenizer
from transformers import DPRConfig
from transformers import DPRContextEncoder
from transformers import DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer, DPRQuestionEncoder

from dpr.models.biencoder import BiEncoder
from dpr.models.hf_models import BertTensorizer

logger = logging.getLogger(__name__)


class DPRTensorizer(BertTensorizer):
    def __init__(self, tokenizer: t.Union[DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer], max_length: int, pad_to_max: bool = True):
        super().__init__(tokenizer=tokenizer, max_length=max_length, pad_to_max=pad_to_max)


def get_dpr_biencoder_components(cfg, inference_only: bool = False, **kwargs) \
        -> Tuple[
            DPRTensorizer,
            DPRTensorizer,
            BiEncoder,
            t.Optional[torch.optim.Optimizer]]:
    dropout = cfg.encoder.dropout if hasattr(cfg.encoder, "dropout") else 0.0
    q_encoder_cfg = DPRConfig.from_pretrained(cfg.encoder.q_pretrained_model_cfg,
                                              projection_dim=cfg.encoder.projection_dim,
                                              hidden_dropout_prob=dropout,
                                              attention_probs_dropout_prob=dropout)
    q_encoder = DPRQuestionEncoder.from_pretrained(pretrained_model_name_or_path=cfg.encoder.q_pretrained_model_cfg,
                                                   config=q_encoder_cfg, **kwargs)

    ctx_encoder_cfg = DPRConfig.from_pretrained(cfg.encoder.ctx_pretrained_model_cfg,
                                                projection_dim=cfg.encoder.projection_dim,
                                                hidden_dropout_prob=dropout,
                                                attention_probs_dropout_prob=dropout)
    ctx_encoder = DPRContextEncoder.from_pretrained(pretrained_model_name_or_path=cfg.encoder.ctx_pretrained_model_cfg,
                                                    config=ctx_encoder_cfg, **kwargs)

    fix_ctx_encoder = cfg.encoder.fix_ctx_encoder if hasattr(cfg.encoder, "fix_ctx_encoder") else False
    fix_q_encoder = cfg.encoder.fix_q_encoder if hasattr(cfg.encoder, "fix_q_encoder") else False
    biencoder = BiEncoder(q_encoder, ctx_encoder, fix_q_encoder=fix_q_encoder, fix_ctx_encoder=fix_ctx_encoder)
    optimizer = (
        get_optimizer(
            biencoder,
            learning_rate=cfg.train.learning_rate,
            adam_eps=cfg.train.adam_eps,
            weight_decay=cfg.train.weight_decay,
        )
        if not inference_only
        else None
    )
    ctx_tensorizer, question_tensorizer = get_dpr_tensorizers(cfg=cfg)
    return ctx_tensorizer, question_tensorizer, biencoder, optimizer


def get_dpr_tensorizers(cfg: DPRConfig) \
        -> Tuple[
            DPRTensorizer,
            DPRTensorizer]:
    sequence_length = cfg.encoder.sequence_length
    ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(cfg.encoder.ctx_pretrained_model_cfg, do_lower_case=cfg.do_lower_case)
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(cfg.encoder.q_pretrained_model_cfg, do_lower_case=cfg.do_lower_case)

    if cfg.special_tokens:
        _add_special_tokens(ctx_tokenizer, cfg.special_tokens)
        _add_special_tokens(question_tokenizer, cfg.special_tokens)

    return DPRTensorizer(ctx_tokenizer, sequence_length), DPRTensorizer(question_tokenizer, sequence_length)


def _add_special_tokens(tokenizer, special_tokens):
    logger.info("Adding special tokens %s", special_tokens)
    logger.info("Tokenizer: %s", type(tokenizer))
    special_tokens_num = len(special_tokens)
    # TODO: this is a hack-y logic that uses some private tokenizer structure which can be changed in HF code

    assert special_tokens_num < 500
    unused_ids = [tokenizer.vocab["[unused{}]".format(i)] for i in range(special_tokens_num)]
    logger.info("Utilizing the following unused token ids %s", unused_ids)

    for idx, id in enumerate(unused_ids):
        old_token = "[unused{}]".format(idx)
        del tokenizer.vocab[old_token]
        new_token = special_tokens[idx]
        tokenizer.vocab[new_token] = id
        tokenizer.ids_to_tokens[id] = new_token
        logging.debug("new token %s id=%s", new_token, id)

    tokenizer.additional_special_tokens = list(special_tokens)
    logger.info("additional_special_tokens %s", tokenizer.additional_special_tokens)
    logger.info("all_special_tokens_extended: %s", tokenizer.all_special_tokens_extended)
    logger.info("additional_special_tokens_ids: %s", tokenizer.additional_special_tokens_ids)
    logger.info("all_special_tokens %s", tokenizer.all_special_tokens)


def get_optimizer(
        model: nn.Module,
        learning_rate: float = 1e-5,
        adam_eps: float = 1e-8,
        weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    print(model)
    optimizer_grouped_parameters = get_hf_model_param_grouping(model, weight_decay)
    return get_optimizer_grouped(optimizer_grouped_parameters, learning_rate, adam_eps)


def get_hf_model_param_grouping(
        model: nn.Module,
        weight_decay: float = 0.0,
):
    no_decay = ["bias", "LayerNorm.weight"]

    return [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]


def get_optimizer_grouped(
        optimizer_grouped_parameters: List,
        learning_rate: float = 1e-5,
        adam_eps: float = 1e-8,
) -> torch.optim.Optimizer:
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps)
    return optimizer


def get_bert_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    return BertTokenizer.from_pretrained(pretrained_cfg_name, do_lower_case=do_lower_case)

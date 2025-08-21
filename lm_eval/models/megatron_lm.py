# Implementation is based on the one from:
# https://github.com/yuzc19/lm-evaluation-harness/blob/0c8c0d8165f4caa937c149aa92693e7ac3bfaece/lm_eval/models/megatron_lm.py

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import importlib
import pathlib
from copy import deepcopy
from typing import List, Literal
import filelock
import numpy as np
import torch
from tqdm import tqdm
from enum import Enum, auto
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import Collator
from lm_eval.utils import (
    get_rolling_token_windows,
    make_disjoint_window,
    simple_parse_args_string,
)

import os
import torch
import torch.nn.functional as F
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.training.checkpointing import get_rng_state, load_checkpoint, _load_base_checkpoint
from megatron.training.utils import print_rank_0

from megatron.training.initialize import initialize_megatron
from megatron.training import get_args, get_tokenizer, get_model
from megatron.core import mpu, tensor_parallel, dist_checkpointing, parallel_state
from megatron.training.arguments import parse_args, validate_args
from megatron.training.global_vars import set_global_variables
from megatron.training.initialize import (
    setup_logging,
    _set_random_seed,
    _initialize_distributed,
    _init_autoresume,
    _initialize_tp_communicators,
    _compile_dependencies,
)
from megatron.training.arguments import core_transformer_config_from_args, get_patch_args
from megatron.inference.text_generation.api import generate_and_post_process
from gpt_model_provider_for_inference import model_provider


@register_model("megatron_lm")
class MegatronLM(LM):
    def __init__(
        self,
        batch_size: int = 1,
        max_gen_toks: int = 256,
        **kwargs,
    ):
        
        super().__init__()

        initialize_megatron(args_defaults={'no_load_rng': True, 'no_load_optim': True, 'exit_on_missing_checkpoint': False}, ignore_unknown_args=True)
        # initialize_megatron(args_defaults={'no_load_rng': True, 'no_load_optim': True, 'exit_on_missing_checkpoint': True}, ignore_unknown_args=True)
        GPUS_PER_NODE = 8 # This can vary between systems. Could be read from environment variables e.g. os.environ["SLURM_GPUS_PER_NODE"]
        model = get_model(model_provider, wrap_with_ddp=False)
        load_checkpoint(model, None, None)
        model = model[0]
        model.eval()
        self.add_special_tokens = False

        self.args = get_args()
        print(self.args)
        self.model = model
        self._max_length = self.args.max_position_embeddings
        self._batch_size = int(batch_size)
        self._max_gen_toks = max_gen_toks

        self.is_pipeline_first_stage = mpu.is_pipeline_first_stage()

        self._world_size = self.args.world_size 
        model_parallel_size = self.args.tensor_model_parallel_size * self.args.pipeline_model_parallel_size * self.args.context_parallel_size
        self._rank = torch.distributed.get_rank()
        self.data_parallel_group = parallel_state.get_data_parallel_group()
        self.is_model_parallel_src_rank = parallel_state.get_model_parallel_src_rank() == self._rank
        # DP "source" convenience: use MP-src within each DP replica.
        self.is_data_parallel_src_rank = self.is_model_parallel_src_rank
        self._data_parallel_rank = self._rank // model_parallel_size
        self._device = torch.device(f"cuda:{torch.distributed.get_rank() % GPUS_PER_NODE}")

        self.tokenizer = get_tokenizer()

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        args = simple_parse_args_string(arg_string)
        if additional_config:
            args["batch_size"] = additional_config.get("batch_size", 1)

        return cls(**args)

    @property
    def eot_token_id(self):
        try:
            return self.args.eos_id
        except AttributeError:
            return None

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device
    
    @property
    def dp_rank(self):
        return self._data_parallel_rank
    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def accelerator(self):
        return self._Accelerator(self.world_size)

    class _Accelerator:
        def __init__(self, world_size):
            self.world_size = world_size

        def wait_for_everyone(self):
            torch.distributed.barrier()

        def gather(self, local_tensor):
            gathered_tensors = [
                torch.zeros(1, dtype=local_tensor.dtype).cuda()
                for _ in range(self.world_size)
            ]
            torch.distributed.all_gather(gathered_tensors, local_tensor)
            return torch.cat(gathered_tensors)

    def tok_encode(self, string: str):
        return self.tokenizer.tokenize(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.detokenize(tokens)

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole_enc = self.tok_encode(context + continuation)
        context_enc = self.tok_encode(context)
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        new_reqs = []
        for context, continuation in [req.args for req in requests]:
            # print("Before encoding:", context)
            if context == "":
                # end of text as context
                context_enc, continuation_enc = (
                    [self.eot_token_id],
                    self.tok_encode(continuation),
                )
                # print("After encoding -- if-branch:", context_enc, continuation_enc)
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)
                # print("After encoding -- else:", context_enc, continuation_enc)

            # assert False
            new_reqs.append(((context, continuation), context_enc, continuation_enc))
        return self._loglikelihood_tokens(new_reqs)

    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        loglikelihoods = []
        print("fn: loglikelihood_rolling: requests:", requests)
        for (string,) in tqdm([req.args for req in requests], disable=disable_tqdm):
            rolling_token_windows = list(
                map(
                    make_disjoint_window,
                    get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.eot_token_id,
                        max_seq_len=self.max_length - 1,
                        context_len=1,
                    ),
                )
            )

            rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            string_nll = self._loglikelihood_tokens(
                rolling_token_windows,
            )

            # discard is_greedy
            string_nll = [x[0] for x in string_nll]

            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)

            # cache this loglikelihood_rolling request
            self.cache_hook.add_partial("loglikelihood_rolling", (string,), string_nll)
        return loglikelihoods

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        # Do not early-return; all PP ranks must enter Megatron generate path.
        res = []

        def _collate(x):
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        re_ord = Collator(requests, sort_fn=_collate)
        chunks = re_ord.get_batched(n=self.batch_size, batch_fn=None)
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self._rank != 0)),
            desc="Running loglikelihood requests",
        )
        for chunk in chunks:
            inps = []
            ctxlens = []
            contlens = []
            for _, context_enc, continuation_enc in chunk:
                # Leave one token for generation. Tokens_to_generate = 0 breaks NeMo.
                inp = (context_enc + continuation_enc)[-(self.max_length) :]

                ctxlen = len(context_enc) - max(
                    0, len(context_enc) + len(continuation_enc) - (self.max_length)
                )
                ctxlens.append(ctxlen)
                contlens.append(len(continuation_enc))

                inps.append(self.tok_decode(inp))
            outputs = generate_and_post_process(
                model=self.model,
                prompts=inps,
                tokens_to_generate=0,
                add_BOS=False,
                return_output_log_probs=True,
                return_topk_logprobs=1,
                data_parallel=True,
            )
            # Only PP first stage receives outputs; other stages return None.
            if self.is_pipeline_first_stage:
                (
                    batch_output,
                    batch_tokens,
                    batch_logprobs,
                    batch_token_ids,
                    logprobs_topk,
                ) = outputs

                batch_token_ids = np.asarray(batch_token_ids)
                min_ctxlen = min(ctxlens)
                batch_greedy_tokens = (
                    logprobs_topk.indices[:, min_ctxlen - 1:, 0].cpu().numpy()
                )

                # append per-sample answers to res
                for (
                    token_ids,
                    greedy_tokens,
                    logprobs,
                    ctxlen,
                    contlen,
                    (cache_key, _, _),
                ) in zip(
                    np.asarray(batch_token_ids),
                    logprobs_topk.indices[:, min(ctxlens) - 1:, 0].cpu().numpy(),
                    batch_logprobs,
                    ctxlens,
                    contlens,
                    chunk,
                ):
                    logprobs = (logprobs[ctxlen - 1:])[:contlen]
                    logprob = sum(logprobs)

                    continuation_tokens = (token_ids[ctxlen:])[:contlen]
                    len_diff = ctxlen - min_ctxlen
                    min_len = min(len(continuation_tokens), len(greedy_tokens[len_diff:]))
                    is_greedy = continuation_tokens[:min_len] == (greedy_tokens[len_diff:])[:min_len]
                    if not isinstance(is_greedy, bool):
                        is_greedy = is_greedy.all()
                    answer = (logprob, is_greedy)

                    if cache_key is not None:
                        self.cache_hook.add_partial("loglikelihood", cache_key, answer)

                    res.append(answer)

            pbar.update(len(chunk))
        pbar.close()
        if mpu.is_pipeline_first_stage():
            return re_ord.get_original(res)
        else:
            return None

    def generate_until(self, requests):
        assert False, "Not implemented"
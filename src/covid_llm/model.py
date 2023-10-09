import copy
import logging
import os

import torch.nn as nn
import transformers
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer
import pandas as pd

logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from transformers import StoppingCriteria, LlamaForCausalLM
from typing import Dict
from . import conversation as conversation_lib
from utils.basics.os_utils import time_logger
from utils.pkg.hf_utils import download_hf_ckpt_to_local
import torch as th
from torch.nn.utils import rnn
from bidict import bidict

IGNORE_INDEX = -100
import time


def find_consecutive_subarrays(arr):
    if not arr:
        return []

    subarrays = []
    current_subarray = [arr[0]]

    for i in range(1, len(arr)):
        if arr[i] == arr[i - 1] + 1:
            current_subarray.append(arr[i])
        else:
            subarrays.append(current_subarray)
            current_subarray = [arr[i]]

    subarrays.append(current_subarray)
    return subarrays


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        if model.get_output_embeddings() is not None:
            output_embeddings = model.get_output_embeddings().weight.data
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings[-num_new_tokens:] = output_embeddings_avg


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = [tokenizer(keyword).input_ids for keyword in keywords]
        self.keyword_ids = [keyword_id[0] for keyword_id in self.keyword_ids if
                            type(keyword_id) is list and len(keyword_id) == 1]
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: th.LongTensor, scores: th.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            for keyword_id in self.keyword_ids:
                if output_ids[0, -1] == keyword_id:
                    return True
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False


def build_one_instance_supervised(tokenizer, sources, conv_template):
    # ! The code is modified from LLaVA's code
    conv = conv_template.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
    # Mask targets
    role_sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)  # </s>
        cur_len = 1  # Currently processed length, start from masking BOS token
        target[:cur_len] = IGNORE_INDEX
        for i, round_text in enumerate(rounds):
            if round_text == "":
                break
            # ! Mask human instructions
            parts = round_text.split(role_sep)
            if len(parts) != 2:
                break
            parts[0] += role_sep
            round_len = len(tokenizer(round_text).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2  # BOS + space
            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX  # The rest are masked
        # if cur_len < tokenizer.model_max_length:
        #     if cur_len != total_len:
        #         target[:] = IGNORE_INDEX
        #         logger.debug(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. (ignored)")
        assert sum(target != -100) > 0

    return [], input_ids, targets


def process_batch_instance(tokenizer, conversation_list, max_tgt_len, conv_template, device):
    _, batch_input_ids, batch_target_ids = build_one_instance_supervised(tokenizer, conversation_list,
                                                                         conv_template)
    input_ids = rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    target_ids = rnn.pad_sequence(batch_target_ids, batch_first=True, padding_value=IGNORE_INDEX)
    assert input_ids.size() == target_ids.size()
    input_ids = input_ids[:, :max_tgt_len]
    target_ids = target_ids[:, :max_tgt_len]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    assert attention_mask.size() == input_ids.size()
    return input_ids.to(device), target_ids.to(device), attention_mask.long().to(device)


def process_batch_instance_for_inference(left_tokenizer, batch_input_text, device):
    input_ids = left_tokenizer(
        batch_input_text,
        return_tensors="pt",
        padding="longest",
        max_length=left_tokenizer.model_max_length,
        truncation=True,
        add_special_tokens=True,
    ).input_ids
    attention_mask = input_ids.ne(left_tokenizer.pad_token_id)
    assert attention_mask.size() == input_ids.size()
    return input_ids.to(device), attention_mask.long().to(device)


class CovidLLM(nn.Module):
    '''LoRA for LLaMa model'''

    def __init__(self, cfg, data, logger):
        super(CovidLLM, self).__init__()
        self.cfg = cfg
        self.data = data
        self.logger = logger
        self.device = th.cuda.current_device() if th.cuda.is_available() else th.device('cpu')

        if self.cfg.ds.bf16.enable:
            self.float_type = th.bfloat16
        else:
            self.float_type = th.float32
        if self.cfg.ds.fp16.enabled:
            self.float_type = th.float16
        self.conv_template = conversation_lib.conv_templates[cfg.conv_template]
        max_tgt_len = cfg['max_tgt_len']
        self.gpt_response_prompt = data.prompt.gpt.template.split('{answer}')[0]

        # # Load checkpoint
        download_hf_ckpt_to_local(cfg.llm.hf_name, cfg.llm.local_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.llm.local_dir,
            use_fast=False,
            model_max_length=max_tgt_len,
            padding_side="right",
        )
        # ! UNK and EOS token leads to error
        # self.tokenizer.pad_token = self.tokenizer.unk_token  # Leads to error
        self.tokenizer.pad_token = '<pad>'  # Deal with empty unk token bug
        with time_logger(f'initialization of LLM decoder from {cfg.llm.local_dir}'):
            self.llm = LlamaForCausalLM.from_pretrained(cfg.llm.local_dir)
        self.llm.config.use_cache = False
        self.cls_token_names = class_tokens = [r.label_token for i, r in data.label_info.iterrows()]
        fields_to_add = cfg.data.static_cols + cfg.data.dynamic_cols
        field_names = [cfg.tree_node_alias.get(f, f) for f in fields_to_add]
        field_tokens = sum([[f'<{f}>', f'</{f}>'] for f in field_names], [])
        special_tokens = []

        special_tokens = []
        if cfg.get('add_class_token', True):
            special_tokens += class_tokens
        if cfg.get('add_field_token', True):
            special_tokens += field_tokens
        if cfg.get('add_pad_token', True):
            special_tokens += ['<pad>']
        if cfg.get('add_info_token', True):
            special_tokens += ['<information>', '</information>']
        if len(special_tokens) > 0:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict={'additional_special_tokens': special_tokens},
                tokenizer=self.tokenizer,
                model=self.llm,
            )
            self.choice_ids = [self.tokenizer([_]).input_ids[0][1] for _ in class_tokens]
            self.tok_to_id = bidict({t: self.tokenizer.convert_tokens_to_ids(t) for t in special_tokens})
            self.id_to_tok = self.tok_to_id.inverse
            self.cls_tokens = self.tokenizer.convert_tokens_to_ids(class_tokens)

        self.left_tokenizer = copy.deepcopy(self.tokenizer)
        self.left_tokenizer.padding_side = 'left'

        # Data related
        for id, _ in data.label_info.iterrows():
            data.label_info.loc[id]['label_name'] = self.tokenizer.decode(self.tokenizer(_.label_name).input_ids[1:])

        self.lid_to_lname = bidict({_.label_token: _.label_name
                                    for id, _ in data.label_info.iterrows()})
        self.lname_to_lid = self.lid_to_lname.inverse

        if self.cfg.use_lora:
            # add the lora module
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.cfg.lora.r,
                lora_alpha=self.cfg.lora.alpha,
                lora_dropout=self.cfg.lora.dropout,
                target_modules=self.cfg.lora.target_modules,
            )
            self.llm = get_peft_model(self.llm, peft_config)
            self.llm.print_trainable_parameters()
        logger.info('LLAMA proj initialized.')

        if cfg.frozen_llm:
            for name, param in self.llm.named_parameters():
                param.requires_grad = False

            # ! Since new tokens are added, it is vital to train them
            for p in self.llm.get_input_embeddings().parameters():
                p.requires_grad = True
            for p in self.llm.get_output_embeddings().parameters():
                p.requires_grad = True
            logging.info('The LLM LLAMA is frozen except input and output embeddings.')
        self.max_tgt_len = max_tgt_len

    def get_input_embeddings(self, input_ids):
        batch_size = input_ids.shape[0]
        # !Lookup text embeddings
        if self.llm.base_model.__class__.__name__ == 'LlamaModel':
            inputs_embeds = self.llm.model.embed_tokens(
                input_ids).expand(batch_size, -1, -1)  # bsz x s2 x embed_dim
        else:
            inputs_embeds = self.llm.model.model.embed_tokens(
                input_ids).expand(batch_size, -1, -1)  # bsz x s2 x embed_dim
        return inputs_embeds

    def forward(self, inputs):
        node_ids, prompt_tree_lol, conversation_list = inputs
        # ! Get Graph Language
        # ! Tokenization: batch instance to input and target IDs.
        input_ids, target_ids, attention_mask = process_batch_instance(self.tokenizer, conversation_list,
                                                                       self.max_tgt_len, self.conv_template,
                                                                       self.device)
        inputs_embeds = self.get_input_embeddings(input_ids)
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=target_ids,
        )
        return outputs, target_ids

    def match_label_from_text(self, text):
        if self.cfg.add_class_token:
            predicted_tokens = self.tokenizer(text).input_ids
            matched = [self.id_to_tok[cls] for cls in self.cls_tokens if cls in predicted_tokens]
        else:
            text = text.replace('<s>', '')
            matched = [label_name for label_id, label_name in self.lid_to_lname.items() if label_id in text]
        if len(matched) == 0:
            return text, False
        elif len(matched) == 1:
            return matched[0], True
        else:
            return f'Multiple labels matched {matched}', False

    def generate(self, inputs, choice_only=False):
        # ! Prepare input
        node_ids, prompt_tree_lol, conversation_list = inputs['batch']
        # <node> [1286, 72, 19] </node> -> <node> [3, 768] emb </node>
        batch_input_text = []
        for c in conversation_list:
            conv = self.conv_template.copy()
            conv.append_message(conv.roles[0], c[0]['value'])
            conv.append_message(conv.roles[1], self.gpt_response_prompt)  # ASSISTANT: The answer is:
            # conv.append_message(conv.roles[1], None)  # ASSISTANT:
            # Remove Gold response
            _prompt = conv.get_prompt().strip(conv.sep2)
            batch_input_text.append(_prompt)
        readout_pos = self.cfg.get('choice_readout_pos', 0)

        start_time = time.time()
        batch_input_ids, attention_mask = process_batch_instance_for_inference(
            self.left_tokenizer, batch_input_text, self.device)
        batch_inputs_embeds = self.get_input_embeddings(batch_input_ids)
        attention_mask = attention_mask.to(self.device)
        # Mask embedding attn_mask=0 to zeros
        masked_batch_embedding = batch_inputs_embeds * attention_mask.unsqueeze(-1).to(batch_inputs_embeds.dtype)
        # Run model inference
        with th.inference_mode():
            batch_output = self.llm.generate(
                inputs_embeds=masked_batch_embedding,
                attention_mask=attention_mask,
                max_new_tokens=inputs['max_tgt_len'] if not choice_only else 3,
                # Too low temp leads to inf prob error.
                output_scores=choice_only,
                use_cache=True,
                return_dict_in_generate=choice_only,
            )
        if choice_only:  # The answer is:
            batch_preds = batch_output.scores[readout_pos][:, self.choice_ids].argmax(-1).cpu().tolist()
            batch_out_text = [self.cls_token_names[_] for _ in batch_preds]
        else:
            batch_out_text = self.tokenizer.batch_decode(batch_output, skip_special_tokens=False)
        outputs = {'dialog': [p + o for p, o in zip(batch_input_text, batch_out_text)],
                   'generated_text': batch_out_text}
        if self.cfg.add_loop_inference:
            self.logger.info(f"BATCH inference time: {time.time() - start_time:.2f} seconds")
            input_id_list = self.tokenizer(batch_input_text).input_ids
            loop_outputs = []
            # ! Generate one by one as batch generation requires adding <pad> tokens to prompt and leads to confusion
            start_time = time.time()
            for i, (node_id, input_ids) in enumerate(zip(node_ids, input_id_list)):
                input_ids = th.as_tensor(input_ids).view(1, -1).to(self.device)
                input_embeds = self.get_input_embeddings(input_ids)
                # Run model inference
                with th.inference_mode():
                    output = self.llm.generate(
                        inputs_embeds=input_embeds,
                        max_new_tokens=inputs['max_tgt_len'] if not choice_only else 3,
                        output_scores=choice_only,
                        use_cache=True,
                        return_dict_in_generate=choice_only,
                    )

                # Decode output tokens
                if not choice_only:
                    out_text = self.tokenizer.decode(output[0], skip_special_tokens=False)
                    # out_text = out_text.strip().rstrip(stop_str).strip()
                    loop_outputs.append(out_text)
                else:
                    # out_topk_choices = [self.tokenizer.convert_ids_to_tokens(s.topk(3).indices.squeeze())
                    #                     for s in output.scores]
                    # logger.debug(f"Gold {inputs['gold_text'][i]}. Generated: {out_topk_choices}")
                    class_logits = output.scores[readout_pos].squeeze()[self.choice_ids]
                    out_text = self.cls_token_names[class_logits.argmax().item()]
                    loop_outputs.append(out_text)
            outputs['loop_generated_text'] = loop_outputs
            self.logger.info(f"LOOP inference time: {time.time() - start_time:.2f} seconds")
        return outputs

    def generate_prob(self, inputs):
        # ! Prepare input
        node_ids, prompt_tree_lol, encode_seq, node_id_to_encode_id, conversation_list = inputs['batch']
        # <node> [1286, 72, 19] </node> -> <node> [3, 768] emb </node>
        emb = {f: self.encoder[f](seq.to(self.float_type).to(self.device)) for f, seq in encode_seq.items()}
        prompt = []
        for c in conversation_list:
            conv = self.conv_template.copy()
            conv.append_message(conv.roles[0], c[0]['value'])
            # conv.append_message(conv.roles[1], self.gpt_response_prompt) # ASSISTANT: The answer is:
            # conv.append_message(conv.roles[1], None) # ASSISTANT:
            # Remove Gold response
            _prompt = conv.get_prompt().strip(conv.sep2)
            prompt.append(_prompt)

        input_id_list = self.tokenizer(prompt).input_ids
        outputs = []

        # ! Generate one by one as batch generation requires adding <pad> tokens to prompt and leads to confusion
        for i, (node_id, input_ids) in enumerate(zip(node_ids, input_id_list)):
            input_ids = th.as_tensor(input_ids).view(1, -1).to(self.device)
            input_embeds = self.prompt_wrap(emb, [node_id], input_ids, node_id_to_encode_id)
            # Define stopping criteria for generation
            conv = self.conv_template.copy()
            stop_str = conv.sep if conv.sep_style != conversation_lib.SeparatorStyle.TWO else conv.sep2
            stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)
            # Run model inference
            with th.inference_mode():
                output = self.llm.generate(
                    inputs_embeds=input_embeds,
                    max_new_tokens=inputs['max_tgt_len'],
                    do_sample=False,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )

            # Decode output tokens
            out_text = self.tokenizer.decode(output[0], skip_special_tokens=False)
            out_text = out_text.strip().rstrip(stop_str).strip()
            outputs.append(out_text)

        return {'dialog': [p + o for p, o in zip(prompt, outputs)],
                'generated_text': outputs,
                }

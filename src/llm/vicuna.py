"""
#! Modified from FastChat with Hugging Face generation APIs.
"""

import torch
from fastchat.model import load_model

from llm.llm import LLM
from utils.basics import logger


class Vicuna(LLM):
    def __init__(self, model_path, data, debug=False, num_gpus=1, max_gpu_memory='48Gib',
                 device="cuda", cpu_offloading=False,
                 load_8bit=False, temperature=0.7, max_input_len=1800,
                 **kwargs):
        self.model, self.tokenizer = load_model(
            model_path,
            device,
            num_gpus,
            max_gpu_memory,
            load_8bit,
            cpu_offloading,
            debug=debug,
        )
        self.temperature = max(float(temperature), 0.01)  # Too low temp leads to inf prob error.
        self.choices = choices = list(data.choice_to_label_id)
        self.choice_ids = [self.tokenizer([_]).input_ids[0][1] for _ in choices]
        self.max_input_len = max_input_len

    @torch.inference_mode()
    def generate_text(self, prompt, max_new_tokens=10, choice_only=False, choices=['']):
        # example_prompt = """
        # "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful,
        # detailed, and polite answers to the user's questions. USER: Hello! Who are you? ASSISTANT:"
        # """
        input_ids = self.tokenizer([prompt]).input_ids
        if (origin_len := len(input_ids[0])) > self.max_input_len:
            logger.warning(f'Input truncated {self.max_input_len}/{origin_len}!!')
            input_ids[0] = input_ids[0][:self.max_input_len]
        output = self.model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=True,
            temperature=self.temperature,
            max_new_tokens=1 if choice_only else max_new_tokens,
            return_dict_in_generate=True, output_scores=True
        )
        if choice_only:
            choice = self.choices[output.scores[0].squeeze()[self.choice_ids].argmax()]
            return choice
        return self.tokenizer.decode(output.sequences[0][-max_new_tokens:])

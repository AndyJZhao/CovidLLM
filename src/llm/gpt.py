import os
from time import sleep

import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential

from utils.basics import logger
from .llm import LLM
openai.api_key = os.environ["OPENAI_API_KEY"]


class GPT(LLM):
    model2price = {
        "gpt-3.5-turbo": (0.0015 / 1000, 0.002 / 1000),
        "gpt-4": (0.03 / 1000, 0.06 / 1000),
        "gpt-4-32k": (0.06 / 1000, 0.12 / 1000),
    }
    token_per_word = 1.33

    def __init__(self, openai_name="gpt-3.5-turbo", temperature=0, top_p=1, max_tokens=200, sleep_time=1, **kwargs):
        self.model = openai_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.sleep_time = sleep_time
        logger.critical(f'Using OPENAI {openai_name.upper()}')
        logger.critical(f'Using OPENAI-APIKey{os.environ["OPENAI_API_KEY"]}')

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate_text(self, prompt, max_new_tokens=10, choice_only=False):
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0 if choice_only else self.temperature,
            top_p=self.top_p,
            max_tokens=1 if choice_only else self.max_tokens
        )
        sleep(self.sleep_time)
        return response["choices"][0]["message"]["content"]

    def estimate_price(self, prompt):
        prompt_price, completion_price = self.model2price[self.model]
        num_word = len(prompt.split())
        num_token = num_word * self.token_per_word
        price = num_token * prompt_price + self.max_tokens * completion_price
        return price

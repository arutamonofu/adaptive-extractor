# src/aee/llm.py

import time
import logging
import json
import requests
from threading import Lock
from typing import Any, List, Union, Optional
from functools import wraps

import dspy
from aee.core.config import settings, LLMInstanceConfig

logger = logging.getLogger(__name__)

class OllamaLM(dspy.LM):
    MAX_HISTORY = 100  # Keep only last N interactions to save RAM

    def __init__(self, config: LLMInstanceConfig):
        super().__init__(config.model)

        self.model = config.model
        self.temperature = config.temperature
        self.timeout = config.timeout
        self.max_retries = config.max_retries
        self.top_p = config.top_p
        
        oc = config.ollama
        self.base_url = oc.ollama_base_url.rstrip("/") + "/api/chat"
        self.num_ctx = oc.num_ctx
        self.num_predict = oc.num_predict
        self.stream = oc.stream
        self.repeat_penalty = oc.repeat_penalty
        self.repeat_last_n = oc.repeat_last_n
        self.provider = "ollama"
        self.history = []

    def __call__(self, prompt: Union[str, List[dict]] = None, **kwargs) -> List[str]:
        if prompt is None:
            prompt = kwargs.get("messages")
            
        if prompt is None:
            return [""]

        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt

        payload = {
            "model": self.model,
            "messages": messages,
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": self.temperature,
                "num_predict": self.num_predict,
                "top_p": self.top_p,
                "repeat_penalty": self.repeat_penalty,
                "repeat_last_n": self.repeat_last_n,
            },
            "stream": self.stream
        }

        attempt = 0
        while attempt < self.max_retries:
            try:
                with requests.post(self.base_url, json=payload, stream=self.stream, timeout=self.timeout) as response:
                    response.raise_for_status()
                    
                    full_content = []
                    print(f"\n[LLM] Streaming response from {self.model}...", end="", flush=True)
                    for line in response.iter_lines():
                        if line:
                            try:
                                body = json.loads(line)
                                if "message" in body and "content" in body["message"]:
                                    content_chunk = body["message"]["content"]
                                    full_content.append(content_chunk)
                                    if self.stream:
                                        print(content_chunk, end='', flush=True)
                                if body.get("done", False):
                                    if self.stream:
                                        print() # Перенос строки в конце ответа
                                    break
                            except json.JSONDecodeError:
                                continue
                    
                    text_response = "".join(full_content)

                self.history.append({
                    "prompt": messages,
                    "messages": messages,
                    "outputs": [text_response],
                    "model": self.model,
                    "kwargs": kwargs
                })

                if len(self.history) > self.MAX_HISTORY:
                    self.history = self.history[-self.MAX_HISTORY:]

                return [text_response]

            except Exception as e:
                attempt += 1
                logger.warning(f"Ollama error (Attempt {attempt}/{self.max_retries}): {e}")
                if attempt == self.max_retries:
                    logger.error(f"Ollama failed after retries: {e}")
                    raise e
                time.sleep(2)
        
        return [""]


def _apply_rate_limit(lm: dspy.LM, delay: float) -> dspy.LM:
    """
    Applies a thread-safe rate limit specific to this LM instance.
    """
    original_call = lm.__call__
    instance_lock = Lock()

    @wraps(original_call)
    def _throttled_call(*args, **kwargs):
        with instance_lock:
            result = original_call(*args, **kwargs)
            if delay > 0:
                time.sleep(delay)
            return result

    lm.__call__ = _throttled_call
    return lm


def create_lm(config: LLMInstanceConfig) -> dspy.LM:
    logger.info(f"Initializing LLM: {config.model} (Ollama: {config.use_ollama})")

    if config.use_ollama:
        lm = OllamaLM(config)
    else:
        api_key = config.non_ollama.api_key.get_secret_value() if config.non_ollama.api_key else None
        lm = dspy.LM(
            model=config.model,
            api_key=api_key,
            temperature=config.temperature,
            max_tokens=config.non_ollama.max_tokens,
            cache=True
        )

    if config.rate_limit_delay > 0:
        lm = _apply_rate_limit(lm, config.rate_limit_delay)

    return lm


def setup_student() -> dspy.LM:
    lm = create_lm(settings.llm.student)
    dspy.settings.configure(lm=lm)
    return lm


def setup_teacher() -> dspy.LM:
    return create_lm(settings.llm.teacher)
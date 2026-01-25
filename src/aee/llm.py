# src/aee/llm.py

import time
import logging
import json
import requests
from functools import wraps
from threading import Lock
from typing import Any, Optional, List, Union

import dspy
from aee.core.config import settings

logger = logging.getLogger(__name__)

_API_LOCK = Lock()


class UniversityOllama(dspy.LM):
    """
    Robust DSPy LM implementation for Native Ollama API.
    
    Key Features:
    1. Direct API Access: Bypasses intermediate libraries to guarantee 'num_ctx' is respected.
    2. Streaming: Prevents proxy/load balancer timeouts during long generations.
    3. Resilience: Built-in retry logic for network fluctuations.
    4. DSPy Compatibility: Correctly formats history for optimizers (MIPROv2).
    """
    
    def __init__(
        self, 
        model: str, 
        base_url: str, 
        num_ctx: int = 32000, 
        temperature: float = 0.1,
        timeout: int = 1200, 
        max_retries: int = 3,
        **kwargs
    ):
        super().__init__(model)
        
        # Ensure we hit the chat endpoint
        self.base_url = base_url.rstrip("/") + "/api/chat"
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.kwargs = kwargs
        self.provider = "ollama"
        self.history = []

    def __call__(self, prompt: Union[str, List[dict]] = None, only_completed: bool = True, **kwargs) -> List[str]:
        """
        Sends request to Ollama with Streaming and Retries.
        """

        if prompt is None:
            prompt = kwargs.get("messages")
            
        if prompt is None:
            logger.warning("UniversityOllama called with no prompt/messages")
            return [""]

        messages = []
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": str(prompt)}]

        payload = {
            "model": self.model,
            "messages": messages,
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": self.temperature,
                "repeat_penalty": 1.2,
                "repeat_last_n": 2048,
                "num_predict": 3000,
                **self.kwargs.get("options", {})
            },
            "stream": True
        }

        attempt = 0
        while attempt < self.max_retries:
            try:
                with _API_LOCK:
                    with requests.post(self.base_url, json=payload, stream=True, timeout=self.timeout) as response:
                        response.raise_for_status()
                        
                        full_content = []
                        print(f"\n[DEBUG] Streaming response from {self.model}...", end="", flush=True)

                        for line in response.iter_lines():
                            if line:
                                try:
                                    body = json.loads(line)
                                    if "message" in body and "content" in body["message"]:
                                        content_chunk = body["message"]["content"]
                                        full_content.append(content_chunk)
                                        print(content_chunk, end="", flush=True) 
                                    
                                    if body.get("done", False):
                                        print("\n[DEBUG] Done.\n")
                                        break
                                except json.JSONDecodeError:
                                    continue
                        
                        text_response = "".join(full_content)

                mock_response = {
                    "choices": [{"message": {"role": "assistant", "content": text_response}}],
                    "model": self.model,
                    "usage": {"completion_tokens": len(full_content)}
                }

                self.history.append({
                    "prompt": messages,
                    "messages": messages,
                    "outputs": [text_response],
                    "response": mock_response,
                    "kwargs": kwargs
                })
                
                return [text_response]

            except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout, requests.exceptions.ChunkedEncodingError) as e:
                attempt += 1
                logger.warning(f"Ollama Request Failed (Attempt {attempt}/{self.max_retries}): {e}")
                if attempt == self.max_retries:
                    logger.error("Max retries reached. Returning empty response.")
                    return [""]
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Unexpected Ollama Error: {e}")
                return [""]
        
        return [""]

    def request(self, prompt: str, **kwargs):
        """Compatibility alias"""
        return self.__call__(prompt, **kwargs)


def _apply_rate_limit(lm: dspy.LM, delay: float) -> dspy.LM:
    """Decorator to enforce serial execution with a delay."""
    original_call = lm.__call__

    @wraps(original_call)
    def _throttled_call(*args: Any, **kwargs: Any) -> Any:
        with _API_LOCK:
            if delay > 0:
                time.sleep(delay)
            return original_call(*args, **kwargs)

    lm.__call__ = _throttled_call
    return lm


def create_lm(model: str, api_key: str, api_base: str | None = None, **kwargs: Any) -> dspy.LM:
    """
    Universal Factory: Routes to Local Ollama or Cloud Providers.
    """
    
    if api_base and "ollama" in api_base:
        clean_url = api_base.replace("/v1", "").replace("/api/chat", "").rstrip("/")
        
        # Automatic Context Sizing based on Model Capabilities
        # Scientific Mode requires > 50k tokens for Few-Shot + PDF
        if any(x in model for x in ["128k", "deepseek-r1", "gpt-oss"]):
            ctx_limit = 100000  # High limit for capable models
        elif "qwen3" in model:
            ctx_limit = 40000   # Qwen3 limit
        else:
            ctx_limit = 32000   # Safe default

        logger.info(f"🚀 Init Native Ollama (Stream): {model} | Ctx: {ctx_limit}")
        
        return UniversityOllama(
            model=model,
            base_url=clean_url,
            num_ctx=ctx_limit,
            temperature=settings.temperature,
            timeout=1200,
            **kwargs
        )

    logger.info(f"☁️ Init Cloud LLM: {model}")
    
    full_model_name = model
    if api_base and "/" not in model:
        full_model_name = f"openai/{model}"

    lm = dspy.LM(
        model=full_model_name,
        api_key=api_key,
        api_base=api_base,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        **kwargs,
    )

    if settings.rate_limit_delay > 0:
        lm = _apply_rate_limit(lm, settings.rate_limit_delay)

    return lm


def setup_student() -> dspy.LM:
    """
    Configures the Student model (Execution/Extraction).
    Typically a faster model with large context (e.g., Mistral Small 128k).
    """
    lm = create_lm(
        model=settings.student_model_name,
        api_key=settings.active_api_key,
        api_base=settings.active_api_base
    )
    dspy.settings.configure(lm=lm)
    return lm


def setup_teacher() -> dspy.LM:
    """
    Configures the Teacher model (Reasoning/Bootstrapping).
    Typically a smarter/larger model (e.g., DeepSeek R1).
    """
    return create_lm(
        model=settings.teacher_model_name,
        api_key=settings.active_api_key,
        api_base=settings.active_api_base
    )
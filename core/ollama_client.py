"""
Ollama API client for text generation and embeddings.
"""

import requests
import json
import logging
from typing import Dict, List, Optional, Any, Union
from core.config import config

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama API."""

    def __init__(self, api_url: Optional[str] = None, timeout: Optional[int] = None):
        """Initialize Ollama client."""
        self.api_url = api_url or config.ollama.api_url
        self.timeout = timeout or config.ollama.timeout
        self.session = requests.Session()

    def _make_request(self, endpoint: str, payload: Dict[str, Any]) -> Union[Dict[str, Any], requests.Response]:
        """Make HTTP request to Ollama API."""
        url = f"{self.api_url}/api/{endpoint}"

        try:
            response = self.session.post(
                url,
                json=payload,
                timeout=self.timeout,
                stream=payload.get("stream", False)
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API request failed: {e}")
            raise

    def generate_text(
        self,
        prompt: str,
        model: str = "llama2",
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None
    ) -> Union[str, Dict[str, Any]]:
        """Generate text using Ollama model."""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }

        if options:
            payload["options"] = options

        response = self._make_request("generate", payload)

        if stream:
            return response
        else:
            return response.json().get("response", "")

    def generate_text_stream(self, prompt: str, model: str = "llama2", options: Optional[Dict[str, Any]] = None):
        """Generate text with streaming using Ollama model."""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True
        }

        if options:
            payload["options"] = options

        response = self._make_request("generate", payload)

        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line.decode('utf-8'))
                    if chunk.get("done", False):
                        break
                    token = chunk.get("response", "")
                    if token:
                        yield token
                except json.JSONDecodeError:
                    continue

    def get_embeddings(self, text: Union[str, List[str]], model: Optional[str] = None) -> List[List[float]]:
        """Get embeddings for text using Ollama."""
        model_name = model or config.ollama.embedding_model

        if isinstance(text, str):
            text = [text]

        payload = {
            "model": model_name,
            "prompt": text[0] if len(text) == 1 else text
        }

        response = self._make_request("embeddings", payload)

        # Handle both single and batch embeddings
        if isinstance(response.get("embedding"), list):
            if isinstance(response["embedding"][0], list):
                # Batch embeddings
                return response["embedding"]
            else:
                # Single embedding
                return [response["embedding"]]
        else:
            raise ValueError("Unexpected embedding response format")

    def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        try:
            response = self.session.get(f"{self.api_url}/api/tags", timeout=self.timeout)
            response.raise_for_status()
            return response.json().get("models", [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def check_health(self) -> bool:
        """Check if Ollama service is healthy."""
        try:
            response = self.session.get(f"{self.api_url}/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def generate_with_context(
        self,
        messages: List[Dict[str, str]],
        model: str = "llama2",
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None
    ) -> Union[str, Dict[str, Any]]:
        """Generate text with conversation context."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }

        if options:
            payload["options"] = options

        response = self._make_request("chat", payload)

        if stream:
            return response
        else:
            return response.get("message", {}).get("content", "")


# Global client instance
ollama_client = OllamaClient()


def generate_text(prompt: str, model: str = "llama2", **kwargs) -> str:
    """Convenience function for text generation."""
    return ollama_client.generate_text(prompt, model, **kwargs)


def get_embeddings(text: Union[str, List[str]], model: Optional[str] = None) -> List[List[float]]:
    """Convenience function for getting embeddings."""
    return ollama_client.get_embeddings(text, model)


def list_available_models() -> List[Dict[str, Any]]:
    """Convenience function to list available models."""
    return ollama_client.list_models()


def check_ollama_status() -> bool:
    """Convenience function to check Ollama service status."""
    return ollama_client.check_health()


if __name__ == "__main__":
    # Test the client
    if check_ollama_status():
        print("✅ Ollama service is running")
        models = list_available_models()
        print(f"Available models: {[m['name'] for m in models]}")

        # Test embedding generation
        try:
            embedding = get_embeddings("Hello world")
            print(f"✅ Embedding generated successfully (dimension: {len(embedding[0])})")
        except Exception as e:
            print(f"❌ Embedding generation failed: {e}")
    else:
        print("❌ Ollama service is not running")
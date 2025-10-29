import os
from typing import Any, Dict, Optional, TYPE_CHECKING
from dotenv import load_dotenv

load_dotenv()

try:  # pragma: no cover
    from openai import OpenAI
except ImportError as e:  # pragma: no cover
    OpenAI = None  # type: ignore

_CLIENT: Any = None  # defer concrete type to runtime


def _build_client() -> Any:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not endpoint or not api_key:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set")
    if OpenAI is None:  # pragma: no cover
        raise ImportError("openai package not installed. Install with `pip install openai`.")
    # Ensure endpoint ends with /openai/v1/ for the Azure OpenAI v1 API
    # Accept endpoints already containing /openai/.* and trust user if custom.
    return OpenAI(base_url=endpoint.rstrip("/" ) + "/" if not endpoint.endswith("/") else endpoint, api_key=api_key)


def get_client() -> Any:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = _build_client()
    return _CLIENT


def chat_completion(
    deployment: str,
    system: str,
    user: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    response_format: Optional[Dict[str, Any]] = None,
) -> str:
    """Send a chat completion request using the OpenAI SDK (Azure OpenAI v1 endpoint).

    Parameters are consistent with previous implementation for backward compatibility.
    """
    client = get_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    kwargs: Dict[str, Any] = {
        "model": deployment,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # Basic mapping for response_format; OpenAI v1 supports response_format={"type": "json_object"}
    if response_format:
        # If user passed full dict, forward as-is; if simple alias, convert
        if isinstance(response_format, dict):
            kwargs["response_format"] = response_format
        else:  # pragma: no cover
            kwargs["response_format"] = {"type": str(response_format)}
    # Sequential adaptive retries for evolving parameter requirements.
    # Order of transforms we may apply:
    # 1. max_tokens -> max_completion_tokens
    # 2. remove temperature if rejected
    # 3. max_completion_tokens -> max_tokens (inverse future-proof)
    attempted_transforms = []
    for _ in range(4):  # hard cap to avoid infinite loops
        try:
            completion = client.chat.completions.create(**kwargs)
            break
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            # 1. Switch max_tokens -> max_completion_tokens
            if (
                "Unsupported parameter" in msg
                and "'max_tokens'" in msg
                and "max_completion_tokens" in msg
                and "max_tokens->max_completion_tokens" not in attempted_transforms
                and "max_tokens" in kwargs
            ):
                val = kwargs.pop("max_tokens")
                kwargs["max_completion_tokens"] = val
                attempted_transforms.append("max_tokens->max_completion_tokens")
                continue
            # 2. Remove temperature if not allowed
            if (
                ("Unsupported value" in msg or "Unsupported parameter" in msg)
                and "temperature" in msg
                and "drop-temperature" not in attempted_transforms
                and "temperature" in kwargs
            ):
                kwargs.pop("temperature", None)
                attempted_transforms.append("drop-temperature")
                continue
            # 3. Inverse: max_completion_tokens -> max_tokens
            if (
                "Unsupported parameter" in msg
                and "'max_completion_tokens'" in msg
                and "max_completion_tokens->max_tokens" not in attempted_transforms
                and "max_completion_tokens" in kwargs
            ):
                val = kwargs.pop("max_completion_tokens")
                kwargs["max_tokens"] = val
                attempted_transforms.append("max_completion_tokens->max_tokens")
                continue
            # No recognized fallback left
            raise RuntimeError(
                f"Chat completion request failed after transformations {attempted_transforms}: {e}"
            ) from e
    else:  # for-else executes if loop did not break
        raise RuntimeError(
            f"Chat completion failed: exceeded adaptive retry attempts ({attempted_transforms})"
        )

    try:
        return completion.choices[0].message.content or ""  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return str(completion)

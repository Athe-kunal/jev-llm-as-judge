"""Qwen chat formatting for prompts sent to the vLLM servers."""


def chat_prefix(user: str, think: bool = False) -> str:
    """Qwen chat format for one user message, up to the reply.

    Args:
      user: The user message.
      think: Let the model reason in a <think> block first (default: an
        empty think block, so it answers at once).
    """
    head = f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    return head + ("<think>\n" if think else "<think>\n\n</think>\n\n")

"""
Chat class for maintaining conversation history
"""
from typing import Optional, List, Dict, Literal

class Chat:
    """
    Maintains conversation history as a list of dicts with fixed structure.
    """
    def __init__(
        self, 
        system_prompt: Optional[str] = None
        ):
        """
        Initialize the chat with a conversation type and optional system prompt.
        Args:
            conversation_type (str): Type of conversation, either "chat" or "base".
            system_prompt (Optional[str]): Optional system prompt to initialize the chat.
        """
        self.messages: List[Dict[str, str]] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def __repr__(self) -> str:
        return f"Chat with {len(self.messages)} messages\n{self.get_conversation_log()}"
    
    def add_system_message(self, content: str) -> None:
        """Add a system message"""
        self.messages.append({"role": "system", "content": content})
    
    def add_user_message(self, content: str) -> None:
        """Add a user message"""
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message"""
        self.messages.append({"role": "assistant", "content": content})

    def clear(self) -> None:
        """Clear all non-system messages"""
        self.messages = [msg for msg in self.messages if msg.get("role") == "system"]

    def get_conversation(self) -> List[Dict[str, str]]:
        """Return the raw conversation list"""
        return self.messages

    def get_conversation_log(self) -> str:
        """Return human-readable conversation log"""
        log_lines = []
        for msg in self.messages:
            role = msg.get("role")
            content = msg.get("content")
            if role == "system":
                log_lines.append(f"System: {content}")
            elif role == "user":
                log_lines.append(f"You: {content}")
            elif role == "assistant":
                log_lines.append(f"Assistant: {content}")
        return "\n".join(log_lines)
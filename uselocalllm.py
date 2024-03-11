from typing import Optional, List, Mapping, Any

from llama_index.core import SimpleDirectoryReader, SummaryIndex
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata
)
from llama_index.core.llms.callbacks import llm_completion_callback

#from llama_index.core.llms import ChatMessage
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms import ChatMessage

model = OpenAILike(
    api_key="localai_fake",
    api_type="localai_fake",
    api_base="http://localhost:8080/v1",
    model="llama",
    is_chat_model=True,
    timeout=60)

response = model.chat(messages=[ChatMessage(content="how r u?")])
print(response)
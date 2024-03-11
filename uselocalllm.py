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
from llama_index.core import Settings

model = OpenAILike(
    api_key="localai_fake",
    api_type="localai_fake",
    api_base="http://localhost:8080/v1",
    model="llama",
    is_chat_model=True,
    timeout=60)

Settings.llm = model

documents = SimpleDirectoryReader("./repo").load_data()
index = SummaryIndex.from_documents(documents)


# Query and print response
query_engine = index.as_query_engine()
response = query_engine.query("How is the average time calculated ? In which function called where ?")
print(response)
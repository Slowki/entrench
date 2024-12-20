import json
from typing import Final, Sequence

import ollama

from entrench.embedder.api import Embedder, EmbeddingContext
from entrench.status import Status

_SYSTEM_PROMPT: Final = """\
You are a language model designed to generate context for text chunks to be used in contextual retrieval systems.
Your goal is to produce accurate, concise, and relevant context that captures the essential information of each text chunk, enabling effective retrieval based on user queries.
You are processing chunks of text from markdown files.
"""

_USER_PROMPT: Final = """\
You are processing a document from a file with the path {path}.
The title of the document is "{title}" and the piece of text comes from the section "{section}".

Generate a sentence that provides context for the following piece of text:
{text}
"""


class OllamaEmbedder(Embedder):
    """An embedder that uses Ollama."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        embedding_model: str = "mxbai-embed-large:335m",
        context_model: str = "llama3.2:1b-instruct-q5_K_M",
    ) -> None:
        self.host = host
        self.embedding_model = embedding_model
        self.context_model = context_model
        self.client = ollama.AsyncClient(host=self.host)

    async def set_up(self) -> None:
        status = Status.get()
        models = [x.model for x in (await self.client.list()).models]

        if self.embedding_model not in models:
            status.update(f"Pulling {self.embedding_model}")
            await self.client.pull(self.embedding_model)

        if self.context_model not in models:
            status.update(f"Pulling {self.context_model}")
            await self.client.pull(self.context_model)

        return await super().set_up()

    async def embed_file(
        self, context: EmbeddingContext, text: str
    ) -> Sequence[Sequence[float]]:
        """Generate an embedding for the given text.

        Args:
            context: The entire document.
            text: The chunk of text to embed.
        """

        try:
            chat_response = await self.client.chat(
                model=self.context_model,
                messages=[
                    ollama.Message(role="system", content=_SYSTEM_PROMPT),
                    ollama.Message(
                        role="user",
                        content=_USER_PROMPT.format(
                            path=context.file_path,
                            title=context.title,
                            section=context.section,
                            text=text,
                        ),
                    ),
                ],
                format={
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "summary": {
                    "type": "string"
                    }
                },
                "required": ["summary"]
                },
            )
            data = json.loads(chat_response.message.content)  # type: ignore
        except ollama.ResponseError:
            data = {"summary": ""}
            pass

        return await self.embed(data["summary"] + "\n" + text)

    async def embed(self, text: str) -> Sequence[Sequence[float]]:
        """Generate an embedding for the given text.

        Args:
            text: The chunk of text to embed.
        """
        embedding_response = await self.client.embed(
            model=self.embedding_model, input=text
        )
        return embedding_response.embeddings

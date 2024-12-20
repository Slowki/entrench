"""The base API for embedders."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence


@dataclass
class EmbeddingContext:
    file_path: str = ""
    title: str = ""
    section: str = ""


class Embedder(ABC):
    """Abstract base class for embedding generators."""

    CHUNK_SIZE: int = 1000

    async def set_up(self) -> None:
        """Set up the embedding generator."""

    @abstractmethod
    async def embed_file(
        self, context: EmbeddingContext, text: str
    ) -> Sequence[Sequence[float]]:
        """Generate an embedding for the given text.

        Args:
            context: The context of the chunk.
            text: The chunk of text to embed.
        """

    @abstractmethod
    async def embed(self, text: str) -> Sequence[Sequence[float]]:
        """Generate an embedding for the given text.

        Args:
            text: The chunk of text to embed.
        """

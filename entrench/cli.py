"""Generate and store embedding of your Markdown files in your codebase."""

import asyncio
import hashlib
import struct
import subprocess
import sys
from collections import defaultdict
from pathlib import Path, PurePath
from typing import Final, Sequence, Iterable

import click
import numpy as np
import rich
from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode

from entrench.embedder.api import EmbeddingContext
from entrench.embedder.ollama import OllamaEmbedder
from entrench.status import Status, make_status

_OLLAMA_CONTAINER_NAME: Final = "ollama"
_TRENCH_FILE: Final = ".trench"
_DIVIDER: Final = "----"


class EmbeddingsFile:
    """A file sthat stores embeddings."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.directory = path.parent
        self.file_hashes = {}
        self._file_embeddings = None

        try:
            with self.path.open("rt") as file:
                while line := file.readline().strip():
                    if not line or line == _DIVIDER:
                        break
                    file_name, hash = line.split()
                    self.file_hashes[file_name] = hash
        except FileNotFoundError:
            pass

    @property
    def file_embeddings(self) -> dict[str, list[str]]:
        if self._file_embeddings is None:
            self._file_embeddings = defaultdict(list)
            try:
                hit_divider = False
                with self.path.open("rt") as file:
                    while line := file.readline().strip():
                        if line == _DIVIDER:
                            hit_divider = True
                            continue
                        if not hit_divider:
                            continue
                        file_name, embedding = line.split()

                        self._file_embeddings[file_name].append(embedding)
            except FileNotFoundError:
                pass

        return self._file_embeddings

    def get_hash(self, path: PurePath) -> str:
        """Get the hash of the version of the file the last time it was embedded."""
        return self.file_hashes.get(path.name, "")

    def set_embeddings(self, path: PurePath, hash: str, embeddings: list[str]) -> None:
        """Set the embedding of a file."""
        self.file_hashes[path.name] = hash
        self.file_embeddings[path.name] = embeddings

    def write(self) -> None:
        """Write the embeddings to the file."""
        # First read in all the embeddings
        with self.path.open("wt") as file:
            for file_name, hash in sorted(self.file_hashes.items(), key=lambda x: x[0]):
                file.write(f"{file_name} {hash}\n")
            file.write(_DIVIDER + "\n")
            for file_name, embeddings in sorted(self.file_embeddings.items(), key=lambda x: x[0]):
                for embedding in embeddings:
                    file.write(f"{file_name} {embedding}\n")

    def __hash__(self) -> int:
        return hash(self.path)


class Repository:
    def __init__(self, directory: Path) -> None:
        self.root = Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"], cwd=directory, text=True
            ).strip()
        )
        self.directory = directory
        self.embedding_files = {}

        for file in self.list_files():
            if file.name == _TRENCH_FILE:
                self.embedding_files[file.parent] = EmbeddingsFile(self.root / file)

    def get_embedding_file(self, path: PurePath) -> EmbeddingsFile | None:
        return self.embedding_files.get(path)

    def list_files(self) -> list[PurePath]:
        """List all files in the repository with the given extensions."""
        lines = subprocess.check_output(
            ["git", "ls-files"], cwd=self.root, text=True
        ).splitlines()
        return [Path(line) for line in lines]

    def read_file(self, path: PurePath) -> str:
        """Read the content of a file."""
        return (self.root / path).read_text()

    def write_file(self, path: PurePath, content: str) -> str:
        """Write the content of a file."""
        return (self.root / path).write_text(content)

    def hash_file(self, path: PurePath) -> str:
        return hashlib.sha256(self.read_file(path).encode()).hexdigest()


def serialize_embedding(embedding: Sequence[Sequence[float]]) -> str:
    """Serialize an embedding to a base64 string."""
    embedding_bytes = bytearray(len(embedding) * len(embedding[0]) * 4)
    offset = 0
    for vector in embedding:
        for value in vector:
            struct.pack_into("<f", embedding_bytes, offset, value)
            offset += 4
    return embedding_bytes.hex()


def ensure_ollama() -> None:
    """Ensure that the Ollama server is running."""
    if (
        _OLLAMA_CONTAINER_NAME
        not in subprocess.check_output(
            ["docker", "ps", "--format={{.Names}}"], text=True
        ).splitlines()
    ):
        Status.get().update("Starting Ollama")
        subprocess.check_call(
            [
                "docker",
                "run",
                "-d",
                "--gpus=all",
                "-v",
                "ollama:/root/.ollama",
                "-p",
                "11434:11434",
                "--name",
                _OLLAMA_CONTAINER_NAME,
                "ollama/ollama",
            ],
        )


@click.group(help=__doc__)
@click.option(
    "--spawn-ollama",
    default=True,
    help="Ensure that the Ollama server is running.",
)
def cli(spawn_ollama: bool) -> None:
    """CLI entrypoint."""
    if spawn_ollama:
        ensure_ollama()


def chunk_markdown_file(path: PurePath, content: str) -> Iterable[tuple[EmbeddingContext, str]]:
    """Chunk a markdown file into sections."""
    context = EmbeddingContext(
        file_path=str(path),
        title="",
        section="",
    )
    md = MarkdownIt("gfm-like", {"linkify": False})
    tree = SyntaxTreeNode(md.parse(content))

    if tree.children[0].type == "heading":
        context.title = " ".join(x.content for x in tree.children[0].walk() if x.type == "text")

    chunk = ""
    for token in md.parse(content):
        content = token.content.strip()
        if token.type in frozenset({"heading_close", "paragraph_close"}):
            chunk += "\n"
            continue

        if len(chunk) + len(content) > 1000:
            yield context, chunk
            chunk = content
        else:
            chunk += content

    if chunk:
        yield context, chunk

async def update_embeddings(directory: Path) -> None:
    repo = Repository(directory)
    updated_files = set()

    with make_status():
        embedder = OllamaEmbedder()
        await embedder.set_up()

        for path in repo.list_files():
            if path.suffix != ".md":
                continue

            embeddings_file = repo.get_embedding_file(path) or EmbeddingsFile(
                repo.root / path.with_name(_TRENCH_FILE)
            )
            current_hash = repo.hash_file(path)

            if embeddings_file is None or current_hash != embeddings_file.get_hash(
                path
            ):
                updated_files.add(embeddings_file)

                Status.get().update(f"Embedding {path}")
                embeddings = []
                for context, chunk in chunk_markdown_file(path, repo.read_file(path)):
                    embeddings.append(embedder.embed_file(
                        context, chunk,
                    ))
                embeddings_file.set_embeddings(path, current_hash, [serialize_embedding(x) for x in await asyncio.gather(*embeddings)])
                rich.print(f"{path} embedding was updated")

        for embeddings_file in updated_files:
            embeddings_file.write()


@cli.command()
@click.argument(
    "directory",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
    default=".",
)
def update(directory: Path) -> None:
    """Update embeddings."""
    asyncio.run(update_embeddings(directory))


async def query_embeddings(directory: Path, query: str) -> None:
    repo = Repository(directory)

    closest_distance = float("inf")
    closest_path = None

    with make_status():
        embedder = OllamaEmbedder()
        await embedder.set_up()

        query_vec = np.array(await embedder.embed(query)).squeeze()

        for embedding_file in repo.embedding_files.values():
            for path, embedding_hexes in embedding_file.file_embeddings.items():
                for embedding_hex in embedding_hexes:
                    embedding = np.array(
                        [
                            x[0]
                            for x in struct.iter_unpack("<f", bytes.fromhex(embedding_hex))
                        ]
                    )
                    distance = np.linalg.norm(query_vec - embedding)
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_path = embedding_file.directory / path

    if closest_path is None:
        rich.print("[red]No matches found[/red]")
        return

    rich.print(f"[green]Score[/green] ({closest_distance})", file=sys.stderr)
    rich.print(closest_path)


@cli.command()
@click.argument(
    "query",
)
@click.argument(
    "directory",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
    default=".",
)
def query(directory: Path, query: str) -> None:
    """Update embeddings."""
    asyncio.run(query_embeddings(directory, query))


if __name__ == "__main__":
    cli()

# 🪏 Entrench

Generate and store embedding of your Markdown files in your codebase.

## Why

Pre-computed embeddings are nice to have to provide context to models, so this project tries to format them in a nice `git`-friendly way.

## Set up

- `pip install .`

### Set up in a virtual environment

To install dependencies in a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

## Usage

### Update embeddings

```bash
entrench update
```

### Query the embeddings

`entrench` also provides a utility to query against the embeddings:

```bash
entrench query "Your query"
```

### Using as a pre-commit hook

To use `entrench` as a [pre-commit](https://pre-commit.com/) hook, add the following to your `.pre-commit-config.yaml`:

```yaml
  - repo: https://github.com/Slowki/entrench
    rev: <HASH_OF_THE_COMMIT>
    hooks:
      - id: entrench
```

## How it works

- First entrench hashes all the markdown files tracked by git
- Then it compares that to stored hashes and recomputed embeddings for modified files
- Then it updates `.trench` files

### File Format

`.trench` files are stored in the same directory as the Markdown files and have the following format:

```json
<FILE_NAME_1> <SHA256>
<FILE_NAME_N> <SHA256>
----
<FILE_NAME_1> <EMBEDDING_CHUNK_1>
<FILE_NAME_1> <EMBEDDING_CHUNK_N>
<FILE_NAME_N> <EMBEDDING_CHUNK_1>
<FILE_NAME_N> <EMBEDDING_CHUNK_N>
```

"""Subpackage of helpers shared across the three task packages.

Import from the specific submodule you need:

    from src.common.io import read_jsonl, write_jsonl
    from src.common.models import BASE_MODEL_NAME, load_tokenizer, ...

Each submodule pulls in only its own dependencies, so importing one of the
lightweight ones (``io``, ``eval_prefixes``, ``reward_fn``) does not require
``torch`` or ``transformers`` to be installed.
"""

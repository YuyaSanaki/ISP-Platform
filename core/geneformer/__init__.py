try:
    import sklearn
except ImportError:
    pass
try:
    import tensorflow
except ImportError:
    pass

import importlib

# Heavy submodules (torch, transformers, datasets) load on first use, so light helpers
# such as gene_converter and ortholog_loss_gate can be imported by the Web UI on their own.
_SUBMODULES = (
    "tokenizer",
    "pretrainer",
    "collator_for_classification",
    "emb_extractor",
    "in_silico_perturber",
    "in_silico_perturber_stats",
)
_EXPORTS = {
    "TranscriptomeTokenizer": "tokenizer",
    "GeneformerPretrainer": "pretrainer",
    "DataCollatorForGeneClassification": "collator_for_classification",
    "DataCollatorForCellClassification": "collator_for_classification",
    "EmbExtractor": "emb_extractor",
    "InSilicoPerturber": "in_silico_perturber",
    "InSilicoPerturberStats": "in_silico_perturber_stats",
}
__all__ = [*_SUBMODULES, *_EXPORTS]


def __getattr__(name):
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    if name in _EXPORTS:
        value = getattr(importlib.import_module(f".{_EXPORTS[name]}", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))

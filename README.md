# llm-ccg-parsing

Code for the paper "Step-by-Step CCG Parsing with Large Language Models" (EMNLP Findings 2026).

**Work in progress:** the code is being released in stages.

## Installation

```bash
pip install -e .                  # fine-tuning
pip install -e ".[inference]"     # inference with vLLM
pip install flash-attn==2.8.3 --no-build-isolation
```


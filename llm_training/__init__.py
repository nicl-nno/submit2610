'LLM-training package — GRPO wrapper over TRL for the Meno decoder LLM family.'

from llm_training.schema import TASK_CLASSES, UnifiedExample

__all__ = ["TASK_CLASSES", "UnifiedExample"]
__version__ = "0.1.0"

# : Total number of task classes (sanity constant
NUM_TASK_CLASSES = len(TASK_CLASSES)


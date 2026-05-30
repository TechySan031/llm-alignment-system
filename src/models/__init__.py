"""
Public API for the models layer.

Import from here in all training and evaluation scripts.
Never import directly from submodules in scripts — always go through
this __init__ so internal refactors do not break call sites.
"""
from src.models.loader import (
    load_base_model,
    load_model_for_inference,
    load_peft_model,
    merge_and_save_adapter,
)
from src.models.tokenizer_loader import (
    load_tokenizer,
    get_system_prompt,
    format_chat_prompt,
    SYSTEM_PROMPTS,
)
from src.models.lora_config import (
    build_lora_config,
    prepare_model_for_training,
    detect_model_family,
)
from src.models.parameter_counter import (
    count_parameters,
    print_parameter_table,
    lora_efficiency_report,
)
from src.models.memory_estimator import (
    estimate_training_vram,
    estimate_inference_vram,
    log_current_vram,
    recommend_batch_size,
)
from src.models.architecture_viz import (
    inspect_architecture,
    print_architecture_summary,
    get_layer_shapes,
)
from src.models.quantization import (
    get_bnb_config,
    is_quantization_available,
    get_compute_dtype,
)

__all__ = [
    "load_base_model",
    "load_model_for_inference",
    "load_peft_model",
    "merge_and_save_adapter",
    "load_tokenizer",
    "get_system_prompt",
    "format_chat_prompt",
    "SYSTEM_PROMPTS",
    "build_lora_config",
    "prepare_model_for_training",
    "detect_model_family",
    "count_parameters",
    "print_parameter_table",
    "lora_efficiency_report",
    "estimate_training_vram",
    "estimate_inference_vram",
    "log_current_vram",
    "recommend_batch_size",
    "inspect_architecture",
    "print_architecture_summary",
    "get_layer_shapes",
    "get_bnb_config",
    "is_quantization_available",
    "get_compute_dtype",
]
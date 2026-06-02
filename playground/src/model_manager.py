from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch


BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

SFT_ADAPTER = (
    "playground/models/sft_adapter/"
    "experiments/sft_runs/local_test/final_adapter"
)

DPO_ADAPTER = (
    "playground/models/dpo_adapter/"
    "experiments/dpo_runs/local_test/final_adapter"
)


class ModelManager:

    def __init__(self):

        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL
        )

        self.base_model = None
        self.sft_model = None
        self.dpo_model = None

    def load_base(self):

        if self.base_model is None:

            self.base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float32,
                device_map="cpu"
            )

    def load_sft(self):

        if self.sft_model is None:

            self.load_base()

            self.sft_model = PeftModel.from_pretrained(
                self.base_model,
                SFT_ADAPTER
            )

    def load_dpo(self):

        if self.dpo_model is None:

            self.load_base()

            self.dpo_model = PeftModel.from_pretrained(
                self.base_model,
                DPO_ADAPTER
            )
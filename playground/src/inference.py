# pyrefly: ignore [missing-import]
from src.model_manager import ModelManager

manager = ModelManager()


def _generate(
    model,
    tokenizer,
    prompt,
    temperature=0.7,
    max_tokens=50
):

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    input_length = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=int(max_tokens),
        temperature=float(temperature),
        do_sample=True
    )

    generated_tokens = outputs[0][input_length:]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )


def generate_base(
    prompt,
    temperature=0.7,
    max_tokens=50
):

    manager.load_base()

    return _generate(
        manager.base_model,
        manager.tokenizer,
        prompt,
        temperature,
        max_tokens
    )


def generate_sft(
    prompt,
    temperature=0.7,
    max_tokens=50
):

    manager.load_sft()

    return _generate(
        manager.sft_model,
        manager.tokenizer,
        prompt,
        temperature,
        max_tokens
    )


def generate_dpo(
    prompt,
    temperature=0.7,
    max_tokens=50
):

    manager.load_dpo()

    return _generate(
        manager.dpo_model,
        manager.tokenizer,
        prompt,
        temperature,
        max_tokens
    )
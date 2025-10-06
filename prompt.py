import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from nnsight import LanguageModel
from pydantic import BaseModel
from typing import Literal

class Model(BaseModel):
    id: str
    type: Literal["open_source", "openai"]
    parent_model: "Model | None" = None


def run_with_adapter():
    """Run inference with base model + LoRA adapter"""
    # Load model from JSON
    model_path = Path("model_config.json")
    with open(model_path) as f:
        model_data = json.load(f)

    model_info = Model(**model_data)

    # Load base model and adapter
    print(f"Loading base model: {model_info.parent_model.id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_info.parent_model.id,
        device_map="auto",
    )

    print(f"Loading adapter: {model_info.id}")
    model_with_adapter = PeftModel.from_pretrained(base_model, model_info.id)
    tokenizer = AutoTokenizer.from_pretrained(model_info.id)

    # Wrap with nnsight
    print("Wrapping model with nnsight...")
    llm = LanguageModel(model_with_adapter, tokenizer=tokenizer, dispatch=True)
    prompt = """Some prompt"""
    with llm.generate(prompt) as gen:
        saved = llm.generator.output.save()

    print("\nModel with adapter output:")
    for seq in saved:
        print(llm.tokenizer.decode(seq, skip_special_tokens=True))


if __name__ == "__main__":
    print("="*60)
    print("Running base model...")
    print("="*60)
    run_with_adapter()

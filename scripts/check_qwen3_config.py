"""Check that Qwen3 config/tokenizer can be reached on a prepared host."""

from transformers import AutoConfig, AutoTokenizer


def main() -> None:
    print("Checking tokenizer...")
    AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Base", use_fast=True)
    print("Checking model config...")
    config = AutoConfig.from_pretrained("Qwen/Qwen3-4B-Base")
    print(f"  hidden_size = {config.hidden_size}, num_hidden_layers = {config.num_hidden_layers}")
    print(
        f"  num_attention_heads = {config.num_attention_heads}, "
        f"num_key_value_heads = {config.num_key_value_heads}"
    )


if __name__ == "__main__":
    main()

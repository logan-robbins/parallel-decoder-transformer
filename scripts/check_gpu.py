"""Print Torch and CUDA visibility for a prepared host."""

import torch


def main() -> None:
    print("torch:", torch.__version__)
    print("cuda.is_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU[{i}]: {torch.cuda.get_device_name(i)} {props.total_memory / 1e9:.1f} GB")


if __name__ == "__main__":
    main()

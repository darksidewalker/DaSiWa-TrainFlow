import argparse
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file


ANIMA_METADATA = {
    "ss_base_model_version": "anima-base-v1.0",
    "ss_model_family": "anima",
    "ss_model_name": "anima-base-v1.0",
    "ss_architecture": "anima-base-v1.0/lora",
    "modelspec.architecture": "anima-base-v1.0/lora",
    "modelspec.implementation": "https://huggingface.co/circlestone-labs/Anima",
}


def read_metadata(path: Path) -> dict[str, str]:
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.metadata() or {}


def print_metadata(path: Path) -> None:
    metadata = read_metadata(path)
    keys = [
        "ss_base_model_version",
        "ss_model_family",
        "ss_model_name",
        "ss_architecture",
        "ss_network_module",
        "modelspec.architecture",
        "modelspec.implementation",
        "modelspec.title",
    ]
    for key in keys:
        print(f"{key}: {metadata.get(key, '')}")


def fix_metadata(path: Path, output: Path) -> None:
    tensors = load_file(path)
    metadata = read_metadata(path)
    metadata.update(ANIMA_METADATA)
    metadata.setdefault("ss_network_module", "networks.lora_anima")
    save_file(tensors, output, metadata=metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or fix Anima LoRA safetensors metadata.")
    parser.add_argument("file", type=Path)
    parser.add_argument("--fix", action="store_true", help="write Anima metadata into a safetensors copy")
    parser.add_argument("--output", type=Path, help="output path for --fix")
    args = parser.parse_args()

    if args.fix:
        output = args.output or args.file.with_name(args.file.stem + ".anima" + args.file.suffix)
        fix_metadata(args.file, output)
        print(f"wrote {output}")
        print_metadata(output)
        return

    print_metadata(args.file)


if __name__ == "__main__":
    main()

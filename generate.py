# Copyright © 2026 Paradigma / MLX Community
# CLI Inference script for Limite 1B - Violetto in MLX

import argparse
import sys
from pathlib import Path

import mlx_lm

CANONICAL_SYSTEM_PROMPT = (
    "You are a helpful assistant.\n"
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def format_prompt(problem: str, raw: bool = False) -> str:
    if raw:
        return problem

    # ChatML formatting matching Paradigma chat_template.jinja
    return (
        f"<|im_start|>system\n{CANONICAL_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{problem}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with Limite 1B Violetto on Apple Silicon via MLX"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx_model",
        help="Path to converted MLX model directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Mathematical problem or query to solve",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature (recommended: 0.6)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling top-p (recommended: 0.95)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum generation tokens",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Do not wrap prompt in canonical ChatML math template",
    )

    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(
            f"Error: Model path '{model_path}' not found.\n"
            f"Please run 'python convert.py --mlx-path {args.model}' first."
        )
        sys.exit(1)

    print(f"[*] Loading model and tokenizer from {model_path}...")
    model, tokenizer = mlx_lm.load(str(model_path))

    formatted_input = format_prompt(args.prompt, raw=args.raw)
    print("\n--- Problem Prompt ---")
    print(args.prompt)
    print("\n--- Limite 1B Solution ---")

    gen_kwargs = {
        "temp": args.temperature,
        "top_p": args.top_p,
    }

    response_stream = mlx_lm.stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=formatted_input,
        max_tokens=args.max_tokens,
        **gen_kwargs,
    )

    stats = None
    for response in response_stream:
        sys.stdout.write(response.text)
        sys.stdout.flush()
        stats = response

    print("\n" + "=" * 50)
    if stats is not None:
        print(
            f"Prompt tokens: {stats.prompt_tokens} ({stats.prompt_tps:.1f} tok/s) | "
            f"Generation tokens: {stats.generation_tokens} ({stats.generation_tps:.1f} tok/s)"
        )


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def plot_grid(grid_dir: str, prompt: str, id_lambdas: list[float], l2_lambdas: list[float], output_path: str):
    grid_dir = Path(grid_dir)
    fig, axs = plt.subplots(len(id_lambdas), len(l2_lambdas), figsize=(len(l2_lambdas) * 4, len(id_lambdas) * 4))

    if len(id_lambdas) == 1:
        axs = [axs]
    if len(l2_lambdas) == 1:
        axs = [[ax] for ax in axs]

    for i, id_lambda in enumerate(id_lambdas):
        for j, l2_lambda in enumerate(l2_lambdas):
            run_name = f"{prompt.replace(' ', '_')}_id{id_lambda}_l2{l2_lambda}"
            image_path = grid_dir / run_name / f"{prompt.replace(' ', '_')}.png"
            ax = axs[i][j]

            if image_path.exists():
                ax.imshow(Image.open(image_path))
                ax.set_title(f"ID: {id_lambda}, L2: {l2_lambda}", fontsize=10)
            else:
                ax.set_facecolor("gray")
                ax.set_title("No image", fontsize=10)

            ax.axis("off")

    plt.suptitle(f'Prompt: "{prompt}"', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-dir", default="generated_grid")
    parser.add_argument("--prompt", default="a person with bright green hair")
    parser.add_argument("--output-path", default="generated_grid/grid.png")
    args = parser.parse_args()

    plot_grid(
        grid_dir=args.grid_dir,
        prompt=args.prompt,
        id_lambdas=[0.01, 0.05, 0.1],
        l2_lambdas=[0.001, 0.005, 0.01],
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()

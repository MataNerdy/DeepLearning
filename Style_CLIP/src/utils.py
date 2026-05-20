from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image


def tensor_to_image_array(image: torch.Tensor):
    """Convert StyleGAN tensor in [-1, 1] to a matplotlib-friendly RGB array."""
    return (image[0].detach().cpu().clamp(-1, 1).permute(1, 2, 0) + 1) / 2


def save_generated_image(image: torch.Tensor, path: str | Path) -> None:
    """Save StyleGAN tensor in [-1, 1] as image file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    final_img = (image[0].detach().cpu().clamp(-1, 1) + 1) / 2
    save_image(final_img, str(path))


def plot_losses(losses: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(losses["all"], label="Total Loss")
    plt.plot(losses["clip"], label="CLIP Loss")
    plt.plot(losses["id"], label="ID Loss")
    plt.plot(losses["l2"], label="L2 Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss value")
    plt.title("Loss Dynamics during Optimization")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_progress(generated_img: torch.Tensor, target_img: torch.Tensor, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(tensor_to_image_array(generated_img))
    axs[0].set_title("Edited image")
    axs[0].axis("off")

    axs[1].imshow(tensor_to_image_array(target_img))
    axs[1].set_title("Source image")
    axs[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

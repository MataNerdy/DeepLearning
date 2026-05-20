import argparse
import csv
import os
import sys
from pathlib import Path

import torch

from losses import CLIPLoss, IDLoss
from utils import plot_losses, plot_progress, save_generated_image


def add_stylegan_to_path(stylegan_dir: str) -> None:
    stylegan_path = Path(stylegan_dir).resolve()
    if not stylegan_path.exists():
        raise FileNotFoundError(
            f"StyleGAN2 repo not found: {stylegan_path}. "
            "Clone it first: git clone https://github.com/rosinality/stylegan2-pytorch.git"
        )
    sys.path.insert(0, str(stylegan_path))


def load_generator(stylegan_dir: str, weights_path: str, device: str, size: int = 1024, latent_dim: int = 512):
    add_stylegan_to_path(stylegan_dir)
    from model import Generator

    generator = Generator(size=size, style_dim=latent_dim, n_mlp=8).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    generator.load_state_dict(state_dict["g_ema"])
    generator.eval()
    return generator


def init_latent(generator, device: str, seed: int = 3456, latent_dim: int = 512):
    torch.manual_seed(seed)
    latent_z = torch.randn(10000, latent_dim, device=device)
    latent_z_generated = latent_z.mean(0, keepdim=True)

    with torch.no_grad():
        latent_w = generator.style(latent_z_generated)
        target_img, _ = generator([latent_w], input_is_latent=True, randomize_noise=False)

    return latent_z_generated, latent_w, target_img


def optimize_latent(
    generator,
    clip_loss,
    id_loss,
    latent_w,
    target_img,
    prompt: str,
    output_dir: str,
    num_steps: int = 200,
    lr: float = 0.08,
    id_lambda: float = 0.06,
    l2_lambda: float = 0.0045,
    log_every: int = 50,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    latent = latent_w.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([latent], lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    losses = {"id": [], "l2": [], "clip": [], "all": []}

    for step in range(num_steps):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        optimizer.zero_grad()
        generated_img, _ = generator([latent], input_is_latent=True, randomize_noise=False)

        loss_clip = clip_loss(generated_img, prompt)
        loss_l2 = (latent - latent_w).pow(2).sum()
        loss_id = id_loss(generated_img, target_img)
        loss = loss_clip + l2_lambda * loss_l2 + id_lambda * loss_id

        loss.backward()
        optimizer.step()
        scheduler.step()

        losses["clip"].append(float(loss_clip.item()))
        losses["l2"].append(float(loss_l2.item()))
        losses["id"].append(float(loss_id.item()))
        losses["all"].append(float(loss.item()))

        if step % log_every == 0 or step == num_steps - 1:
            print(
                f"[{step}/{num_steps}] "
                f"Loss: {loss.item():.4f} | "
                f"CLIP: {loss_clip.item():.4f} | "
                f"ID: {loss_id.item():.4f} | "
                f"L2: {loss_l2.item():.4f}"
            )
            plot_progress(generated_img, target_img, output_dir / f"progress_step_{step:04d}.png")

    safe_prompt = prompt.replace(" ", "_").replace("/", "_")
    save_generated_image(generated_img, output_dir / f"{safe_prompt}.png")
    torch.save(latent.detach().cpu(), output_dir / f"{safe_prompt}.pt")
    plot_losses(losses, output_dir / "losses.png")

    with open(output_dir / "losses.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "total_loss", "clip_loss", "id_loss", "l2_loss"])
        for i in range(len(losses["all"])):
            writer.writerow([i, losses["all"][i], losses["clip"][i], losses["id"][i], losses["l2"][i]])

    return losses


def run_grid_search(generator, clip_loss, id_loss, latent_w, target_img, output_dir: str, num_steps: int):
    prompts = [
        "a person with bright green hair",
        "a person with fully dyed green hair",
    ]
    id_lambdas = [0.01, 0.05, 0.1]
    l2_lambdas = [0.001, 0.005, 0.01]

    results = []
    grid_dir = Path(output_dir)
    grid_dir.mkdir(parents=True, exist_ok=True)

    for prompt in prompts:
        for id_lambda in id_lambdas:
            for l2_lambda in l2_lambdas:
                run_name = f"{prompt.replace(' ', '_')}_id{id_lambda}_l2{l2_lambda}"
                print(f"\n=== {run_name} ===")
                run_out = grid_dir / run_name
                optimize_latent(
                    generator=generator,
                    clip_loss=clip_loss,
                    id_loss=id_loss,
                    latent_w=latent_w,
                    target_img=target_img,
                    prompt=prompt,
                    output_dir=run_out,
                    num_steps=num_steps,
                    id_lambda=id_lambda,
                    l2_lambda=l2_lambda,
                    log_every=num_steps + 1,
                )
                results.append([run_name, prompt, id_lambda, l2_lambda])

    with open(grid_dir / "grid_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run_name", "prompt", "id_lambda", "l2_lambda"])
        writer.writerows(results)


def parse_args():
    parser = argparse.ArgumentParser(description="Text-guided face editing with StyleGAN2, CLIP and ArcFace.")
    parser.add_argument("--stylegan-dir", default="stylegan2-pytorch")
    parser.add_argument("--stylegan-weights", required=True)
    parser.add_argument("--arcface-weights", required=True)
    parser.add_argument("--prompt", default="a person with bright green hair")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--id-lambda", type=float, default=0.06)
    parser.add_argument("--l2-lambda", type=float, default=0.0045)
    parser.add_argument("--seed", type=int, default=3456)
    parser.add_argument("--grid-search", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    generator = load_generator(args.stylegan_dir, args.stylegan_weights, device)
    _, latent_w, target_img = init_latent(generator, device=device, seed=args.seed)

    clip_loss = CLIPLoss(device=device)
    id_loss = IDLoss(args.arcface_weights, device=device)

    if args.grid_search:
        run_grid_search(
            generator=generator,
            clip_loss=clip_loss,
            id_loss=id_loss,
            latent_w=latent_w,
            target_img=target_img,
            output_dir=args.output_dir,
            num_steps=args.num_steps,
        )
    else:
        optimize_latent(
            generator=generator,
            clip_loss=clip_loss,
            id_loss=id_loss,
            latent_w=latent_w,
            target_img=target_img,
            prompt=args.prompt,
            output_dir=args.output_dir,
            num_steps=args.num_steps,
            lr=args.lr,
            id_lambda=args.id_lambda,
            l2_lambda=args.l2_lambda,
        )


if __name__ == "__main__":
    main()

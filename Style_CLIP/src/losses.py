import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

from arcface import Backbone


class CLIPLoss(nn.Module):
    """CLIP-based text-image loss.

    Minimizes cosine distance between image and text embeddings.
    """

    def __init__(self, device: str = "cuda", stylegan_size: int = 1024, clip_model: str = "ViT-B/32"):
        super().__init__()
        self.model, _ = clip.load(clip_model, device=device)
        self.upsample = nn.Upsample(scale_factor=7)
        self.avg_pool = nn.AvgPool2d(kernel_size=stylegan_size // 32)

    def forward(self, image: torch.Tensor, text: str) -> torch.Tensor:
        image = self.avg_pool(self.upsample(image)).to(image.device)
        image_features = self.model.encode_image(image)

        text_tokens = clip.tokenize([text]).to(image.device)
        text_features = self.model.encode_text(text_tokens)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        similarity = (image_features @ text_features.T).squeeze()
        return 1 - similarity


class IDLoss(nn.Module):
    """ArcFace-based identity preservation loss."""

    def __init__(self, model_weights: str, device: str = "cuda"):
        super().__init__()
        print("Loading ResNet ArcFace")
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode="ir_se")
        self.facenet.load_state_dict(torch.load(model_weights, map_location=device))
        self.pool = nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval().to(device)

    def extract_feats(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] != 256:
            x = self.pool(x)

        # Center crop used in the original StyleCLIP helper.
        x = x[:, :, 35:223, 32:220]
        x = self.face_pool(x)
        return self.facenet(x)

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n_samples = y.shape[0]
        y_feats = self.extract_feats(y).detach()
        y_hat_feats = self.extract_feats(y_hat)

        loss = 0.0
        for i in range(n_samples):
            sim = torch.dot(y_feats[i], y_hat_feats[i])
            loss += 1 - sim

        return loss / n_samples

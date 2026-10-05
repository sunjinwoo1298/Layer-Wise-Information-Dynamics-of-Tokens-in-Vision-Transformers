"""
ViT/DeiT Model Wrapper for Layer-Wise Token Extraction and Intercepted Forward Passes.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import timm


@dataclass
class ModelMetadata:
    """Metadata detailing the Vision Transformer architecture and weights."""
    model_name: str
    pretrained: bool
    num_layers: int
    hidden_dim: int
    num_heads: int
    num_original_tokens: int
    num_classes: int
    patch_size: Tuple[int, int]
    img_size: Tuple[int, int]
    device: str

    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_original_tokens": self.num_original_tokens,
            "num_classes": self.num_classes,
            "patch_size": list(self.patch_size),
            "img_size": list(self.img_size),
            "device": self.device,
        }


class ViTWrapper(nn.Module):
    """
    Modular wrapper around a Vision Transformer (timm / HuggingFace) that provides:
    1. Standard full forward pass.
    2. Partial forward passes (embedding -> layer l, and layer l -> classification head).
    3. Intermediate token interception (decoupling CLS prefix from visual patch tokens).
    4. Exact baseline parity guarantee.
    """

    def __init__(
        self,
        model_name: str = "deit_tiny_patch16_224",
        pretrained: bool = True,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        
        # Select device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load backbone
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.to(self.device)
        self.model.eval()

        # Architecture introspection
        self.num_layers = len(self.model.blocks)
        self.hidden_dim = self.model.embed_dim
        self.num_heads = self.model.blocks[0].attn.num_heads
        
        # Patch & Image sizes
        if hasattr(self.model.patch_embed, "patch_size"):
            ps = self.model.patch_embed.patch_size
            self.patch_size = ps if isinstance(ps, tuple) else (ps, ps)
        else:
            self.patch_size = (16, 16)

        if hasattr(self.model.patch_embed, "img_size"):
            ims = self.model.patch_embed.img_size
            self.img_size = ims if isinstance(ims, tuple) else (ims, ims)
        else:
            self.img_size = (224, 224)

        self.num_original_tokens = self.model.patch_embed.num_patches
        self.num_classes = getattr(self.model, "num_classes", 1000)

        # Distillation token check (DeiT models with distillation have 2 prefix tokens: cls + dist)
        self.num_prefix_tokens = getattr(self.model, "num_prefix_tokens", 1)

        self.metadata = ModelMetadata(
            model_name=self.model_name,
            pretrained=self.pretrained,
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_original_tokens=self.num_original_tokens,
            num_classes=self.num_classes,
            patch_size=self.patch_size,
            img_size=self.img_size,
            device=str(self.device),
        )

    def embed_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed input image tensor into initial token sequence (Prefix [CLS] + Patches + PosEmbed).
        x: (B, C, H, W)
        returns: (B, num_prefix + N, d)
        """
        x = x.to(self.device)
        # Use timm's internal embedding methods
        if hasattr(self.model, "_pos_embed"):
            x = self.model.patch_embed(x)
            x = self.model._pos_embed(x)
            x = self.model.norm_pre(x) if hasattr(self.model, "norm_pre") else x
            x = self.model.pos_drop(x) if hasattr(self.model, "pos_drop") else x
        elif hasattr(self.model, "prepare_tokens"):
            x = self.model.prepare_tokens(x)
        else:
            # Fallback standard timm ViT structure
            x = self.model.patch_embed(x)
            cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = x + self.model.pos_embed
            x = self.model.pos_drop(x)
        return x

    def forward_layers(
        self,
        tokens: torch.Tensor,
        start_layer: int,
        end_layer: int,
    ) -> torch.Tensor:
        """
        Pass tokens sequentially through blocks from start_layer (inclusive) to end_layer (exclusive).
        start_layer: 0-indexed
        end_layer: 0-indexed (up to num_layers)
        """
        for i in range(start_layer, end_layer):
            tokens = self.model.blocks[i](tokens)
        return tokens

    def forward_head(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Pass final tokens through LayerNorm and classification head.
        tokens: (B, num_prefix + k, d)
        returns: (B, num_classes)
        """
        if hasattr(self.model, "norm") and self.model.norm is not None:
            tokens = self.model.norm(tokens)

        if hasattr(self.model, "forward_head"):
            return self.model.forward_head(tokens)
        
        # Fallback manual head execution
        if self.num_prefix_tokens == 1:
            cls_out = tokens[:, 0]
        else:
            cls_out = tokens[:, 0]
        return self.model.head(cls_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard uncompressed full forward pass."""
        x = x.to(self.device)
        return self.model(x)

    def extract_tokens_at_layer(
        self,
        x: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract token representations right before entering block layer_idx (1-indexed layer).
        Layer 1 corresponds to tokens outputted after block 0.
        
        Returns:
            prefix_tokens: (B, num_prefix_tokens, d) - e.g. [CLS] token
            visual_tokens: (B, N, d) - visual patch tokens
        """
        assert 1 <= layer_idx <= self.num_layers, f"Layer index {layer_idx} out of range [1, {self.num_layers}]"
        
        with torch.no_grad():
            tokens = self.embed_input(x)
            # Run through blocks 0 to layer_idx - 1
            tokens = self.forward_layers(tokens, 0, layer_idx)
            
            prefix_tokens = tokens[:, :self.num_prefix_tokens, :]
            visual_tokens = tokens[:, self.num_prefix_tokens:, :]
            return prefix_tokens, visual_tokens

    def forward_from_pruned_tokens(
        self,
        prefix_tokens: torch.Tensor,
        retained_visual_tokens: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Recombines prefix tokens with retained visual tokens at layer_idx and completes
        the forward pass through the remaining transformer layers.
        
        prefix_tokens: (B, num_prefix, d)
        retained_visual_tokens: (B, k, d)
        layer_idx: The 1-indexed layer after which pruning was applied.
                   Remaining blocks executed: [layer_idx, num_layers).
        """
        assert 1 <= layer_idx <= self.num_layers, f"Layer {layer_idx} out of bounds"
        
        # Concatenate prefix + retained visual tokens
        combined_tokens = torch.cat([prefix_tokens, retained_visual_tokens], dim=1)
        
        if layer_idx < self.num_layers:
            combined_tokens = self.forward_layers(combined_tokens, layer_idx, self.num_layers)
            
        logits = self.forward_head(combined_tokens)
        return logits

"""NR-GLADNet formal model implementation.



This file keeps the first-paper LiteGLAMFusionNetV2 model and the

minimal CNN, MB-MSA, and fusion components required by that model.

"""



from __future__ import annotations



from pathlib import Path

from typing import Any



import torch

import torch.nn as nn

import torch.nn.functional as F



__all__ = ["LiteGLAMFusionNetV2"]



# CNN building blocks

class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

def _valid_group_count(num_channels: int, max_groups: int) -> int:
    max_groups = max(1, min(int(max_groups), int(num_channels)))
    for groups in range(max_groups, 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1

class IBN2d(nn.Module):
    """Split channels into InstanceNorm and BatchNorm partitions."""

    def __init__(self, num_channels: int):
        super().__init__()
        if num_channels <= 0:
            raise ValueError("num_channels must be > 0.")
        self.in_channels = int(num_channels) // 2
        self.bn_channels = int(num_channels) - self.in_channels
        self.instance_norm = (
            nn.InstanceNorm2d(self.in_channels, affine=True, track_running_stats=False)
            if self.in_channels > 0 else None
        )
        self.batch_norm = nn.BatchNorm2d(self.bn_channels) if self.bn_channels > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.instance_norm is None:
            return self.batch_norm(x)
        if self.batch_norm is None:
            return self.instance_norm(x)
        first, second = torch.split(x, [self.in_channels, self.bn_channels], dim=1)
        return torch.cat([self.instance_norm(first), self.batch_norm(second)], dim=1)

def build_norm2d(num_channels: int, norm_type: str, num_groups: int = 8) -> nn.Module:
    norm_type = str(norm_type).lower()
    if norm_type == "bn":
        return nn.BatchNorm2d(num_channels)
    if norm_type == "gn":
        return nn.GroupNorm(_valid_group_count(num_channels, num_groups), num_channels=num_channels)
    if norm_type == "ln":
        return LayerNorm2d(num_channels)
    if norm_type == "in":
        return nn.InstanceNorm2d(num_channels, affine=True, track_running_stats=False)
    if norm_type == "ibn":
        return IBN2d(num_channels)
    raise ValueError("norm_type must be one of: bn, gn, ln, in, ibn")

class FusedMBConvLite(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 2,
        norm_type: str = "ibn",
    ):
        super().__init__()
        hidden_channels = int(in_channels * expansion)
        self.use_residual = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            build_norm2d(hidden_channels, norm_type),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            build_norm2d(out_channels, norm_type),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        return x + out if self.use_residual else out

class ConvNeXtLiteBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 4, norm_type: str = "ln"):
        super().__init__()
        hidden_channels = int(channels * expansion)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels),
            build_norm2d(channels, norm_type),
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)



# MB-MSA building blocks

class TokenBatchNorm1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x):
        if x.ndim == 2:
            return self.norm(x)
        if x.ndim == 3:
            return self.norm(x.transpose(1, 2)).transpose(1, 2)
        raise ValueError(f"Expected [B,C] or [B,N,C], got {tuple(x.shape)}")

def build_token_norm(channels: int, norm_type: str = "ln") -> nn.Module:
    norm_type = str(norm_type).lower()
    if norm_type == "ln":
        return nn.LayerNorm(channels)
    if norm_type == "bn":
        return TokenBatchNorm1d(channels)
    raise ValueError("norm_type must be one of: ln, bn")

class MBConvBlock(nn.Module):
    """
    MBConv Block for local feature extraction.

    Structure: DW Conv → BN → GELU → PW Conv → BN → SE → Residual

    Args:
        channels: input/output channel dimension
        reduction: SE block reduction ratio
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden_channels = channels * reduction

        # Depthwise convolution (local spatial features)
        self.dw_conv = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1,
            groups=channels, bias=False
        )
        self.dw_norm = nn.BatchNorm2d(channels)

        # Pointwise convolution (channel mixing)
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.pw_norm = nn.BatchNorm2d(channels)

        # SE Block (channel attention)
        self.se_reduce = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.se_expand = nn.Conv2d(hidden_channels, channels, kernel_size=1)

        self.act = nn.GELU()

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W) local features
        """
        residual = x

        # Depthwise convolution
        x = self.dw_conv(x)
        x = self.dw_norm(x)
        x = self.act(x)

        # Pointwise convolution
        x = self.pw_conv(x)
        x = self.pw_norm(x)

        # SE attention
        se = F.adaptive_avg_pool2d(x, 1)
        se = self.act(self.se_reduce(se))
        se = torch.sigmoid(self.se_expand(se))
        x = x * se

        return residual + x

class GlobalBranch(nn.Module):
    """
    Global attention branch with dual pooling downsampling.

    Structure: DualPool → Self-Attention → Bilinear Upsample

    Args:
        channels: input channel dimension
        num_heads: number of attention heads
        window_size: downsample size for global pooling
        dropout: attention dropout
    """

    def __init__(self, channels, num_heads=3, window_size=7, dropout=0.0):
        super().__init__()
        self.window_size = window_size
        self.channels = channels

        # Dual pooling downsampling
        self.pool_h = nn.AdaptiveMaxPool2d((window_size, None))
        self.pool_w = nn.AdaptiveAvgPool2d((None, window_size))

        # Self-attention
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W) global features (same resolution as input)
        """
        B, C, H, W = x.shape
        residual = x

        # Dual pooling: max + avg, then combine
        # Using adaptive pooling to reduce spatial dims
        size = (self.window_size, self.window_size)
        x_max = F.adaptive_max_pool2d(x, size)  # (B, C, gw, gw)
        x_avg = F.adaptive_avg_pool2d(x, size)  # (B, C, gw, gw)
        x_pooled = x_max + x_avg  # (B, C, gw, gw)

        # Reshape for attention: (B, gw*gw, C)
        gw = self.window_size
        x_attn = x_pooled.flatten(2).transpose(1, 2)  # (B, gw*gw, C)

        # Self-attention
        x_attn = self.norm(x_attn)
        qkv = self.qkv(x_attn).reshape(B, gw * gw, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_attn = (attn @ v).transpose(1, 2).reshape(B, gw * gw, C)
        x_attn = self.proj(x_attn)
        x_attn = self.proj_drop(x_attn)

        # Reshape back to (B, C, gw, gw)
        x_attn = x_attn.transpose(1, 2).reshape(B, C, gw, gw)

        # Bilinear upsample to original resolution
        x_attn = F.interpolate(x_attn, size=(H, W), mode='bilinear', align_corners=False)

        return x_attn

class MBMSA(nn.Module):
    """
    MB-Multi-Head Self-Attention module.

    Combines local MBConv and global self-attention in parallel.

    Structure:
        Input → Split channels → [MBConv (local) || GlobalAttention (global)] → Concat

    Args:
        channels: input channel dimension
        num_heads: number of attention heads for global attention
        window_size: downsample size for global branch
        reduction: SE block reduction ratio
        dropout: dropout rate
    """

    def __init__(self, channels, num_heads=3, window_size=7, reduction=4, dropout=0.0):
        super().__init__()
        self.channels = channels

        # Channel split: half for local, half for global
        self.local_channels = channels // 2
        self.global_channels = channels - self.local_channels

        # Local branch: MBConv
        self.local_branch = MBConvBlock(self.local_channels, reduction)

        # Global branch: Self-Attention with dual pooling
        self.global_branch = GlobalBranch(
            self.global_channels, num_heads, window_size, dropout
        )

    def forward(self, x, H=None, W=None):
        """
        Args:
            x: (B, N, C) - sequence of patch tokens
            H: spatial height (optional, computed from N if None)
            W: spatial width (optional, computed from N if None)
        Returns:
            (B, N, C) - fused local + global features
        """
        B, N, C = x.shape

        # Compute spatial dimensions if not provided
        if H is None or W is None:
            # For non-square patches, try to find factors
            # Common AST sizes: 12x101, 16x16, etc.
            H = W = int(N ** 0.5)
            # If not perfect square, use the pre-computed values
            if H * W != N:
                # Store H, W during first forward pass or compute from patch info
                # For now, assume H*W = N with H being the smaller dimension
                H = int(N ** 0.5)
                W = N // H

        # Reshape to spatial: (B, C, H, W)
        x_spatial = x.transpose(1, 2).reshape(B, C, H, W)

        # Split channels
        x_local = x_spatial[:, :self.local_channels, :, :]  # (B, C/2, H, W)
        x_global = x_spatial[:, self.local_channels:, :, :]  # (B, C/2, H, W)

        # Process local and global branches
        x_local = self.local_branch(x_local)  # (B, C/2, H, W)
        x_global = self.global_branch(x_global)  # (B, C/2, H, W)

        # Concatenate along channel dim
        x_fused = torch.cat([x_local, x_global], dim=1)  # (B, C, H, W)

        # Reshape back to sequence: (B, N, C)
        x_fused = x_fused.flatten(2).transpose(1, 2)

        return x_fused

class MBMSABlock(nn.Module):
    """
    Transformer Block with MB-MSA replacing standard attention.

    Structure:
        z'_l = MB-MSA(LN(z_{l-1})) + z_{l-1}
        z_l = MLP(LN(z'_l)) + z'_l

    Args:
        channels: embedding dimension
        num_heads: number of attention heads
        mlp_ratio: MLP hidden dim ratio
        window_size: downsample size for global branch in MB-MSA
        reduction: SE block reduction ratio
        dropout: dropout rate
    """

    def __init__(self, channels, num_heads=3, mlp_ratio=4.0,
                 window_size=7, reduction=4, dropout=0.0, norm_type="ln"):
        super().__init__()

        # MB-MSA (replaces standard multi-head self-attention)
        self.norm1 = build_token_norm(channels, norm_type)
        self.mb_msa = MBMSA(channels, num_heads, window_size, reduction, dropout)

        # MLP
        self.norm2 = build_token_norm(channels, norm_type)
        mlp_hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x, H=None, W=None):
        """
        Args:
            x: (B, N, C)
            H: spatial height
            W: spatial width
        Returns:
            (B, N, C)
        """
        # MB-MSA with pre-norm and residual
        x = x + self.mb_msa(self.norm1(x), H, W)
        # MLP with pre-norm and residual
        x = x + self.mlp(self.norm2(x))
        return x



# LiteGLAMFusionNetV2 components

def _align_spatial_map(x: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    if tuple(x.shape[-2:]) == tuple(target_hw):
        return x
    if x.shape[-2] >= target_hw[0] and x.shape[-1] >= target_hw[1]:
        return F.adaptive_avg_pool2d(x, target_hw)
    return F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)

def _init_identity_1x1(conv: nn.Conv2d) -> None:
    if conv.in_channels != conv.out_channels:
        return
    nn.init.zeros_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
    with torch.no_grad():
        idx = torch.arange(conv.in_channels)
        conv.weight[idx, idx, 0, 0] = 1.0

class GAM2d(nn.Module):
    """Global Attention Mechanism for 2D feature maps."""

    def __init__(self, channels: int, reduction: int = 4, spatial_kernel: int = 7):
        super().__init__()
        hidden_channels = max(8, channels // reduction)
        padding = spatial_kernel // 2
        self.channel_attn = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, channels),
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=spatial_kernel, padding=padding, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=spatial_kernel, padding=padding, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        channel_gate = self.channel_attn(x.permute(0, 2, 3, 1).reshape(b * h * w, c))
        channel_gate = torch.sigmoid(channel_gate).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x * channel_gate
        spatial_gate = torch.sigmoid(self.spatial_attn(x))
        return x * spatial_gate

class DTNR(nn.Module):
    """Dynamic Threshold Noise Reduction for input spectrograms."""

    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        mid_channels = max(in_channels // 2, 16)
        hidden = max(in_channels // reduction, 8)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.threshold_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        alpha = self.threshold_net(feat)
        mean_val = feat.abs().mean(dim=(2, 3), keepdim=True)
        tau = alpha.unsqueeze(-1).unsqueeze(-1) * mean_val
        y = torch.where(feat > tau, feat - tau, torch.where(feat < -tau, feat + tau, torch.zeros_like(feat)))
        return x + y

class GAMCNNBranch(nn.Module):
    """Local CNN branch with GAM replacing the earlier global perception block."""

    def __init__(
        self,
        in_chans: int = 1,
        width: int = 64,
        use_convnext_lite: bool = True,
        cnn_norm_type: str = "ibn",
        convnext_norm_type: str = "ln",
        gam_reduction: int = 4,
        ablate_no_gam: bool = False,
    ):
        super().__init__()
        mid_channels = width * 2
        final_channels = width * 4
        self.mid_channels = mid_channels
        self.out_channels = final_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, width, kernel_size=3, stride=2, padding=1, bias=False),
            build_norm2d(width, cnn_norm_type),
            nn.GELU(),
        )
        self.stage1 = nn.Sequential(
            FusedMBConvLite(width, width, norm_type=cnn_norm_type),
            FusedMBConvLite(width, width, norm_type=cnn_norm_type),
        )
        self.stage2 = nn.Sequential(
            FusedMBConvLite(width, mid_channels, stride=2, norm_type=cnn_norm_type),
            FusedMBConvLite(mid_channels, mid_channels, norm_type=cnn_norm_type),
        )
        stage3_blocks: list[nn.Module] = [
            FusedMBConvLite(mid_channels, final_channels, stride=2, norm_type=cnn_norm_type)
        ]
        if use_convnext_lite:
            stage3_blocks.extend([
                ConvNeXtLiteBlock(final_channels, norm_type=convnext_norm_type),
                ConvNeXtLiteBlock(final_channels, norm_type=convnext_norm_type),
            ])
        else:
            stage3_blocks.extend(
                [
                    FusedMBConvLite(final_channels, final_channels, norm_type=cnn_norm_type),
                    FusedMBConvLite(final_channels, final_channels, norm_type=cnn_norm_type),
                ]
            )
        self.stage3 = nn.Sequential(*stage3_blocks)
        self.ablate_no_gam = ablate_no_gam
        if not ablate_no_gam:
            self.gam = GAM2d(final_channels, reduction=gam_reduction)
        else:
            self.gam = nn.Identity()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        mid_map = self.stage2(x)
        final_map = self.gam(self.stage3(mid_map))
        return mid_map, final_map

class MidFeatureMapFusion(nn.Module):
    """Fuse CNN mid maps with AST patch maps, then return patch tokens."""

    def __init__(self, cnn_channels: int, ast_dim: int, alpha_init: float = 0.0):
        super().__init__()
        self.cnn_proj = nn.Conv2d(cnn_channels, ast_dim, kernel_size=1, bias=False)
        self.gate = nn.Conv2d(ast_dim * 2, ast_dim, kernel_size=1)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(
        self,
        patch_tokens: torch.Tensor,
        cnn_mid_map: torch.Tensor,
        patch_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, dim = patch_tokens.shape
        ast_map = patch_tokens.transpose(1, 2).reshape(batch, dim, *patch_hw)
        cnn_map = _align_spatial_map(self.cnn_proj(cnn_mid_map), patch_hw)
        gate = torch.sigmoid(self.gate(torch.cat([ast_map, cnn_map], dim=1)))
        candidate = gate * ast_map + (1.0 - gate) * cnn_map
        fused_map = ast_map + self.alpha * (candidate - ast_map)
        return fused_map.flatten(2).transpose(1, 2), gate

class FinalFeatureMapFusion(nn.Module):
    """Fuse final CNN and AST maps while preserving the AST path at initialization."""

    def __init__(self, cnn_channels: int, ast_dim: int, fusion_dim: int, alpha_init: float = 0.0):
        super().__init__()
        self.cnn_proj = nn.Conv2d(cnn_channels, fusion_dim, kernel_size=1, bias=False)
        self.ast_proj = nn.Conv2d(ast_dim, fusion_dim, kernel_size=1, bias=False)
        _init_identity_1x1(self.ast_proj)
        self.gate = nn.Conv2d(fusion_dim * 2, fusion_dim, kernel_size=1)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(
        self,
        cnn_final_map: torch.Tensor,
        ast_final_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ast_map = self.ast_proj(ast_final_map)
        cnn_map = _align_spatial_map(self.cnn_proj(cnn_final_map), tuple(ast_map.shape[-2:]))
        gate = torch.sigmoid(self.gate(torch.cat([ast_map, cnn_map], dim=1)))
        candidate = gate * ast_map + (1.0 - gate) * cnn_map
        fused_map = ast_map + self.alpha * (candidate - ast_map)
        return fused_map, gate

def _get_patch_hw(input_fdim: int, input_tdim: int, fstride: int, tstride: int) -> tuple[int, int]:
    with torch.no_grad():
        probe = torch.zeros(1, 1, input_fdim, input_tdim)
        proj = nn.Conv2d(1, 192, kernel_size=(16, 16), stride=(fstride, tstride))
        out = proj(probe)
    return int(out.shape[2]), int(out.shape[3])


class StandardTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        norm_type: str = "ln",
    ):
        super().__init__()
        self.norm1 = build_token_norm(dim, norm_type)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = build_token_norm(dim, norm_type)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + self.drop1(y)
        x = x + self.mlp(self.norm2(x))
        return x

class LiteGLAMFusionNetV2(nn.Module):
    """
    Second lightweight GLAM variant with a custom 128-dim 5-layer AST branch.

    Layer 1: Standard MSA
    Layer 2: MB-MSA
    Mid fusion after Layer 2
    Layer 3: Standard MSA
    Layer 4: MB-MSA
    Layer 5: Standard MSA
    """

    def __init__(
        self,
        label_dim: int = 2,
        input_fdim: int = 128,
        input_tdim: int = 128,
        fstride: int = 10,
        tstride: int = 10,
        imagenet_pretrain: bool = False,
        cnn_width: int = 64,
        fusion_dim: int = 256,
        embed_dim: int = 128,
        num_heads: int = 4,
        window_size: int = 7,
        mb_reduction: int = 4,
        dropout: float = 0.1,
        ast_pretrain_path: str | Path | None = None,
        use_dtnr: bool = False,
        cnn_norm_type: str = "ibn",
        convnext_norm_type: str = "ln",
        ast_norm_type: str = "ln",
        verbose: bool = False,
        # Ablation flags
        ablate_no_mb_msa: bool = False,
        ablate_no_gam: bool = False,
        ablate_simple_late_fusion: bool = False,
    ):
        super().__init__()
        if imagenet_pretrain or ast_pretrain_path is not None:
            raise ValueError("LiteGLAMFusionNetV2 uses embed_dim=128 and does not support DeiT-Tiny pretrained weights.")
        del ast_pretrain_path
        self.ast_pretrain_path = None
        self.ast_dim = int(embed_dim)
        self.mb_msa_layers = {1, 3}
        self.mid_fusion_layer = 2
        self.use_dtnr = bool(use_dtnr)
        self.dtnr = DTNR(1) if self.use_dtnr else nn.Identity()
        self.patch_fdim, self.patch_tdim = _get_patch_hw(input_fdim, input_tdim, fstride, tstride)
        num_patches = self.patch_fdim * self.patch_tdim

        self.patch_embed = nn.Conv2d(1, self.ast_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.ast_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.ast_dim))
        self.pos_drop = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        if ablate_no_mb_msa:
            # Ablation: replace all MB-MSA blocks with standard Transformer blocks
            self.blocks = nn.ModuleList([
                StandardTransformerBlock(self.ast_dim, num_heads=num_heads, dropout=dropout, norm_type=ast_norm_type)
                for _ in range(5)
            ])
            self.mb_msa_layers = set()  # no MB-MSA layers
        else:
            self.blocks = nn.ModuleList(
                [
                    StandardTransformerBlock(self.ast_dim, num_heads=num_heads, dropout=dropout, norm_type=ast_norm_type),
                    MBMSABlock(
                        channels=self.ast_dim,
                        num_heads=num_heads,
                        mlp_ratio=4.0,
                        window_size=window_size,
                        reduction=mb_reduction,
                        dropout=dropout,
                        norm_type=ast_norm_type,
                    ),
                    StandardTransformerBlock(self.ast_dim, num_heads=num_heads, dropout=dropout, norm_type=ast_norm_type),
                    MBMSABlock(
                        channels=self.ast_dim,
                        num_heads=num_heads,
                        mlp_ratio=4.0,
                        window_size=window_size,
                        reduction=mb_reduction,
                        dropout=dropout,
                        norm_type=ast_norm_type,
                    ),
                    StandardTransformerBlock(self.ast_dim, num_heads=num_heads, dropout=dropout, norm_type=ast_norm_type),
                ]
            )
        self.norm = build_token_norm(self.ast_dim, ast_norm_type)

        self.cnn_branch = GAMCNNBranch(
            in_chans=1,
            width=cnn_width,
            cnn_norm_type=cnn_norm_type,
            convnext_norm_type=convnext_norm_type,
            ablate_no_gam=ablate_no_gam,
        )
        self.mid_fusion = MidFeatureMapFusion(self.cnn_branch.mid_channels, self.ast_dim)
        self.final_fusion = FinalFeatureMapFusion(self.cnn_branch.out_channels, self.ast_dim, fusion_dim)
        self.fusion_head = nn.Sequential(
            build_token_norm(fusion_dim, ast_norm_type),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, label_dim),
        )
        self.cnn_head = nn.Linear(self.cnn_branch.out_channels, label_dim)
        self.ast_head = nn.Sequential(build_token_norm(self.ast_dim, ast_norm_type), nn.Linear(self.ast_dim, label_dim))

        # Store ablation flags
        self.ablate_simple_late_fusion = ablate_simple_late_fusion
        if ablate_simple_late_fusion:
            cnn_vec_dim = self.cnn_branch.out_channels
            self.late_fusion_head = nn.Sequential(
                nn.Linear(cnn_vec_dim + self.ast_dim, fusion_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim, label_dim),
            )

        if verbose:
            print("--------------- LiteGLAMFusionNetV2 ---------------")
            print(f"Input STFT: [B,1,{input_tdim},{input_fdim}]")
            print(f"Patch grid: {self.patch_fdim} x {self.patch_tdim}")
            print(f"AST dim: {self.ast_dim}, heads: {num_heads}, depth: 5")
            print("Layer 1/3/5: Standard Transformer")
            print("Layer 2/4: MB-MSA")
            print("CNN feature-map fusion after layer 2")
            print(f"DTNR: {'on' if self.use_dtnr else 'off'}")
            print(f"CNN norm: {cnn_norm_type}, ConvNeXt norm: {convnext_norm_type}, AST norm: {ast_norm_type}")
            print("AST pretrain path: none")

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError(f"Expected input [B,1,T,F] or [B,T,F], got {tuple(x.shape)}")
        if x.shape[1] != 1:
            raise ValueError(f"LiteGLAMFusionNetV2 expects single-channel STFT input, got C={x.shape[1]}")
        return x.transpose(2, 3).contiguous()

    def forward(self, x: torch.Tensor, return_maps: bool = False) -> torch.Tensor | dict[str, Any]:
        x_img = self._prepare_input(x)
        x_img = self.dtnr(x_img)
        cnn_mid_map, cnn_final_map = self.cnn_branch(x_img)

        batch = x_img.shape[0]
        patch_hw = (self.patch_fdim, self.patch_tdim)
        patch_tokens = self.patch_embed(x_img).flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls_token, patch_tokens], dim=1)
        tokens = self.pos_drop(tokens + self.pos_embed)

        mid_gate = None
        for idx, block in enumerate(self.blocks):
            if idx == self.mid_fusion_layer and not self.ablate_simple_late_fusion:
                cls = tokens[:, :1]
                patch_tokens, mid_gate = self.mid_fusion(tokens[:, 1:], cnn_mid_map, patch_hw)
                tokens = torch.cat([cls, patch_tokens], dim=1)
            if idx in self.mb_msa_layers:
                cls = tokens[:, :1]
                patch_tokens = block(tokens[:, 1:], *patch_hw)
                tokens = torch.cat([cls, patch_tokens], dim=1)
            else:
                tokens = block(tokens)

        tokens = self.norm(tokens)
        ast_vec = tokens[:, 0]
        ast_patch_map = tokens[:, 1:].transpose(1, 2).reshape(batch, self.ast_dim, *patch_hw)
        cnn_vec = F.adaptive_avg_pool2d(cnn_final_map, 1).flatten(1)

        if self.ablate_simple_late_fusion:
            # Simple late fusion: concat CNN and AST vectors, then linear
            fused_vec = torch.cat([cnn_vec, ast_vec], dim=1)
            fusion_logits = self.late_fusion_head(fused_vec)
            if not return_maps:
                return fusion_logits
            return {
                "logits": fusion_logits,
                "cnn_logits": self.cnn_head(cnn_vec),
                "ast_logits": self.ast_head(ast_vec),
            }
        else:
            # Default: attention-based feature-map fusion
            fused_map, final_gate = self.final_fusion(cnn_final_map, ast_patch_map)
            fused_vec = F.adaptive_avg_pool2d(fused_map, 1).flatten(1)
            fusion_logits = self.fusion_head(fused_vec)
            if not return_maps:
                return fusion_logits
            return {
                "logits": fusion_logits,
                "cnn_logits": self.cnn_head(cnn_vec),
                "ast_logits": self.ast_head(ast_vec),
                "fused_map": fused_map,
                "cnn_final_map": cnn_final_map,
                "ast_final_map": ast_patch_map,
                "mid_gate": mid_gate,
                "final_gate": final_gate,
            }

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

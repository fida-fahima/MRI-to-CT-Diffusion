
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm.auto import tqdm
import numpy as np
from .utils import Config

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.dim % 2 == 1: # Pad if dim is odd
             embeddings = F.pad(embeddings, (0,1))
        return embeddings

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_dim, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(x)
        h = self.norm1(h)
        h = h + self.time_mlp(t)[:, :, None, None]
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        return h + self.residual_conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(b, self.num_heads, c // self.num_heads, h * w).transpose(2, 3)
        k = k.view(b, self.num_heads, c // self.num_heads, h * w).transpose(2, 3)
        v = v.view(b, self.num_heads, c // self.num_heads, h * w).transpose(2, 3)
        attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(c // self.num_heads), dim=-1)
        out = (attn @ v).transpose(2, 3).reshape(b, c, h, w)
        out = self.proj(out)
        return out + x

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, condition_channels=1,
                 base_channels=56, channel_multipliers=(1, 2, 4, 6),
                 num_res_blocks=2, attention_resolutions=(16,), dropout=0.1):
        super().__init__()
        
        config = Config()
        self.in_channels = in_channels + condition_channels
        time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        channels = [base_channels] + [base_channels * m for m in channel_multipliers]
        num_levels = len(channels)
        self.input_proj = nn.Conv2d(self.in_channels, base_channels, 3, padding=1)
        self.encoder_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        in_ch = base_channels
        current_res = config.img_size
        
        for i in range(num_levels):
            out_ch = channels[i]
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResidualBlock(in_ch, out_ch, time_dim, dropout))
                in_ch = out_ch
                if current_res in attention_resolutions:
                    blocks.append(AttentionBlock(out_ch))
            self.encoder_blocks.append(blocks)
            self.downsample_blocks.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1) if i < num_levels - 1 else nn.Identity())
            if i < num_levels - 1: current_res //= 2
        mid_ch = channels[-1]
        self.middle = nn.ModuleList([
            ResidualBlock(mid_ch, mid_ch, time_dim, dropout), AttentionBlock(mid_ch),
            ResidualBlock(mid_ch, mid_ch, time_dim, dropout)
        ])
        self.decoder_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        in_ch = mid_ch
        for i in reversed(range(num_levels)):
            out_ch = channels[i]
            skip_ch = channels[i]
            blocks = nn.ModuleList()
            res_input_ch = in_ch + skip_ch
            blocks.append(ResidualBlock(res_input_ch, out_ch, time_dim, dropout))
            in_ch = out_ch
            current_res *= 2
            if current_res in attention_resolutions: blocks.append(AttentionBlock(out_ch))
            for _ in range(num_res_blocks - 1):
                blocks.append(ResidualBlock(in_ch, out_ch, time_dim, dropout))
                if current_res in attention_resolutions: blocks.append(AttentionBlock(out_ch))
            self.decoder_blocks.append(blocks)
            self.upsample_blocks.append(nn.ConvTranspose2d(out_ch, channels[i-1], 4, stride=2, padding=1) if i > 0 else nn.Identity())
            if i > 0: in_ch = channels[i-1]
        self.final = nn.Sequential(nn.GroupNorm(8, base_channels), nn.SiLU(), nn.Conv2d(base_channels, out_channels, 3, padding=1))

    def forward(self, x, timesteps, condition):
        x = torch.cat([x, condition], dim=1)
        t = self.time_mlp(timesteps)
        x = self.input_proj(x)
        skip_connections = []
        for level_blocks, downsample in zip(self.encoder_blocks, self.downsample_blocks):
            for module in level_blocks:
                x = module(x, t) if isinstance(module, ResidualBlock) else module(x)
            skip_connections.append(x)
            x = downsample(x)
        for module in self.middle:
            x = module(x, t) if isinstance(module, ResidualBlock) else module(x)
        skip_connections = list(reversed(skip_connections))
        for i, (level_blocks, upsample) in enumerate(zip(self.decoder_blocks, self.upsample_blocks)):
            skip = skip_connections[i]
            if x.shape[-2:] != skip.shape[-2:]: x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            for module in level_blocks:
                 x = module(x, t) if isinstance(module, ResidualBlock) else module(x)
            x = upsample(x)
        return self.final(x)


class DDPMDiffusion:
    def __init__(self, model, timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cuda'):
        self.model = model
        self.timesteps = timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

    def q_sample(self, x_start, t, noise=None):
        noise = torch.randn_like(x_start) if noise is None else noise
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod.gather(0, t).reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(0, t).reshape(-1, 1, 1, 1)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_x0_from_noise(self, x_t, t, noise):
        sqrt_recip_alphas_cumprod_t = self.sqrt_recip_alphas_cumprod.gather(0, t).reshape(-1, 1, 1, 1)
        sqrt_recipm1_alphas_cumprod_t = self.sqrt_recipm1_alphas_cumprod.gather(0, t).reshape(-1, 1, 1, 1)
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise

    def p_losses(self, x_start, condition, t, noise=None):
        if noise is None: noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.model(x_noisy, t, condition)
        loss_noise = F.l1_loss(predicted_noise, noise)
        x_recons = self.predict_x0_from_noise(x_noisy, t, predicted_noise)
        loss_image = F.l1_loss(x_recons, x_start)
        total_loss = loss_noise + loss_image
        return total_loss, loss_noise, loss_image
    
    @torch.no_grad()
    def ddim_sample_step(self, x_t, t, t_prev, condition, eta=0.0):
        t_tensor = torch.full((x_t.shape[0],), t, device=self.device, dtype=torch.long)
        pred_noise = self.model(x_t, t_tensor, condition)
        alpha_t = self.alphas_cumprod[t].clone().detach()
        alpha_t_prev = self.alphas_cumprod[t_prev].clone().detach() if t_prev >= 0 else torch.tensor(1.0, device=self.device)
        pred_x0 = self.predict_x0_from_noise(x_t, t_tensor, pred_noise).clamp(-1., 1.)
        sigma = 0.0
        if t > t_prev and eta > 0:
            variance_term = torch.clamp((1.0 - alpha_t_prev) / (1.0 - alpha_t) * (1.0 - alpha_t / alpha_t_prev), min=1e-12)
            sigma = eta * torch.sqrt(variance_term)
        sigma_sq = sigma**2 if isinstance(sigma, torch.Tensor) else torch.tensor(sigma**2, device=self.device)
        sqrt_one_minus_alpha_t_prev_minus_sigma_sq_term = torch.clamp(1.0 - alpha_t_prev - sigma_sq, min=1e-12)
        sqrt_one_minus_alpha_t_prev_minus_sigma_sq = torch.sqrt(sqrt_one_minus_alpha_t_prev_minus_sigma_sq_term)
        pred_dir_xt = sqrt_one_minus_alpha_t_prev_minus_sigma_sq * pred_noise
        noise_component = torch.randn_like(x_t) * sigma if t > 0 and eta > 0 else torch.zeros_like(x_t)
        x_prev = torch.sqrt(alpha_t_prev) * pred_x0 + pred_dir_xt + noise_component
        return x_prev

    @torch.no_grad()
    def ddim_sample(self, condition, num_steps=50, eta=0.0):
        self.model.eval() 
        b, _, h, w = condition.shape
        timesteps_np = np.linspace(self.timesteps - 1, 0, num_steps, dtype=int)
        timesteps = torch.from_numpy(timesteps_np).long().to(self.device)
        timesteps_prev_np = np.concatenate([timesteps_np[1:], [-1]])
        timesteps_prev = torch.from_numpy(timesteps_prev_np).long().to(self.device)
        img = torch.randn(b, 1, h, w, device=self.device)
        
        for t, t_prev in zip(timesteps, timesteps_prev):
            img = self.ddim_sample_step(img, t, t_prev, condition, eta)

        return img
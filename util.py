import numpy as np
import math
import torch
import matplotlib.pyplot as plt
import re
import os

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb



def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


def build_scheduler(optimizer, args, steps_per_epoch):
    total_steps = int(args.pretrain_epochs * steps_per_epoch)
    warmup_steps = int(args.warmup_epochs * steps_per_epoch)
    base_lr = args.pretrain_lr
    min_lr = args.min_lr

    def lr_lambda(current_step):
        if current_step < warmup_steps and warmup_steps > 0:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            if current_step >= total_steps:
                return float(min_lr) / float(max(1e-12, base_lr))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            # cosine decay from 1 -> 0 as progress goes 0 -> 1
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return (float(min_lr) / float(max(1e-12, base_lr))) + (1.0 - float(min_lr) / float(max(1e-12, base_lr))) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def show_image(image, title=''):
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def lr_mul(step, base_lr, min_lr, warmup_steps, total_steps):
    if step < warmup_steps and warmup_steps > 0:
        return float(step) / float(max(1, warmup_steps))
    else:
        if step >= total_steps:
            return float(min_lr) / float(max(1e-12, base_lr))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    # cosine decay from 1 -> 0 as progress goes 0 -> 1
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return (float(min_lr) / float(max(1e-12, base_lr))) + (1.0 - float(min_lr) / float(max(1e-12, base_lr))) * cosine_decay

def checkpoint_exists(args, ckpts, optimizer, scheduler, steps_per_epoch):
    latest_epoch = -1
    latest_path = None
    for p in ckpts:
        m = re.search(r"mae_pretrain_epoch(\d+)\.pth$", p)
        if m:
            e = int(m.group(1))
            if e > latest_epoch:
                latest_epoch = e
                latest_path = p
    if latest_path is not None:
        state_dict = torch.load(latest_path, map_location=args.device)
        try:
            start_epoch = latest_epoch
            # compute resumed global step and set optimizer lr accordingly
            resumed_steps = int(start_epoch * steps_per_epoch)
            total_steps = int(args.pretrain_epochs * steps_per_epoch)
            warmup_steps = int(args.warmup_epochs * steps_per_epoch)
            mul = lr_mul(resumed_steps, args.pretrain_lr, args.min_lr, warmup_steps, total_steps)
            
            for g in optimizer.param_groups:
                if 'lr_scale' in g:
                    g['lr'] = args.pretrain_lr * mul * g['lr_scale']
                else:
                    g['lr'] = args.pretrain_lr * mul

            # sync scheduler internal counter to resumed_steps
            try:
                scheduler.last_epoch = resumed_steps - 1
                scheduler.step()
            except Exception:
                # fallback: repeatedly step (safe but may be slower)
                for _ in range(resumed_steps):
                    scheduler.step()

            print(f"Resumed from checkpoint {latest_path} (epoch {start_epoch}), set lr={optimizer.param_groups[0]['lr']}")
        except Exception as e:
            print(f"Failed to load checkpoint {latest_path}: {e}")
            start_epoch = 0
    else:
        start_epoch = 0
    return state_dict, optimizer, scheduler, start_epoch
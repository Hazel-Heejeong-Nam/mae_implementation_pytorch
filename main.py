import argparse 
import os 

import numpy as np
import torch
import torch.nn as nn
import wandb
import submitit
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_manage import patch_dataset
from model.models_mae import MaskedAutoencoderViT
from util import build_scheduler, show_image


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

# argparser
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # wandb, slurm
    # parser.add_argument("--use_slurm", action="store_true", default=False)
    parser.add_argument("--use_wandb", action="store_true", default=True)

    # base
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--mode", default="pretrain", type=str, choices=["pretrain", "linprobe", "finetune"])
    parser.add_argument("--checkpoint_dir", default="./checkpoints", type=str)
    parser.add_argument("--data_dir", default="./galaxy_dataset", type=str)

    # pretraining parameters
    parser.add_argument("--pretrain_epochs", default=100, type=int)
    parser.add_argument("--pretrain_lr", default=1e-3, type=float)
    parser.add_argument("--pretrain_mask_ratio", default=0.75, type=float)  
    parser.add_argument("--vit_model", default="vit_base_patch16", type=str)
    parser.add_argument("--normalized_prediction", action="store_true", default=True)
    parser.add_argument("--validation_epochs", default=1, type=int)

    # LR schedule
    parser.add_argument("--warmup_epochs", default=10, type=int,
                        help="linear warmup epochs for LR")
    parser.add_argument("--min_lr", default=0.0, type=float,
                        help="minimum lr after decay")


    # linear probing parameters
    parser.add_argument("--linprobe_epochs", default=20, type=int)
    parser.add_argument("--linprobe_lr", default=1.5e-4, type=float)

    args = parser.parse_args()

    return args



def main_worker(args):
    print(torch.cuda.is_available())
    train_dataset = patch_dataset(data_dir=args.data_dir, mode="train")
    val_dataset = patch_dataset(data_dir=args.data_dir, mode="test")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    if args.mode == "pretrain":
        print("Pretraining mode")
        mae = MaskedAutoencoderViT(img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False)
        mae.to(args.device)
        print('model created')

        optimizer = torch.optim.AdamW(mae.parameters(), lr=args.pretrain_lr, weight_decay=0.05)
        steps_per_epoch = len(train_loader)
        scheduler = build_scheduler(optimizer, args, steps_per_epoch)
        for epoch in range(args.pretrain_epochs):
            print("Epoch {}/{}".format(epoch+1, args.pretrain_epochs))
            mae.train()
            train_loss = 0.0
            for images, _ in tqdm(train_loader):
                images = images.to(args.device)

                loss, pred, mask = mae(images, mask_ratio=args.pretrain_mask_ratio)
                # pred shape bs, 196 (num masked patch), 768 (768 = 16*16*3)
                # mask shape bs, 196
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # step scheduler once per optimizer step
                scheduler.step()

                train_loss += loss.item() * images.size(0)

            train_loss /= len(train_loader.dataset)
            print(f"Epoch [{epoch+1}/{args.pretrain_epochs}], Train Loss: {train_loss:.4f}")

            if args.use_wandb:
                wandb.log({"Pretrain Train Loss": train_loss, "epoch": epoch+1, "lr": optimizer.param_groups[0]['lr']})

            # Save checkpoint
            if (epoch + 1) % 10 == 0 or (epoch + 1) == args.pretrain_epochs:
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(args.checkpoint_dir, f"mae_pretrain_epoch{epoch+1}.pth")
                torch.save(mae.state_dict(), checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")

            # Validation
            if (epoch + 1) % args.validation_epochs == 0 or (epoch + 1) == args.pretrain_epochs:
                mae.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for images, _ in tqdm(val_loader):
                        images = images.to(args.device)
                        loss, pred, mask = mae(images, mask_ratio=args.pretrain_mask_ratio)
                        val_loss += loss.item() * images.size(0)

                val_loss /= len(val_loader.dataset)
                print(f"Epoch [{epoch+1}/{args.pretrain_epochs}], Val Loss: {val_loss:.4f}")

                if args.use_wandb:
                    wandb.log({"Pretrain Val Loss": val_loss, "epoch": epoch+1})    

                pred = mae.unpatchify(pred)
                pred = torch.einsum('nchw->nhwc', pred).detach().cpu()

                # visualize the mask
                mask = mask.detach()
                mask = mask.unsqueeze(-1).repeat(1, 1, mae.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
                mask = mae.unpatchify(mask)  # 1 is removing, 0 is keeping
                mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
                
                x = torch.einsum('nchw->nhwc', x)

                # masked image
                im_masked = x * (1 - mask)
                im_paste = x * (1 - mask) + pred * mask
                plt.rcParams['figure.figsize'] = [24, 24]
                plt.subplot(1, 4, 1)
                show_image(x[0], "original")

                plt.subplot(1, 4, 2)
                show_image(im_masked[0], "masked")
                plt.subplot(1, 4, 3)
                show_image(pred[0], "reconstruction")
                plt.subplot(1, 4, 4)
                show_image(im_paste[0], "reconstruction + visible")

                plt.show()


    elif args.mode in ["linprobe", "finetune"]:
        pass

if __name__ =="__main__":
    args = parse_args()

    if args.debug:
        args.batch_size = 128
        args.use_wandb = False

    if args.use_wandb:
        api_key = open("wandb_key.txt", "r").read().strip()
        wandb.login(key=api_key)
        wandb.init(project="mae_galaxy", config=vars(args), name=f"{args.vit_model}_mask{args.pretrain_mask_ratio}")

    # if args.use_slurm:
    #     executor = submitit.AutoExecutor(folder="logs_slurm")
    #     executor.update_parameters(
    #         mem_gb=32,
    #         gpus_per_node=1,
    #         cpus_per_task=args.num_workers + 2,
    #         nodes=1,
    #         timeout_min=720,
    #         slurm_partition="gpu",
    #         slurm_signal_delay_s=120,
    #     )
    #     job = executor.submit(main_worker, args)
    #     print(job)

    # else:
    main_worker(args)
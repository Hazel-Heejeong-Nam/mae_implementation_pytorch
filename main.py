import argparse 
import os 

import numpy as np
import torch
import wandb
import submitit
import glob
from timm.models.layers import trunc_normal_

from data_manage import patch_dataset
from model.models_mae import mae_vit_base, mae_vit_large, mae_vit_huge
from model.models_vit import vit_base_patch16, vit_large_patch16, vit_huge_patch14
from util import build_scheduler, interpolate_pos_embed, checkpoint_exists
from train import train_one_epoch_pretrain, train_one_epoch_linprobe

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

# argparser
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # wandb, slurm
    parser.add_argument("--use_slurm", action="store_true", default=False)
    parser.add_argument("--use_wandb", action="store_true", default=True)
    parser.add_argument("--slurm_walltime", default=9, type=int, help="Slurm walltime in hours.")

    # base
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--mode", default="linprobe", type=str, choices=["pretrain", "linprobe", "finetune"])
    parser.add_argument("--checkpoint_dir", default="./checkpoints", type=str)
    parser.add_argument("--data_dir", default="./galaxy_dataset", type=str)

    # training general
    parser.add_argument("--validation_epochs", default=1, type=int)
    parser.add_argument("--warmup_epochs", default=5, type=int)
    parser.add_argument("--min_lr", default=0.0, type=float)


    # pretraining parameters
    parser.add_argument("--pretrain_epochs", default=100, type=int)
    parser.add_argument("--pretrain_lr", default=1e-4, type=float)
    parser.add_argument("--pretrain_mask_ratio", default=0.75, type=float)  
    parser.add_argument("--patch_size", default=16, type=int)   
    parser.add_argument("--vit_model", default="vit_large", choices=["vit_base", "vit_large", "vit_huge"], type=str)
    parser.add_argument("--normalized_prediction", action="store_true", default=True)


    # linear probing parameters
    parser.add_argument("--linprobe_epochs", default=100, type=int)
    parser.add_argument("--linprobe_lr", default=1e-3, type=float)
    parser.add_argument("--global_pool", action="store_true", default=False)
    parser.add_argument("--nb_classes", default=10, type=int, help="Number of classes for linear probing.")

    args = parser.parse_args()

    return args



def main_worker(args):
    if args.use_wandb:
        try:
            api_key = open("/users/hnam16/hazel/code/mae_implementation_pytorch/wandb_key.txt", "r").read().strip()
            wandb.login(key=api_key)
            print("Logged in to wandb")
        except:
            args.use_wandb = False
            print("Failed to login to wandb, running without wandb")

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
        wandb.init(project="mae_galaxy_v2", config=vars(args), name=fname)
        print("Pretraining mode")
        if args.vit_model == "vit_base":
            mae = mae_vit_base()
        elif args.vit_model == "vit_large":
            mae = mae_vit_large()
        elif args.vit_model == "vit_huge":
            mae = mae_vit_huge()
        mae.to(args.device)
        print('model created')

        os.makedirs(args.checkpoint_dir+"/validation_images", exist_ok=True)

        optimizer = torch.optim.AdamW(mae.parameters(), lr=args.pretrain_lr, weight_decay=0.05)
        steps_per_epoch = len(train_loader)
        scheduler = build_scheduler(optimizer, args, steps_per_epoch)
        start_epoch = 0
        ckpt_dir = args.checkpoint_dir
        if ckpt_dir and os.path.isdir(ckpt_dir) and not args.debug:
            ckpts = glob.glob(os.path.join(ckpt_dir, "mae_pretrain_epoch*.pth"))
            if len(ckpts) > 0:
                state_dict, optimizer, scheduler, start_epoch = checkpoint_exists(args, ckpts, optimizer, scheduler, steps_per_epoch)
                mae.load_state_dict(state_dict)
            else:
                start_epoch = 0
        else:
            start_epoch = 0

        # resume from latest checkpoint in checkpoint_dir if present
        for epoch in range(start_epoch, args.pretrain_epochs):
            print("Epoch {}/{}".format(epoch+1, args.pretrain_epochs))
            mae, optimizer, scheduler = train_one_epoch_pretrain(args, mae, optimizer, scheduler, train_loader, val_loader, epoch, fname)


    # Linear Probing 
    elif args.mode == "linprobe":
        probing_name = f"linprobe_lr{args.linprobe_lr}_epochs{args.linprobe_epochs}"
        wandb.init(project="linprobe_galaxy_v2", config=vars(args), name=fname+probing_name)

        print("Linear Probing mode")
        if args.vit_model == "vit_base":
            model = vit_base_patch16(num_classes=args.nb_classes,global_pool=args.global_pool)
        elif args.vit_model == "vit_large":
            model = vit_large_patch16(num_classes=args.nb_classes,global_pool=args.global_pool)
        elif args.vit_model == "vit_huge":
            model = vit_huge_patch14(num_classes=args.nb_classes,global_pool=args.global_pool)

        checkpoint = torch.load(os.path.join(args.checkpoint_dir, f"mae_pretrain_epoch{args.pretrain_epochs}.pth"), map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % os.path.join(args.checkpoint_dir, f"mae_pretrain_epoch{args.pretrain_epochs}.pth"))
        interpolate_pos_embed(model, checkpoint)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint, strict=False)
        if args.global_pool:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer: following MoCo v3
        trunc_normal_(model.head.weight, std=0.01)

        # for linear prob only
        # hack: revise model's head with BN
        model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
        # freeze all but the head
        for _, p in model.named_parameters():
            p.requires_grad = False
        for _, p in model.head.named_parameters():
            p.requires_grad = True
        print('#params trainable:', sum(p.numel() for p in model.parameters() if p.requires_grad))
        model.to(args.device)

        optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.linprobe_lr, weight_decay=0.05)
        steps_per_epoch = len(train_loader)
        scheduler = build_scheduler(optimizer, args, steps_per_epoch)
        criterion = torch.nn.CrossEntropyLoss()
    
        for epoch in range(args.linprobe_epochs):
            print("Epoch {}/{}".format(epoch+1, args.linprobe_epochs))
            model, optimizer, scheduler = train_one_epoch_linprobe(args, model, criterion, optimizer, scheduler, train_loader, val_loader, epoch, probing_name)
        


if __name__ =="__main__":
    args = parse_args()

    if args.debug:
        # args.batch_size = 2
        if args.mode=="pretrain":
            args.pretrain_epochs = 2
        elif args.mode=='linprobe':
            args.linprobe_epochs = 2
        args.use_wandb = False

    fname = f"pretrain_{args.vit_model}_mask{args.pretrain_mask_ratio}_lr{args.pretrain_lr}_patchsize{args.patch_size}_epochs{args.pretrain_epochs}"
    checkpoint_dir = os.path.join(args.checkpoint_dir, fname)
    args.checkpoint_dir = checkpoint_dir

    if args.mode == "pretrain":
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        # set batch size to 128
    elif args.mode == "linprobe":
        assert os.path.exists(args.checkpoint_dir), "Checkpoint directory does not exist for linear probing"
        # set batch size to 512
    if args.use_slurm:
        executor = submitit.AutoExecutor(folder="logs_slurm")
        executor.update_parameters(
            mem_gb=10,
            gpus_per_node=1,
            cpus_per_task=args.num_workers + 2,
            nodes=1,
            timeout_min=args.slurm_walltime * 60,  
            slurm_partition="gpu",
            slurm_signal_delay_s=120,
        )
        job = executor.submit(main_worker, args)
        print(job)

    # else:
    main_worker(args)
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
from model.models_mae import mae_vit_base, mae_vit_large, mae_vit_huge
from model.models_vit import vit_base_patch16, vit_large_patch16, vit_huge_patch14
from util import build_scheduler, interpolate_pos_embed
from timm.models.layers import trunc_normal_

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

# argparser
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # wandb, slurm
    parser.add_argument("--use_slurm", action="store_true", default=False)
    parser.add_argument("--use_wandb", action="store_true", default=True)

    # base
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--debug", action="store_true", default=True)
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
    parser.add_argument("--pretrain_lr", default=1e-3, type=float)
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
        wandb.init(project="mae_galaxy", config=vars(args), name=fname)
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

    # Linear Probing 
    elif args.mode == "linprobe":
        probing_name = f"linprobe_lr{args.linprobe_lr}_epochs{args.linprobe_epochs}"
        wandb.init(project="linprobe_galaxy", config=vars(args), name=fname+probing_name)

        print("Linear Probing mode")
        if args.vit_model == "vit_base":
            model = vit_base_patch16(num_classes=args.nb_classes,global_pool=args.global_pool)
        elif args.vit_model == "vit_large":
            model = vit_large_patch16(num_classes=args.nb_classes,global_pool=args.global_pool)
        elif args.vit_model == "vit_huge":
            model = vit_huge_patch14(num_classes=args.nb_classes,global_pool=args.global_pool)

        checkpoint = torch.load(os.path.join(args.checkpoint_dir, f"mae_pretrain_epoch{args.pretrain_epochs}.pth"), map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % os.path.join(args.checkpoint_dir, f"mae_pretrain_epoch{args.pretrain_epochs}.pth"))
        # state_dict = model.state_dict()
        # for k in ['head.weight', 'head.bias']:
        #     if k in checkpoint and checkpoint[k].shape != state_dict[k].shape:
        #         print(f"Removing key {k} from pretrained checkpoint")
        #         del checkpoint[k]

        # interpolate position embedding
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
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            for images, labels in tqdm(train_loader):
                images = images.to(args.device)
                labels = labels.to(args.device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # step scheduler once per optimizer step
                scheduler.step()

                train_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_loss /= len(train_loader.dataset)
            train_acc = 100.*correct/total
            print(f"Epoch [{epoch+1}/{args.linprobe_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

            if args.use_wandb:
                wandb.log({"Linprobe Train Loss": train_loss, "Linprobe Train Acc": train_acc, "epoch": epoch+1, "lr": optimizer.param_groups[0]['lr']})

            # Save checkpoint
            if (epoch + 1) % 10 == 0 or (epoch + 1) == args.linprobe_epochs:
                lin_checkpoint_path = os.path.join(args.checkpoint_dir, probing_name)
                os.makedirs(lin_checkpoint_path, exist_ok=True)
                torch.save(model.state_dict(), lin_checkpoint_path+f"/linprobe_epoch{epoch+1}.pth")
                print(f"Probing checkpoint saved at {lin_checkpoint_path}")

            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in tqdm(val_loader):
                    images = images.to(args.device)
                    labels = labels.to(args.device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            val_loss /= len(val_loader.dataset)
            val_acc = 100.*correct/total
            print(f"Epoch [{epoch+1}/{args.linprobe_epochs}], Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}%")


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
            timeout_min=540,
            slurm_partition="gpu",
            slurm_signal_delay_s=120,
        )
        job = executor.submit(main_worker, args)
        print(job)

    # else:
    main_worker(args)
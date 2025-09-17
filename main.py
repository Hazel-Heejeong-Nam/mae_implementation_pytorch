# import
import argparse 
import os 

import torch
import wandb
import submitit

from data_manage import patch_dataset



# argparser
def parse_args():

    # wandb, slurm
    parser.add_argument("--slurm", action="store_true", default=False)
    parser.add_argument("--use_wandb", action="store_true", default=True)

    # base
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--mode", default="pretrain", type=str, choices=["pretrain", "linprobe", "finetune"])
    parser.add_argument("--checkpoint_dir", default="./checkpoints", type=str)
    parser.add_argument("--data_dir", default="./galaxy_dataset", type=str)

    # pretraining parameters
    parser.add_argument("--pretrain_epochs", default=100, type=int)
    parser.add_argument("--pretrain_lr", default=1.5e-4, type=float)
    parser.add_argument("--pretrain_mask_ratio", default=0.75, type=float)  
    parser.add_argument("--vit_model", default="vit_base_patch16", type=str)
    parser.add_argument("--normalized_prediction", action="store_true", default=True)


    # linear probing parameters
    parser.add_argument("--linprobe_epochs", default=20, type=int)
    parser.add_argument("--linprobe_lr", default=1.5e-4, type=float)

    args = parser.parse_args()

    return args



def main_worker(args):
    train_dataset = patch_dataset(data_dir=args.data_dir, mode="train")
    val_dataset = patch_dataset(data_dir=args.data_dir, mode="test")

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    #     drop_last=True,
    # )

    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    #     drop_last=False,
    # )

    if args.mode == "pretrain":
        print("Pretraining mode")



if __name__ =="__main__":
    args = parse_args()

    if args.debug:
        args.batch_size = 2
        args.use_wandb = False

    if args.use_wandb:
        api_key = open("wandb_key.txt", "r").read().strip()
        wandb.login(key=api_key)
        wandb.init(project="mae_galaxy", config=vars(args), name=f"{args.vit_model}_mask{args.pretrain_mask_ratio}")

    if args.slurm:
        executor = submitit.AutoExecutor(folder="logs_slurm")
        executor.update_parameters(
            mem_gb=32,
            gpus_per_node=1,
            cpus_per_task=args.num_workers + 2,
            nodes=1,
            timeout_min=720,
            slurm_partition="gpu",
            slurm_signal_delay_s=120,
        )
        job = executor.submit(main_worker, args)
        print(job)

    else:
        main_worker(args)
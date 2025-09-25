import os
import torch
import wandb
from tqdm import tqdm    


def train_one_epoch_pretrain(args, mae, optimizer, scheduler, train_loader, val_loader, epoch, fname):
    mae.train()
    train_loss = 0.0
    for images, _ in tqdm(train_loader):
        images = images.to(args.device)

        loss, _, _ = mae(images, mask_ratio=args.pretrain_mask_ratio)
        optimizer.zero_grad() # pred shape bs, 196 (num masked patch), 768 (768 = 16*16*3) # mask shape bs, 196
        loss.backward()
        optimizer.step()
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
                loss, _, _ = mae(images, mask_ratio=args.pretrain_mask_ratio)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"Epoch [{epoch+1}/{args.pretrain_epochs}], Val Loss: {val_loss:.4f}")

        if args.use_wandb:
            wandb.log({"Pretrain Val Loss": val_loss, "epoch": epoch+1})    
    return mae, optimizer, scheduler



def train_one_epoch_linprobe(args, model, criterion, optimizer, scheduler, train_loader, val_loader, epoch, probing_name):
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

    return model, optimizer, scheduler

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score
import numpy as np
from dataset import get_data_loaders
from model import get_model
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join('runs', f'mrnet_{args.plane}_{args.lr}'))

    train_loader, val_loader = get_data_loaders(
        data_root=args.data_root, 
        batch_size=args.batch_size, 
        plane=args.plane, 
        num_workers=args.num_workers,
        num_slices=args.num_slices
    )

    model = get_model(pretrained=True).to(device)

    # Differential learning rates: backbone vs head
    # Feature extractor (ViT) needs to be stable, head learns from scratch
    head_params = list(model.classifier.parameters()) + list(model.attention_pool.parameters())
    backbone_params = [p for n, p in model.named_parameters() if 'classifier' not in n and 'attention_pool' not in n]
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.2}, 
        {'params': head_params, 'lr': args.lr}
    ], weight_decay=1e-2)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Weights for class imbalance
    pos_weights = torch.tensor([0.25, 4.0, 2.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    
    scaler = torch.cuda.amp.GradScaler()
    best_auc = 0.0

    print(f"Starting SOTA Sequence Training on {args.plane}...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
        scheduler.step()
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                all_preds.append(torch.sigmoid(outputs).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        try:
            auc_scores = [roc_auc_score(all_labels[:, i], all_preds[:, i]) for i in range(3)]
            mean_auc = np.mean(auc_scores)
        except:
            mean_auc, auc_scores = 0.5, [0.5, 0.5, 0.5]
            
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {train_loss:.4f} | Val AUC: {mean_auc:.4f}")
        print(f"  -> Abn: {auc_scores[0]:.3f} | ACL: {auc_scores[1]:.3f} | Men: {auc_scores[2]:.3f}")
        
        # TensorBoard Logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('AUC/mean', mean_auc, epoch)
        writer.add_scalar('AUC/Abnormal', auc_scores[0], epoch)
        writer.add_scalar('AUC/ACL', auc_scores[1], epoch)
        writer.add_scalar('AUC/Meniscus', auc_scores[2], epoch)
        
        if mean_auc > best_auc:
            best_auc = mean_auc
            torch.save(model.state_dict(), args.save_path)
            print(f"  *** NEW BEST MODEL SAVED (AUC: {best_auc:.4f}) ***")
            
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--plane', type=str, default='sagittal')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_slices', type=int, default=24)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=2e-4) # Higher head LR
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_path', type=str, default='best_model.pth')
    train(parser.parse_args())

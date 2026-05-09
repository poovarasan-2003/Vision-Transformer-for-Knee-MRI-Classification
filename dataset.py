import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple, Optional

class MRNetDataset(Dataset):
    def __init__(self, data_dir: str, plane: str = 'sagittal', transform: Optional[transforms.Compose] = None, num_slices: int = 24):
        self.data_dir = os.path.join(data_dir, plane)
        self.plane = plane
        self.transform = transform
        self.num_slices = num_slices
        
        self.exam_ids = sorted([f.split('.')[0] for f in os.listdir(self.data_dir) if f.endswith('.npy')])
        self.labels = self._load_labels(data_dir)

    def _load_labels(self, data_dir: str) -> np.ndarray:
        label_files = ['abnormal', 'acl', 'meniscus']
        labels_list = []
        split = 'train' if 'train' in data_dir else 'valid'
        parent_dir = os.path.dirname(data_dir) 
        
        for condition in label_files:
            csv_path = os.path.join(parent_dir, f'{split}-{condition}.csv')
            df = pd.read_csv(csv_path, header=None, names=['exam_id', 'label'])
            df['exam_id'] = df['exam_id'].astype(str).str.zfill(4)
            labels_list.append(df.set_index('exam_id')['label'])
            
        labels_df = pd.concat(labels_list, axis=1)
        labels_df.columns = ['abnormal', 'acl', 'meniscus']
        labels_df = labels_df.reindex(self.exam_ids).fillna(0)
        return labels_df.values.astype(np.float32)

    def _load_volume(self, exam_id: str) -> np.ndarray:
        file_path = os.path.join(self.data_dir, f'{exam_id}.npy')
        return np.load(file_path)

    def __len__(self) -> int:
        return len(self.exam_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        exam_id = self.exam_ids[idx]
        volume = self._load_volume(exam_id) 
        
        # Uniformly sample num_slices
        idxs = np.linspace(0, volume.shape[0] - 1, self.num_slices).astype(int)
        
        processed_slices = []
        for i in idxs:
            s = volume[i].astype(np.float32)
            if s.max() > s.min():
                s = (255 * (s - s.min()) / (s.max() - s.min() + 1e-8)).astype(np.uint8)
            else:
                s = np.zeros_like(s, dtype=np.uint8)
            
            s_rgb = np.stack([s]*3, axis=-1)

            if self.transform:
                s_tensor = self.transform(s_rgb)
            else:
                s_tensor = torch.from_numpy(s_rgb.transpose(2, 0, 1)).float() / 255.0
            
            processed_slices.append(s_tensor)
            
        sequence_tensor = torch.stack(processed_slices, dim=0)
        labels = torch.from_numpy(self.labels[idx]).float()
        return sequence_tensor, labels

def get_data_loaders(data_root: str, batch_size: int = 1, plane: str = 'sagittal', num_workers: int = 2, num_slices: int = 24) -> Tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.05, 0.05)),
        transforms.GaussianBlur(kernel_size=3), # SOTA: Handle MRI blur
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = MRNetDataset(os.path.join(data_root, 'train'), plane=plane, transform=train_transform, num_slices=num_slices)
    val_dataset = MRNetDataset(os.path.join(data_root, 'valid'), plane=plane, transform=val_transform, num_slices=num_slices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader

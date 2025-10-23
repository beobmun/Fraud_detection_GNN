import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import HeteroData
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd
from tqdm import tqdm

from model import FocalLoss
from dataloader import Dataset, collate_fn

#pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

class Train():
    def __init__(self):
        self.model = None
        self.dataset = None
        self.device = None

        self.train_dataloaders = None
        self.val_dataloaders = None
        self.test_dataloaders = None

    def set_dataset(self, dataset):
        self.dataset = dataset
        return self

    def set_model(self, model):
        self.model = model
        return self

    def set_device(self, device):
        self.device = device
        return self

    def set_dataloaders_(self,start, end, n_splits, window_size, memory_size, batch_size):
        days = pd.date_range(start=start, end=end, freq='D')
        train_days = days[:int(len(days)*0.9)]
        # test_days = days[int(len(days)*0.9):]

        tscv = TimeSeriesSplit(n_splits=n_splits)
        for train, val in tscv.split(train_days):
            train_dataset = Dataset(self.dataset, days[train], window_size, memory_size)
            val_dataset = Dataset(self.dataset, days[val], window_size, memory_size)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            train_key = f"{str(days[train[0]]).split(' ')[0]}~{str(days[train[-1]]).split(' ')[0]}"
            val_key = f"{str(days[val[0]]).split(' ')[0]}~{str(days[val[-1]]).split(' ')[0]}"
            self.train_dataloaders[train_key] = train_dataloader
            self.val_dataloaders[val_key] = val_dataloader

        return self
    
    def set_dataloaders(self, start, end, window_size, memory_size, batch_size):
        days = pd.date_range(start=start, end=end, freq='D')
        train_days = days[:int(len(days)*0.6)]
        val_days = days[int(len(days)*0.6):int(len(days)*0.8)]
        # test_days = days[int(len(days)*0.8):]

        train_dataset = Dataset(self.dataset, train_days, window_size, memory_size)
        val_dataset = Dataset(self.dataset, val_days, window_size, memory_size)
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        return self

    def save_results(self, metrics, save_path, epoch=None):
        os.makedirs(save_path, exist_ok=True)
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(os.path.join(save_path, 'metrics.csv'), index=False)
        torch.save(self.model.state_dict(), os.path.join(save_path, f'epoch_{epoch}_model.pth'))

    def train_minibatch(self, history_snapshots_batch, target_batch, optimizer, criterion):
        self.model.train()
        optimizer.zero_grad()

        predictions_dict = self.model(history_snapshots_batch, target_batch)

        loss = 0
        all_preds_flat_batch = list()
        all_labels_flat_batch = list()

        for edge_type, preds in predictions_dict.items():
            if edge_type in target_batch.edge_labels_dict and hasattr(target_batch[edge_type], 'edge_labels'):
                labels = target_batch[edge_type].edge_labels.float()
                if labels.numel() > 0:
                    loss += criterion(preds.squeeze(-1), labels)
                    all_preds_flat_batch.append(preds.squeeze(-1).detach().cpu())
                    all_labels_flat_batch.append(labels.cpu())

        all_preds_flat_batch = torch.cat(all_preds_flat_batch)
        all_labels_flat_batch = torch.cat(all_labels_flat_batch)

        if isinstance(loss, torch.Tensor):
            loss.backward()
            optimizer.step()
            return loss.item(), all_preds_flat_batch, all_labels_flat_batch
        else:
            return 0.0, all_preds_flat_batch, all_labels_flat_batch
        
    def train_batch(self, dataloader, optimizer, criterion, epoch, batch_size):
        total_loss = 0
        num_batches = 0
        all_preds_flat = list()
        all_labels_flat = list()

        with tqdm(total=len(dataloader)*batch_size, desc=f"Epoch {epoch+1} Training", ncols=100, leave=False) as pbar:
            for history_snapshots_batch, target_batch in dataloader:
                history_snapshots_batch = [[snap.to(self.device) for snap in batched_seq.to_data_list()] for batched_seq in history_snapshots_batch]
                target_batch = target_batch.to(self.device)

                loss, batch_pred, batch_labels = self.train_minibatch(history_snapshots_batch, target_batch, optimizer, criterion)
                total_loss += loss
                all_preds_flat.append(batch_pred)
                all_labels_flat.append(batch_labels)
                num_batches += 1
                pbar.update(len(history_snapshots_batch))
            
            if num_batches > 0:
                avg_loss = total_loss / num_batches

            all_preds_flat = torch.cat(all_preds_flat).numpy()
            all_labels_flat = torch.cat(all_labels_flat).numpy()

            if len(np.unique(all_labels_flat)) > 1:
                roc_auc = roc_auc_score(all_labels_flat, all_preds_flat)
                pr_auc = average_precision_score(all_labels_flat, all_preds_flat)
            else:
                roc_auc = -1
                pr_auc = -1

            return avg_loss, roc_auc, pr_auc

    def evaluate(self, dataloader, criterion, epoch, batch_size):
        self.model.eval()
        all_preds_flat = list()
        all_labels_flat = list()
        val_loss = 0

        with torch.no_grad():
            with tqdm(total=len(dataloader)*batch_size, desc=f"Epoch {epoch+1} Evaluating", ncols=100, leave=False) as pbar:
                for history_snapshots_batch, target_batch in dataloader:
                    history_snapshots_batch = [[snap.to(self.device) for snap in batched_seq.to_data_list()] for batched_seq in history_snapshots_batch]
                    target_batch = target_batch.to(self.device)

                    predictions_dict = self.model(history_snapshots_batch, target_batch)

                    batch_loss = 0
                    for edge_type, preds in predictions_dict.items():
                        if edge_type in target_batch.edge_labels_dict and hasattr(target_batch[edge_type], 'edge_labels'):
                            labels = target_batch[edge_type].edge_labels.float()
                            if labels.numel() > 0:
                                batch_loss += criterion(preds.squeeze(-1), labels)
                                all_preds_flat.append(preds.squeeze(-1).cpu())
                                all_labels_flat.append(labels.cpu())
                    if isinstance(batch_loss, torch.Tensor):
                        val_loss += batch_loss.item()
                    
                    pbar.update(len(history_snapshots_batch))

                if not all_preds_flat:
                    return -1, -1, -1
                
            all_preds_flat = torch.cat(all_preds_flat).numpy()
            all_labels_flat = torch.cat(all_labels_flat).numpy()
            avg_val_loss = val_loss / len(dataloader)

            if len(np.unique(all_labels_flat)) > 1:
                roc_auc = roc_auc_score(all_labels_flat, all_preds_flat)
                pr_auc = average_precision_score(all_labels_flat, all_preds_flat)
            else:
                roc_auc = -1
                pr_auc = -1
            return avg_val_loss, roc_auc, pr_auc

    def run_training_(self, epochs, batch_size, learning_rate, train_dataloader=None, val_dataloader=None):
        if train_dataloader is None or val_dataloader is None:
            raise ValueError("Dataloaders must be provided")

        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        criterion = FocalLoss()

        for epoch in range(epochs):
            train_loss = self.train_batch(train_dataloader, optimizer, criterion, epoch, batch_size)
            val_loss, roc_auc, pr_auc = self.evaluate(val_dataloader, criterion, epoch, batch_size)

            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")

    def run_training(self, epochs, batch_size, learning_rate):
        if self.train_dataloader is None or self.val_dataloader is None:
            raise ValueError("Dataloaders must be set using set_dataloaders method")

        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=100, cycle_mult=1.0, max_lr=learning_rate, min_lr=learning_rate*0.01, warmup_steps=25, gamma=0.5)
        criterion = FocalLoss()

        metrics = {'train_loss': [], 'train_roc_auc': [], 'train_pr_auc': [],
                   'val_loss': [], 'val_roc_auc': [], 'val_pr_auc': []}

        try:
            for epoch in range(epochs):
                train_loss, train_roc_auc, train_pr_auc = self.train_batch(self.train_dataloader, optimizer, criterion, epoch, batch_size)
                val_loss, val_roc_auc, val_pr_auc = self.evaluate(self.val_dataloader, criterion, epoch, batch_size)
                scheduler.step()
                
                metrics['train_loss'].append(train_loss)
                metrics['train_roc_auc'].append(train_roc_auc)
                metrics['train_pr_auc'].append(train_pr_auc)
                metrics['val_loss'].append(val_loss)
                metrics['val_roc_auc'].append(val_roc_auc)
                metrics['val_pr_auc'].append(val_pr_auc)

                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Train ROC AUC: {train_roc_auc:.4f}, Train PR AUC: {train_pr_auc:.4f} | Val Loss: {val_loss:.4f}, Val ROC AUC: {val_roc_auc:.4f}, Val PR AUC: {val_pr_auc:.4f}")
                
                if (epoch+1) % 10 == 0:
                    self.save_results(metrics, 'training_results', epoch=str(epoch+1))

        except KeyboardInterrupt:
            self.save_results(metrics, 'training_results', epoch=f'interrupted_{epoch}')
            print("\nKeyboardInterrupt 발생: 학습 중단 및 결과 저장")
            
        else:
            self.save_results(metrics, 'training_results', epoch='final')
            print("Training completed.")
import torch
# from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, dates, window_size=1, memory_size=10):
        self.dataset = dataset
        self.start_date = dates[0]
        self.end_date = dates[-1]
        self.window_size = window_size
        self.memory_size = memory_size

        self.history_snapshots = list()
        self.target_snapshots = list()

        with tqdm(total=len(dates)-window_size, desc="Building snapshots", ncols=100, leave=False) as pbar:
            for i in range(len(dates) - window_size):
                hist_start = dates[i]
                hist_end = dates[i + window_size - 1]
                target_date = dates[i + window_size]

                hist_snapshot = dataset.build_hetero_graph(hist_start, hist_end)
                target_snapshot = dataset.build_hetero_graph(target_date, target_date)

                self.history_snapshots.append(hist_snapshot)
                self.target_snapshots.append(target_snapshot)
                pbar.update(1)


    def __len__(self):
        return len(self.history_snapshots)

    def __getitem__(self, idx):
        s = max(0, idx - self.memory_size + 1)
        return self.history_snapshots[s:idx+1], self.target_snapshots[idx]

def collate_fn(batch):
    hist_seq_list = [Batch.from_data_list(item[0]) for item in batch]
    target_batch = Batch.from_data_list([item[1] for item in batch])

    return hist_seq_list, target_batch
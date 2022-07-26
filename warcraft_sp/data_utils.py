import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class WarcraftImageDataset(Dataset):
    def __init__(self, inputs, labels, true_weights):
        self.inputs = inputs
        self.labels = labels
        self.true_weights = true_weights
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # image = read_image(img_path)
        # label = self.img_labels.iloc[idx, 1]
        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return self.inputs[idx], self.labels[idx], self.true_weights[idx]

class WarcraftDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, use_test_set=True,  normalize=True, batch_size=70, generator=None,num_workers=4):
        self.batch_size = batch_size
        self.generator = generator
        self.num_workers = num_workers

        train_prefix = "train"
        val_prefix = "val"
        test_prefix = "test"
        data_suffix = "maps"
        train_inputs = np.load(os.path.join(data_dir, train_prefix + "_" + data_suffix + ".npy")).astype(np.float32)
        train_inputs = train_inputs.transpose(0, 3, 1, 2)  # channel first

        val_inputs = np.load(os.path.join(data_dir, val_prefix + "_" + data_suffix + ".npy")).astype(np.float32)
        val_inputs = val_inputs.transpose(0, 3, 1, 2)  # channel first
        if use_test_set:
            test_inputs = np.load(os.path.join(data_dir, test_prefix + "_" + data_suffix + ".npy")).astype(np.float32)
            test_inputs = test_inputs.transpose(0, 3, 1, 2)  # channel first

        train_labels = np.load(os.path.join(data_dir, train_prefix + "_shortest_paths.npy"))
        train_true_weights = np.load(os.path.join(data_dir, train_prefix + "_vertex_weights.npy"))
        if normalize:
            mean, std = (
                np.mean(train_inputs, axis=(0, 2, 3), keepdims=True),
                np.std(train_inputs, axis=(0, 2, 3), keepdims=True),
            )
            train_inputs -= mean
            train_inputs /= std 
            val_inputs -= mean
            val_inputs /= std 
            if use_test_set:
                test_inputs -= mean
                test_inputs /= std    
        val_labels = np.load(os.path.join(data_dir, val_prefix + "_shortest_paths.npy"))
        val_true_weights = np.load(os.path.join(data_dir, val_prefix + "_vertex_weights.npy"))
        val_full_images = np.load(os.path.join(data_dir, val_prefix + "_maps.npy"))  
        if use_test_set:
            test_labels = np.load(os.path.join(data_dir, test_prefix + "_shortest_paths.npy"))
            test_true_weights = np.load(os.path.join(data_dir, test_prefix + "_vertex_weights.npy"))
            test_full_images = np.load(os.path.join(data_dir, test_prefix + "_maps.npy"))
        self.training_data = WarcraftImageDataset(train_inputs, train_labels, train_true_weights)
        self.val_data = WarcraftImageDataset(val_inputs, val_labels, val_true_weights)
        if use_test_set:
            self.test_data = WarcraftImageDataset(test_inputs, test_labels, test_true_weights)
    def train_dataloader(self):
        return DataLoader(self.training_data, batch_size=self.batch_size,generator= self.generator, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size,generator= self.generator, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size,generator= self.generator, num_workers=self.num_workers)
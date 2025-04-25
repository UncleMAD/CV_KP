from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd


class AircraftDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # Создаем отображение классов в индексы
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(self.data['Classes'].unique()))}
        self.data['label'] = self.data['Classes'].map(self.class_to_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        label = row['label']
        if self.transform:
            image = self.transform(image)
        return image, label

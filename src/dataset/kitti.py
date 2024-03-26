from torch.utils.data import Dataset

class KittiSegDataset(Dataset):
    def __init__(self, dataset, transform) -> None:
        images, labels = dataset

        self.images = images
        self.labels = labels

        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx):
        
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            transformed = self.transform(image=image, mask=label)
            image = transformed["image"]
            label = transformed["mask"]

        return image, label.long()

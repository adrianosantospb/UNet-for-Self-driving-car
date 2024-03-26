import os
from PIL import Image
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
from torchvision import transforms

from src.core.config import DatasetConfig
np.random.seed(100)


datasetConfig = DatasetConfig()

def open_image(file_path:str, is_mask=False) -> Image:
    '''
    Open image from file. 
    If the image is a label (for segmentation), set is_mask as True
    '''
    return Image.open(file_path).convert('L') if is_mask else Image.open(file_path)

def split_dataset(train_size: float = 0.8, shuffle: bool = True):
    # Get files
    images = np.load(os.path.join(datasetConfig.dir_base, datasetConfig.image_file))
    labels = np.load(os.path.join(datasetConfig.dir_base, datasetConfig.label_file))

    assert len(images) == len(labels), "Número de imagens e rótulos não são iguais."

    # Shuffle data if needed
    if shuffle:
        indices = np.random.permutation(len(images))
        images = images[indices]
        labels = labels[indices]

    # Calculate train size
    train_size = int(train_size * len(images))

    # Split dataset
    train_dataset = images[:train_size], labels[:train_size]
    test_dataset = images[train_size:], labels[train_size:]

    return train_dataset, test_dataset

def save_weights(model, model_name, dir_base):
    print("New best model.")
    # Save the best model
    weights = "{}/{}.pt".format(dir_base,str("best") + model_name)
    torch.save(model.state_dict(), weights)

def load_weights(model, path):
    return model.load_state_dict(torch.load(path))
                          
def get_color_maps():
    Label = namedtuple( "Label", [ "name", "id", "color"])
    maps = [ 
        Label("direct", 0, (0, 128, 0)),        # green
        Label("alternative", 1, (255, 200, 0)),  # yellow
        Label("background", 2, (0, 0, 0)),        # black          
    ]
    return np.array([p.color for p in maps if (p.id != -1 and p.id != 255)])

def show(image, label, color_maps):
    # plot sample image
    _, (ax0, ax1) = plt.subplots(1,2, figsize=(20,40))

    # Image
    ax0.imshow(image.permute(1, 2,0))
    ax0.set_title("Image")

    # Label
    ax1.imshow(color_maps[label])
    ax1.set_title("Label")

    plt.show()

def show_prediction(image, label, predicted, id_to_color, device="cuda"):
    
    inverse_transform = transforms.Compose([
        transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))
    ])
    
    image_base = torch.tensor(image, device=device).permute(2, 0, 1).squeeze(1)
    image_base = inverse_transform(image_base)
    image_base = image_base.permute(1, 2, 0).cpu().numpy()

    # Converter o dicionário 'colors' para uma matriz numpy
    cm_labels = np.array((id_to_color[predicted]), dtype=np.float32)
    overlay_image = cv2.addWeighted(image_base, 1, cm_labels, 0.1, 0)

    _, (axes0, axes1, axes2, axes3) = plt.subplots(1, 4, figsize=(20,10))
    axes0.imshow(image_base)
    axes0.set_title("Image")
    axes0.axis('off')

    axes1.imshow(id_to_color[label.cpu().detach().numpy()])
    axes1.set_title("Groundtruth")
    axes1.axis('off')

    axes2.imshow(id_to_color[predicted])
    axes2.set_title("Predicted")
    axes2.axis('off')

    axes3.imshow(np.clip(overlay_image, 0, 1))
    axes3.set_title("Mask")
    axes3.axis('off')

    plt.show()
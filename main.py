
# ********************************** Imports **********************************

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import segmentation_models_pytorch as smp
import logging
import argparse
import numpy as np

# dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# project
from src.core.config import DatasetConfig, HyperParameters
from src.test import testing
from src.train import evaluating, training
from src.utils.helpers import get_color_maps, load_weights, save_weights, show_prediction, split_dataset
from src.dataset.kitti import KittiSegDataset
from src.models.unet import UNet
from src.core.metrics import meanIoU 

torch.cuda.empty_cache()

device = "cuda" if torch.cuda.is_available() else "cpu"

hyper_parameters = HyperParameters()
id_to_color = get_color_maps()

# find optimal backend for performing convolutions 
torch.backends.cudnn.benchmark = True


def train_fn(config):

    # ********************************** Conf **********************************

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    # ********************************** Datset **********************************
    train_files_list, val_files_list = split_dataset(train_size=0.8)

    #https://albumentations.ai/docs/examples/pytorch_semantic_segmentation/
    # transforms
    train_transform = A.Compose(
        [
            A.Resize(config.image_size, config.image_size),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    val_and_test_transform = A.Compose(
        [A.Resize(config.image_size, config.image_size), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
    )

    train_dataset = KittiSegDataset(train_files_list, transform=train_transform)
    val_dataset = KittiSegDataset(val_files_list, transform=val_and_test_transform)
    
    # Model instance
    model = UNet(in_channels=3, out_channels=hyper_parameters.n_classes, layer_channels=[64, 128, 256, 512]).to(device)
    model.to(device)
    #TODO: fuse

    # ********************************** Hyperparameters **********************************
    num_worker = 4 * int(torch.cuda.device_count())
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=hyper_parameters.batch_size,drop_last=True, num_workers=num_worker, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,batch_size=hyper_parameters.batch_size, num_workers=num_worker, pin_memory=True)
    
    optimizer = optim.Adam(model.parameters(), lr=hyper_parameters.max_lr)
    scheduler = OneCycleLR(optimizer, max_lr= hyper_parameters.max_lr, epochs = hyper_parameters.n_epochs, steps_per_epoch = 2*(len(train_dataloader)), 
                        pct_start=0.3, div_factor=10, anneal_strategy='cos')
    
    # reference : https://smp.readthedocs.io/en/latest/losses.html
    criterion = smp.losses.DiceLoss('multiclass', classes=np.arange(0, hyper_parameters.n_classes), log_loss = True, smooth=1.0)

    # aux
    best_metric = 0

    # Treinamento e Validacao
    for epoch in range(hyper_parameters.n_epochs):
        torch.cuda.empty_cache()

        logger.info('*********** Epoch {} *********** \n'.format(int(epoch)+1))        
        
        # Training
        train_loss = training(model, train_dataloader, criterion, scheduler, optimizer, device)
        logger.info('Training loss: {}'.format(str(train_loss)))
        
        # Validation
        evaluation_loss, evaluation_metric = evaluating(model, val_dataloader, criterion, meanIoU, hyper_parameters.n_classes, device=device)
        logger.info('Evaluation loss: {} and Evaluation metric: {} \n'.format(str(evaluation_loss), str(evaluation_metric)))

        # Saving the best model according to evaluation_metric
        if best_metric < evaluation_metric:
            save_weights(model, hyper_parameters.file_name, hyper_parameters.dir_base)
            best_metric = evaluation_metric

        # Clean cuda cache
        torch.cuda.empty_cache()

    logger.info("\n Using the arc {} the best val loss in {} epochs was {}.".format(hyper_parameters.file_name, hyper_parameters.n_epochs, evaluation_metric))
    
    del model
    torch.cuda.empty_cache()


def test_fn(config):

    _, val_files_list = split_dataset(train_size=0.8) # just for testing
    val_and_test_transform = A.Compose(
        [A.Resize(config.image_size, config.image_size), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
    )
    test_dataset = KittiSegDataset(val_files_list, transform=val_and_test_transform)

    # Load model
    model = UNet(in_channels=3, out_channels=hyper_parameters.n_classes, layer_channels=[64, 128, 256, 512]).to(device)
    load_weights(model, hyper_parameters.weights_path)
    model.to(device)

    image, label, predicted = testing(model, test_dataset, device=device)

    show_prediction(image, label, predicted, id_to_color)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--op', default="train", choices=["train", "test"])
    parser = parser.parse_args()

    config = DatasetConfig()
 
    if parser.op == "train":
        train_fn(config)
    
    test_fn(config)
import numpy as np
import torch 
from tqdm import tqdm

min_val_loss = np.Inf
best_validation_metric = np.Inf


def training(model, train_dataloader, criterion, scheduler, optimizer, device):
    
    torch.cuda.empty_cache()
    
    # Training
    model.train()
    
    train_loss = 0.0

    for inputs, labels in tqdm(train_dataloader, total=len(train_dataloader)):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        y_preds = model(inputs)

        loss = criterion(y_preds, labels)
        train_loss += loss.item()
            
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # adjust learning rate
        if scheduler is not None:
            scheduler.step()
        
    # compute per batch losses, metric value
    train_loss = train_loss / len(train_dataloader)

    return train_loss

def evaluating(model, dataloader, criterion, metric_class, num_classes, device):
    torch.cuda.empty_cache()

    model.eval()
    total_loss = 0.0
    metric_object = metric_class(num_classes)
    
    with torch.no_grad():
        
        for inputs, labels in tqdm(dataloader, total=len(dataloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)                
            y_preds = model(inputs)

            # calculate loss
            loss = criterion(y_preds, labels)
            total_loss += loss.item()

            # update batch metric information            
            metric_object.update(y_preds.cpu().detach(), labels.cpu().detach())

    evaluation_loss = total_loss / len(dataloader)
    evaluation_metric = metric_object.compute()
    return evaluation_loss, evaluation_metric
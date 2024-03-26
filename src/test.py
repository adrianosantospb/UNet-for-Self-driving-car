import torch
import numpy as np
import random

def testing(model, test_dataset, device="cuda"):
    model.eval()
    
    with torch.no_grad():
        
        idx  = random.randrange(len(test_dataset))
        image, label = test_dataset[idx]
        inputImage = image.to(device)
        image_base = inputImage.permute(1, 2, 0).cpu().detach().numpy().copy()
                    
        y_pred = torch.argmax(model(inputImage.unsqueeze(0)), dim=1).squeeze(0)
        predicted = y_pred.cpu().detach().numpy()
        
        # Use a função 'np.where' para substituir valores maiores que 3 por 0
        predicted = np.where(predicted > 2, 0, predicted)

        return image_base, label, predicted
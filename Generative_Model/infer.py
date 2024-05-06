import os

import torch
from torchvision.utils import save_image
from Model import Model


def generate_images(net= Model(),
                    path_to_weigths ="checkpoint2.pth",
                    predict_n_images=50
                    ):
  
    net.load_state_dict(torch.load(path_to_weigths, map_location="cpu")["model"])
   
    
    os.makedirs("results2", exist_ok=True)
    for index in range(predict_n_images):
        net.eval() 
        # call model.eval() before running the forward pass. This will switch the batch normalization layer to evaluation mode, where it uses running statistics computed during training 
        # instead of # batch statistics
        image = net().squeeze()
       
    
        save_image(image, f'results2/{index}.png')
  

if __name__ == '__main__':
    generate_images()
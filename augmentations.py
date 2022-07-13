import torch.nn as nn
import torchvision.transforms as T, GaussianBlur

class SimSiamTransform():
    def __init__(self, image_size, device = 'cuda'):
        p_blur = 0.5 if image_size > 32 else 0 # exclude cifar
        # the paper didn't specify this, feel free to change this value
        # I use the setting from simclr which is 50% chance applying the gaussian blur
        # the 32 is prepared for cifar training where they disabled gaussian blur
        self.transform = nn.Sequential(
            #T.RandomApply([T.RandomRotation(30)], p=0.8),
            #T.RandomApply([T.RandomAffine(0, translate=(0.1, 0.2))], p=0.8),
            T.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([\
                T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=p_blur),
                ).to(device)
    def __call__(self, x1, x2 = None):
        x1 = self.transform(x1)
        if x2 == None:
            x2 = self.transform(x1)
        else :
            x2 = self.transform(x2)
        return x1, x2 
    

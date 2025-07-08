from agvbench.models.augments.basic import (randomblur, cutout, gridmask, spnoise, ricap, yoco)
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import random
from typing import List, Tuple
import numpy as np
import torch.nn.functional as F


def main():
    
    transform_list = [
        transforms.Resize(224)
    ]
    # The visualization and model need different transforms
    transform_vis  = transforms.Compose(transform_list)
    
    img = Image.open("demo/demo.jpg")
    img = transform_vis(img)

    img = randomblur(img)
    img = Image.fromarray(np.uint8(img * 255))
    img.save('randomblur.png')
    raise ValueError("debugging")
    img = cutout(img)
    img = gridmask(img)
    img = spnoise(img)
    img = ricap(img)
    img = yoco(img)

    # Display or save the processed images as needed
    # ...


if __name__=="__main__":
    main()
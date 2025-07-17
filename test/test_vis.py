from agvbench.models.augments.basic import (randomblur, cutout, gridmask, spnoise, ricap, yoco, smdwt_pca, randnquant)
from agvbench.models.augments.basic.softaugment import softaugment
from agvbench.models.augments.mixups import augmix
from torchvision import transforms
from PIL import Image
import torch


def main():
    
    transform_list = [
        transforms.Resize(224),
        transforms.ToTensor(),
    ]
    # The visualization and model need different transforms
    transform_vis  = transforms.Compose(transform_list)
    
    img = Image.open("demo/demo.jpg")
    img = transform_vis(img).unsqueeze(0).cuda()

    # img = randomblur(img)
    # img = cutout(img)
    img = gridmask(img)
    # img = spnoise(img)
    # img = ricap(img)
    # img = yoco(img)
    # img, _ = augmix(img)
    # img, lam = softaugment(img)
    # img = smdwt_pca(img)
    # img = randnquant(img)
    save_image = transforms.ToPILImage()
    img = save_image(img.squeeze(0))
    img = img.convert("RGB")
    img.save('gridmask.png')
    raise ValueError("debugging")


if __name__=="__main__":
    main()
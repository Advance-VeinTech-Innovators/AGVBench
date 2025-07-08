from agvbench.models.augments.basic import (randomblur, cutout, gridmask, spnoise, ricap, yoco)
from agvbench.models.augments.mixups import augmix
from torchvision import transforms
from PIL import Image


def main():
    
    transform_list = [
        transforms.Resize(224),
        transforms.ToTensor(),
    ]
    # The visualization and model need different transforms
    transform_vis  = transforms.Compose(transform_list)
    
    img = Image.open("demo/demo.jpg")
    img = transform_vis(img).unsqueeze(0)

    # img = randomblur(img)
    # img = cutout(img)
    # img = gridmask(img)
    # img = spnoise(img)
    # img = ricap(img)
    # img = yoco(img)
    # img, _ = augmix(img)
    # save_image = transforms.ToPILImage()
    # img = save_image(img.squeeze(0))
    # img = img.convert("RGB")
    # # Save the processed image
    # img.save('augmix.png')
    # raise ValueError("debugging")


if __name__=="__main__":
    main()
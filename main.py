from argparse import ArgumentParser, BooleanOptionalAction

import torch
from torch.optim import Adam
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

from model import VGG
from config import *

parser = ArgumentParser()
parser.add_argument('--org-img', type=str, required=True,
                    help='The path of original image to apply the style on')
parser.add_argument('--style-img', type=str, required=True,
                    help='The path of style image to apply on the original image')
parser.add_argument('--steps', type=int, default=6000,
                    help='Total steps to modify the original image')
parser.add_argument('--save-samples', type=bool, default=True,
                    action=BooleanOptionalAction, help='Save sample images or not')

args = parser.parse_args()

img_loader = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    # transforms.Normalize((.5, .5, .5), (.5, .5))
])

TOTAL_STEPS = args.steps

def load_img(img_name):
    img = Image.open(img_name).convert('RGB')
    img = img_loader(img).unsqueeze(0)
    return img.to(DEVICE)

def main():
    original_img: torch.Tensor = load_img(args.org_img)
    style_img: torch.Tensor = load_img(args.style_img)
    # generated = torch.randn(original_img.shape, device=device, requires_grad=True)
    generated = original_img.clone().requires_grad_(True).to(DEVICE)


    model = VGG().to(DEVICE).eval()
    optim = Adam([generated], lr=LR)

    gen_feature: torch.Tensor
    org_feature: torch.Tensor
    style_feature: torch.Tensor
    loop = tqdm(range(TOTAL_STEPS))
    for step in loop:
        generated_features = model(generated)
        original_features = model(original_img)
        style_features = model(style_img)

        style_loss = original_loss = 0
        for gen_feature, org_feature, style_feature in zip(generated_features, original_features, style_features):
            batch_size, channels, height, width = gen_feature.shape
            original_loss += torch.mean((gen_feature - org_feature) ** 2)

            # Compute Gram Matrix
            G = gen_feature.view(channels, height * width).mm(
                gen_feature.view(channels, height * width).t()
            )
            A = style_feature.view(channels, height * width).mm(
                style_feature.view(channels, height * width).t()
            )
            style_loss += torch.mean((G - A) ** 2)

        total_loss: torch.Tensor = ALPHA * original_loss + BETA * style_loss
        optim.zero_grad()
        total_loss.backward()
        optim.step()

        if step % 200 == 0:
            # print(f'Total Loss: {total_loss.item()}')
            loop.set_postfix(total_loss=total_loss.item())
            if args.save_samples:
                save_image(generated, f'images/generated/gen_img_{step}.png')
    
    save_image(generated, f'images/generated/gen_img_{TOTAL_STEPS}.png')

if __name__ == '__main__':
    main()

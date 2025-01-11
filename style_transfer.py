import torch
from torch import optim
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        # Load pre-trained VGG19 from torchvision, and get only the 'features' part
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.vgg_features = vgg_pretrained_features  # Load pre-trained features

        self.content_layers = ['conv_4']  # Use 'conv_4' layer for content loss
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']  # For style loss
        self.model_layers = self.get_vgg_layers()

    def forward(self, x):
        content_features, style_features = {}, {}
        for name, layer in self.model_layers:
            x = layer(x)
            if name in self.content_layers:
                content_features[name] = x
            if name in self.style_layers:
                style_features[name] = x
        return content_features, style_features

    def get_vgg_layers(self):
        layers = []
        name_map = {
            0: 'conv_1', 5: 'conv_2', 10: 'conv_3', 19: 'conv_4', 28: 'conv_5'
        }
        for name, layer in self.vgg_features._modules.items():
            layers.append((name_map.get(int(name), f'other_{name}'), layer))
        return layers


def gram_matrix(input_tensor):
    b, c, h, w = input_tensor.size()
    features = input_tensor.view(b, c, h * w)
    G = torch.bmm(features, features.transpose(1, 2))
    return G.div(c * h * w)


def blended_style_loss(style_features1, style_features2, target_features, style_weight1=0.5, style_weight2=0.5):
    style_loss = 0
    for layer in style_features1:
        target_gram = gram_matrix(target_features[layer])
        style_gram1 = gram_matrix(style_features1[layer])
        style_gram2 = gram_matrix(style_features2[layer])
        style_gram = style_weight1 * style_gram1 + style_weight2 * style_gram2
        style_loss += nn.functional.mse_loss(target_gram, style_gram)
    return style_loss


def content_loss(content_features, target_features, content_weight=1):
    loss = 0
    for layer in content_features:
        loss += nn.functional.mse_loss(target_features[layer], content_features[layer]) * content_weight
    return loss


# Helper functions for loading images and saving them
def load_image(image_path, size=512, device='cuda'):
    image = Image.open(image_path)
    loader = transforms.Compose([
        transforms.Resize(size),  # Resize the image
        transforms.ToTensor(),  # Convert to tensor
        transforms.Lambda(lambda x: x.mul(255))  # Multiply by 255 to match expected range
    ])
    image_tensor = loader(image).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)
    return image_tensor


def save_image(tensor, filename):
    img = tensor.squeeze(0).detach().cpu()
    img = transforms.ToPILImage()(img)
    img.save(filename)


# Neural style transfer
def neural_style_transfer(content_img_path, style_img_path1, style_img_path2):
    num_steps=300
    content_weight=100
    style_weight=1e4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert num_steps to an integer, just in case it's passed as a string
    num_steps = int(num_steps)

    # Load images
    content_img = load_image(content_img_path, device=device)
    style_img1 = load_image(style_img_path1, device=device)
    style_img2 = load_image(style_img_path2, device=device)

    # Initialize the VGG model with pre-loaded weights
    vgg = VGG().to(device).eval()

    # Extract features for the content and style images
    content_features, _ = vgg(content_img)
    _, style_features1 = vgg(style_img1)
    _, style_features2 = vgg(style_img2)

    # Initialize target image (copy of content image)
    target_img = content_img.clone().requires_grad_(True)

    # Set up optimizer (L-BFGS for faster convergence)
    optimizer = optim.LBFGS([target_img])

    # Optimization loop
    run = [0]
    while run[0] <= num_steps:
        def closure():
            optimizer.zero_grad()

            # Get features of target image
            target_content_features, target_style_features = vgg(target_img)

            # Calculate the content loss
            c_loss = content_loss(content_features, target_content_features, content_weight)

            # Calculate the style loss (blended style)
            s_loss = blended_style_loss(style_features1, style_features2, target_style_features, style_weight1=0.5, style_weight2=0.5)

            # Total loss
            total_loss = c_loss + style_weight * s_loss
            total_loss.backward(retain_graph=True)  # Fix: retain graph for L-BFGS

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"Iteration {run[0]}, Total loss: {total_loss.item()}")
            return total_loss

        optimizer.step(closure)

    # Save the output image
    save_image(target_img, 'output_blended_style.png')

# MODIFIED FROM https://github.com/kevinzakka/clip_playground/blob/main/CLIP_GradCAM_Visualization.ipynb
import urllib.request
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import clip
import os
from PIL import Image
from scipy.ndimage import filters
from torch import nn
from open_clip import tokenize, create_model_and_transforms
import torchvision.transforms as T
import torch
from utils import (
    zero_shot_prompts,
    compute_binary_metric,
    compute_regression_metric,
    read_avi,
)


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalize to [0, 1]"""
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x

def getAttMap(img, attn_map, blur=True):

    # Ensure the grayscale image has three channels if it doesn't already
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = np.dstack((img, img, img))

    if blur:
        attn_map = filters.gaussian_filter(attn_map, 0.02 * max(img.shape[:2]))
    attn_map = normalize(attn_map)
    cmap = plt.get_cmap('jet') 
    attn_map_c = cmap(attn_map)[:, :, :3]  # Use the RGB channels of the colormap

    # Convert the grayscale image to float representation
    img_float = img.astype(float) / 255

    # Apply the heatmap (attn_map) as a colored overlay onto the grayscale image
    attn_map_overlay = 1 * (1 - attn_map.reshape(attn_map.shape + (1,))) * img_float + \
                       attn_map.reshape(attn_map.shape + (1,)) * attn_map_c

    # Ensure the resulting image is within proper bounds
    attn_map_overlay = np.clip(attn_map_overlay, 0, 1)
    
    return attn_map_overlay

def viz_attn(img, attn_map, blur=True):
    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[1].imshow(getAttMap(img, attn_map, blur))
    for ax in axes:
        ax.axis("off")
    
def load_image(img_path, resize=None):
    image = Image.open(img_path).convert("RGB")
    if resize is not None:
        image = image.resize((resize, resize))
    return np.asarray(image).astype(np.float32) / 255.


class Hook:
    """Attaches to a module and records its activations and gradients."""

    def __init__(self, module: nn.Module):
        self.data = None
        self.hook = module.register_forward_hook(self.save_grad)
        
    def save_grad(self, module, input, output):
        self.data = output
        output.requires_grad_(True)
        output.retain_grad()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()
        
    @property
    def activation(self) -> torch.Tensor:
        return self.data
    
    @property
    def gradient(self) -> torch.Tensor:
        return self.data.grad


# Reference: https://arxiv.org/abs/1610.02391
def gradCAM(
    model: nn.Module,
    input: torch.Tensor,
    target: torch.Tensor,
    layer: nn.Module
) -> torch.Tensor:
    # Zero out any gradients at the input.
    if input.grad is not None:
        input.grad.data.zero_()
        
    # Disable gradient settings.
    requires_grad = {}
    for name, param in model.named_parameters():
        requires_grad[name] = param.requires_grad
        param.requires_grad_(False)
        
    # Attach a hook to the model at the desired layer.
    assert isinstance(layer, nn.Module)
    with Hook(layer) as hook:        
        # Do a forward and backward pass.
        output = model(input).squeeze()
        output.backward(target)

        grad = hook.gradient.float()
        act = hook.activation.float()
    
        # Global average pool gradient across spatial dimension
        # to obtain importance weights.
        alpha = grad.mean(dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps over channel
        # dimension.
        gradcam = torch.sum(act * alpha, dim=1, keepdim=True)
        # We only want neurons with positive influence so we
        # clamp any negative ones.
        gradcam = torch.clamp(gradcam, min=0)

    # Resize gradcam to input resolution.
    gradcam = F.interpolate(
        gradcam,
        input.shape[2:],
        mode='bicubic',
        align_corners=False)
    
    # Restore gradient settings.
    for name, param in model.named_parameters():
        param.requires_grad_(requires_grad[name])
        
    return gradcam

if __name__=='__main__':

    # uses the CLIP BPE tokenizer, so it can't process an entire report at once.
    echo_clip, _, preprocess_val = create_model_and_transforms(
        "hf-hub:mkaichristensen/echo-clip", precision="bf16", device="cuda"
    )
    echo_clip.eval()

    numpy_test_video = read_avi(
        "example_video.avi",
        (224, 224),
    )
    test_video = torch.stack(
        [preprocess_val(T.ToPILImage()(frame)) for frame in numpy_test_video], dim=0
    )

    prompt="ECHO DENSITY IN LEFT VENTRICLE SUGGESTIVE OF CATHETER, PACER LEAD, OR ICD LEAD. "
    device='cuda'

    # get the first frame
    image_np = numpy_test_video[0]
    image_input = test_video[0].to(torch.bfloat16).to(device).unsqueeze(0)

    text_input = clip.tokenize([prompt]).to(device)
    tokenized_prompt = tokenize(prompt).cuda()
    prompt_embeddings = F.normalize(
        echo_clip.encode_text(tokenized_prompt), dim=-1
    ).squeeze()
    attn_map = gradCAM(
        echo_clip.visual,
        image_input,
        prompt_embeddings,
        echo_clip.visual.trunk.stages[-1].blocks[-1]
    )
    attn=attn_map.squeeze().detach().cpu().numpy()

    viz_attn(image_np, attn)
    # or just plt.show() if have UI
    plt.savefig(f"GradCamExample.png",dpi=300)
    plt.clf()
    print(f"Your GradCamExample was saved in {os.path.realpath(__file__)}/GradCamExample.png")
from open_clip import tokenize, create_model_and_transforms
import torchvision.transforms as T
import torch
import torch.nn.functional as F
from utils import (
    zero_shot_prompts,
    compute_binary_metric,
    compute_regression_metric,
    read_avi,
)

# You'll need to log in to the HuggingFace hub CLI to download the models
# You can do this with the terminal command "huggingface-cli login"
# You'll be asked to paste your HuggingFace API token, which you can find at https://huggingface.co/settings/token

# Use EchoCLIP for zero-shot tasks like ejection fraction prediction
# or pacemaker detection. It has a short context window because it
# uses the CLIP BPE tokenizer, so it can't process an entire report at once.
echo_clip, _, preprocess_val = create_model_and_transforms(
    "hf-hub:mkaichristensen/echo-clip", precision="bf16", device="cuda"
)

# We'll use random noise in the shape of a 10-frame video in this example, but you can use any image
# We'll load a sample echo video and preprocess its frames.
test_video = read_avi(
    "example_video.avi",
    (224, 224),
)
test_video = torch.stack(
    [preprocess_val(T.ToPILImage()(frame)) for frame in test_video], dim=0
)
test_video = test_video[0:min(40, len(test_video)):2]
test_video = test_video.cuda()
test_video = test_video.to(torch.bfloat16)

# Be sure to normalize the CLIP embedding after calculating it to make
# cosine similarity between embeddings easier to calculate.
test_video_embedding = F.normalize(echo_clip.encode_image(test_video), dim=-1)

# Add in a batch dimension because the zero-shot functions expect one
test_video_embedding = test_video_embedding.unsqueeze(0)


# To perform zero-shot prediction on our "echo" image, we'll need
# prompts that describe the task we want to perform. For example,
# to zero-shot detect pacemakers, we'll use the following prompts
pacemaker_prompts = zero_shot_prompts["pacemaker"]
print(pacemaker_prompts)

# We'll use the CLIP BPE tokenizer to tokenize the prompts
pacemaker_prompts = tokenize(pacemaker_prompts).cuda()
print(pacemaker_prompts)

# Now we can encode the prompts into embeddings
pacemaker_prompt_embeddings = F.normalize(
    echo_clip.encode_text(pacemaker_prompts), dim=-1
)
print(pacemaker_prompt_embeddings.shape)

# Now we can compute the similarity between the video and the prompts
# to get a prediction for whether the video contains a pacemaker. It's
# important to note that this prediction is not calibrated, and can
# range from -1 to 1.
pacemaker_predictions = compute_binary_metric(
    test_video_embedding, pacemaker_prompt_embeddings
)

# If we use a pacemaker detection threshold calibrated using its F1 score on
# our test set, we can get a proper true/false prediction prediction.
f1_calibrated_threshold = 0.298
print(f"Pacemaker detected: {pacemaker_predictions.item() > f1_calibrated_threshold}")


# We can also do the same thing for predicting continuous values,
# like ejection fraction. We'll use the following prompts for
# zero-shot ejection fraction prediction:
ejection_fraction_prompts = zero_shot_prompts["ejection_fraction"]
print(ejection_fraction_prompts)

# However, since ejection fraction can range between 0 and 100,
# we'll need to make 100 versions of each prompt.
prompts = []
prompt_values = []

for prompt in ejection_fraction_prompts:
    for i in range(101):
        prompts.append(prompt.replace("<#>", str(i)))
        prompt_values.append(i)

ejection_fraction_prompts = prompts

# We'll once again tokenize and embed the prompts
ejection_fraction_prompts = tokenize(ejection_fraction_prompts).cuda()
ejection_fraction_embeddings = F.normalize(
    echo_clip.encode_text(ejection_fraction_prompts), dim=-1
)

# And we'll compute the similarity between the image and the prompts
# to get a prediction for the ejection fraction.
ejection_fraction_predictions = compute_regression_metric(
    test_video_embedding, ejection_fraction_embeddings, prompt_values
)
print(f"Predicted ejection fraction is {ejection_fraction_predictions.item():.1f}%")

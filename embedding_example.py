from open_clip import create_model_and_transforms
from template_tokenizer import template_tokenize
import torchvision.transforms as T
import torch
import torch.nn.functional as F
from utils import read_avi

# You'll need to log in to the HuggingFace hub CLI to download the models
# You can do this with the terminal command "huggingface-cli login"
# You'll be asked to paste your HuggingFace API token, which you can find at https://huggingface.co/settings/token

# Use EchoCLIP-R for retrieval-based tasks where you want to find
# the similarity between two echos, like in patient identification or
# echo report retrieval. It has a longer context window because it
# uses the template tokenizer, which we found increases its retrieval
# performance but decreases its performance on other zero-shot tasks.
echo_clip_r, _, preprocess_val = create_model_and_transforms(
    "hf-hub:mkaichristensen/echo-clip-r", precision="bf16", device="cuda"
)

# We'll load a sample echo video and preprocess its frames.
test_video = read_avi(
    "example_video.avi",
    (224, 224),
)
test_video = torch.stack(
    [preprocess_val(T.ToPILImage()(frame)) for frame in test_video], dim=0
)
test_video = test_video.cuda()
test_video = test_video.to(torch.bfloat16)

# Be sure to normalize the CLIP embeddings after calculating them to make
# cosine similarity between embeddings easier to calculate.
test_video_embedding = F.normalize(echo_clip_r.encode_image(test_video), dim=-1)

# To get a single embedding for the entire video, we'll take the mean
# of the 10 frame embeddings.
test_video_embedding = test_video_embedding.mean(dim=0, keepdim=True)

# We'll now load an excerpt of the report associated with our echo
# and tokenize it using the template tokenizer.
with open("example_report.txt", "r") as f:
    test_report = f.read()

template_tokens = template_tokenize(test_report)
template_tokens = torch.tensor(template_tokens, dtype=torch.long).unsqueeze(0).cuda()
print(template_tokens)

# We can then embed the report using EchoCLIP-R.
test_report_embedding = F.normalize(echo_clip_r.encode_text(template_tokens), dim=-1)

print(test_report_embedding.shape)
print(test_video_embedding.shape)

# Since both embeddings are normalized, we can just take the dot product
# to get the cosine similarity between them.
similarity = (test_report_embedding @ test_video_embedding.T).squeeze(0)
print(similarity.item())

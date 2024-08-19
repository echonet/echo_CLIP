# EchoCLIP: A Multimodal Foundation Model For Echocardiography

EchoCLIP is a multimodal foundation model for echocardiography. It is finetuned from CLIP weights on a dataset of >1M pairs of echocardiogram images and their associated expert interpretation text. It can be used for semantic search amongst echo videos as well as zero-shot prediction on a wide range of clinically relevant tasks. For more details, see our paper:

[https://www.nature.com/articles/s41591-024-02959-y](https://www.nature.com/articles/s41591-024-02959-y)
<!-- [Multimodal Foundation Models For Echocardiogram Interpretation](https://arxiv.org/abs/) -->

## Quickstart

This repo contains example code for loading and using EchoCLIP and its long-context variant, EchoCLIP-R. To get started, clone this repo and navigate into it. Then, create a new `conda` environment and install the required packages:

```
git clone https://github.com/echonet/echo_CLIP
cd echo_CLIP
conda env create -n echo-clip
conda activate echo-clip
python -m pip install -r requirements.txt
```
You should now be able to run `embedding_example.py` and `zero_shot_example.py`.

## Repo contents

* `embedding_example.py` demonstrates how to load EchoCLIP-R's weights and use them to calculate the similarity between an example echocardiogram and example report text.
* `zero_shot_example.py` demonstrates how to load EchoCLIP's weights and use them to perform zero-shot pacemaker identification and zero-shot ejection fraction prediction.
* `utils.py` contains implementations of our methods for performing zero-shot binary classification and zero-shot regression. The functions used in `zero_shot_example.py` are defined in this file. The prompts we use for the zero-shot tasks in our paper are all available here. Additionally, this file contains regexes for cleaning and preparing report text before it is tokenized.
* `template_tokenizer.py` contains the implementation of our custom echocardiography report tokenizer, which is designed to compress Cedars-Sinai echo reports into a small number of tokens.
* `template_vocab.txt` contains a vocabulary of 770 words and phrases constructed from the template file our cardiologists use to create their reports. This vocabulary is used by our template tokenizer to efficiently tokenize long reports.
* `blank_wordpiece.tokenizer` is a default config file for initializing a WordPiece tokenizer using HuggingFace's `tokenizers` library. We use it to initialize our custom tokenizer.

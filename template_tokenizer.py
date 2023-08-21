# +
from tokenizers import Tokenizer
import re
from utils import extract_variables, clean_text

template_tokenizer = Tokenizer.from_file("blank_wordpiece.tokenizer")
with open(
    "template_vocab.txt",
    "r",
) as f:
    template_vocab = [l.rstrip("\n") for l in f.readlines()]

template_tokenizer.add_tokens(template_vocab)

variable_tokenizer = Tokenizer.from_file("blank_wordpiece.tokenizer")
variable_vocab = ["[UNK]"]
variable_vocab.extend([str(i) for i in range(101)])
variable_vocab.extend(
    ["‡", "MILD", "MODERATE", "SEVERE", "MILD/MODERATE", "MODERATE/SEVERE"]
)
variable_vocab.extend(["." + str(i) for i in range(10)])
variable_vocab.extend(["0" + str(i) for i in range(10)])
variable_tokenizer.add_tokens(variable_vocab)

max_token_id = (
    len(template_tokenizer.get_vocab()) + len(variable_tokenizer.get_vocab()) - 1
)
BOS_token_id = max_token_id + 1
EOS_token_id = max_token_id + 2

var_symbol = re.compile(r"<#>")

# mapping from ID to number of missing numbers
var_counts = {}

for token, token_id in template_tokenizer.get_vocab().items():
    var_count = len(var_symbol.findall(token))
    var_counts[token_id] = var_count

template_vocab_len = len(template_tokenizer.get_vocab())

# Some final text cleaning replacements.
replacements = [
    (
        "THE INFERIOR VENA CAVA IS NORMAL IN SIZE AND SHOWS A NORMAL RESPIRATORY COLLAPSE, CONSISTENT WITH NORMAL RIGHT ATRIAL PRESSURE (<#>MMHG).",
        "THE INFERIOR VENA CAVA SHOWS A NORMAL RESPIRATORY COLLAPSE CONSISTENT WITH NORMAL RIGHT ATRIAL PRESSURE (<#>MMHG).",
    ),
    (
        "RESTING SEGMENTAL WALL MOTION ANALYSIS.:",
        "RESTING SEGMENTAL WALL MOTION ANALYSIS.",
    ),
]


def simple_replacement(text):
    for r in replacements:
        text = text.replace(r[0], r[1])
    return text


def pad_or_trunc(tokens, length):
    if len(tokens) > length:
        eos_token = max(tokens)
        tokens = tokens[:length]
        tokens[-1] = eos_token
    else:
        tokens = tokens + [0] * (length - len(tokens))
    return tokens


def template_tokenize(report):
    report = clean_text(report)

    # The "variables" (numbers, severity words) are removed from the report
    # and returned in a list. The report text has all variables replaced
    # with a placeholder symbol: <#>
    # The template tokenizer's vocabulary is made up of phrases with this
    # placeholder symbol.
    variables, report = extract_variables(report)
    report = simple_replacement(report)
    toks = template_tokenizer.encode(report)

    # Now we have a list of tokenized phrases, some of which had variables
    # extracted from them, and some of which didn't.
    var_mask = []
    unk = []
    for (start, end), tok, tok_id in zip(toks.offsets, toks.tokens, toks.ids):
        if not tok == "[UNK]":
            var_mask.extend([True] * var_counts[tok_id])
        else:
            source = report[start:end]
            unk.append((source, start))
            var_count = len(var_symbol.findall(source))
            var_mask.extend([False] * var_count)

    tok_ids = [t for t in toks.ids if not t == 0]
    matched_vars = [v for v, mask in zip(variables, var_mask) if mask]

    new_tok_ids = []
    for tok_id in tok_ids:
        var_count = var_counts[tok_id]
        recognized_vars = []
        for _ in range(var_count):
            recognized_vars.append(matched_vars.pop(0))

        # variables are joined with this weird char before being
        # tokenized so that the model can tell where one variable
        # ends and another begins
        var_string = "‡".join(recognized_vars)
        var_toks = variable_tokenizer.encode(var_string).ids
        var_toks = [v + template_vocab_len for v in var_toks]
        new_tok_ids.extend([tok_id, *var_toks])

    new_tok_ids = [BOS_token_id, *new_tok_ids, EOS_token_id]
    return pad_or_trunc(new_tok_ids, 77)


template_detokenizer = {k: f"[{v}]" for v, k in template_tokenizer.get_vocab().items()}

for k, v in variable_tokenizer.get_vocab().items():
    template_detokenizer[v + template_vocab_len] = f"<{k}>"
template_detokenizer[max_token_id + 1] = "[BOS]"
template_detokenizer[max_token_id + 2] = "[EOS]"


def template_detokenize(ids):
    return [template_detokenizer[i] for i in ids]

# %%
from transformers import LlamaConfig, LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset

data = load_dataset("bookcorpus")

# %%
from transformers import LlamaTokenizer, LlamaTokenizerFast

tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
# tokenizer = LlamaTokenizer.from_pretrained('decapoda-research/llama-7b-hf')

def tokenize(text):
    return tokenizer(text['text'])

tokenized = data.map(tokenize, batched=True)


# %%
config = LlamaConfig(
    hidden_size=1024,
    intermediate_size=4096,
    num_hidden_layers=2,
    num_attention_heads=8,
    max_position_embeddings=1024,
)

context_len = config.max_position_embeddings

def dataset_iter():
    dataset = iter(tokenized['train'])
    try:
        while True:
            output_ids = []
            while len(output_ids) < context_len:
                output_ids.extend(next(dataset)['input_ids'])
            yield output_ids[:context_len]
    except StopIteration:
        pass

data_iter = dataset_iter()


model = LlamaForCausalLM(config)

import torch

batch_size = 16
optimizer = torch.optim.Adam()

from transformers import Trainer

# %%
from transformers import DataCollatorForLanguageModeling

collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
collator([tokenized['train'][i] for i in range(4)])

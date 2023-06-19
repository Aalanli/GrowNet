# %%
import time
import torch

from transformers import LlamaConfig, LlamaTokenizer, LlamaForCausalLM
from transformers import LlamaTokenizer, LlamaTokenizerFast
from datasets import load_dataset

import wandb

data = load_dataset("bookcorpus")


tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
# tokenizer = LlamaTokenizer.from_pretrained('decapoda-research/llama-7b-hf')

def tokenize(text):
    return tokenizer(text['text'])

tokenized = data.map(tokenize, batched=True)

config = LlamaConfig(
    hidden_size=1024,
    intermediate_size=4096,
    num_hidden_layers=8,
    num_attention_heads=8,
    max_position_embeddings=1024,
)

context_len = config.max_position_embeddings
config.batch_size = 8


def dataset_iter():
    dataset = iter(tokenized['train'])
    try:
        while True:
            batches = []
            for _ in range(config.batch_size):
                output_ids = []
                while len(output_ids) < context_len:
                    output_ids.extend(next(dataset)['input_ids'])
                batches.append(output_ids[:context_len])
            yield torch.tensor(batches, dtype=torch.int64)
    except StopIteration:
        pass

def accuracy(logits, labels):
    pred = torch.argmax(logits, -1)
    return (labels == pred).float().mean()

model = LlamaForCausalLM(config).cuda()
optimizer = torch.optim.Adam(model.parameters())

epochs = 1
steps_per_log = 100

wandb.init(
    project="BeyondBackprop",
    config=config
)

n_batches = 0
for epoch in range(epochs):
    acc_loss = 0
    acc_acc = 0
    acc_latency = 0
    for data in dataset_iter():
        t1 = time.time()
        data = data.cuda()
        optimizer.zero_grad()
        outputs = model(data, labels=data)
        outputs.loss.backward()
        optimizer.step()
        
        acc_loss += float(outputs.loss)
        acc_acc += float(accuracy(outputs.logits[:, :-1], data[:, 1:]))
        acc_latency += time.time() - t1
        n_batches += 1
        if (n_batches + 1) % steps_per_log == 0:
            num_examples = (n_batches + 1) * config.batch_size
            wandb.log({"acc": acc_acc / steps_per_log, "loss": acc_loss / steps_per_log, "latency": acc_latency / steps_per_log}, step=num_examples)
            acc_loss = 0
            acc_acc = 0
            acc_latency = 0

wandb.finish()


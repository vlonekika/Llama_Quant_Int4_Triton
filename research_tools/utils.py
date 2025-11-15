import time
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from triton_compute_layer import QuantLinearINT4

def quantize_model(model):
  
    skip_patterns = ['lm_head', 'embed_tokens', 'norm', 'ln']
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(skip in name.lower() for skip in skip_patterns):
                continue
            
            if module.in_features < 256 or module.out_features < 256:
                continue
            
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            
            linear = QuantLinearINT4.transform_layer(module)
            setattr(parent, child_name, linear)
    
    return model


def model_memory(model: nn.Module, include_grad=True):

    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
        if include_grad:
            total_bytes += p.numel() * p.element_size()

    return total_bytes / (1024 ** 2)


def perplexity(model, tokenizer, dataset, samples=128, max_len=256):
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i in tqdm(range(min(samples, len(dataset)))):
            text = dataset[i]['text']
            if not text or len(text.strip()) < 10:
                continue
            
            inputs = tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=max_len,
                padding=False
            )

            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            
            if torch.isnan(outputs.loss) or torch.isinf(outputs.loss):
                continue
            
            token_count = attention_mask.sum().item()
            total_loss += outputs.loss.item() * token_count
            total_tokens += token_count

    perplexity = np.exp(total_loss / total_tokens)
    return perplexity

def compute_speed(model, tokenizer, dataset, batch_size=32, max_lens=[128, 512, 2048], samples=512):
   
    model.eval()
    device = model.device
    results = {}

    for batch_len in max_lens:
        times = []
        num_batches = min(samples // batch_size, len(dataset) // batch_size)

        for b in range(num_batches):
            batch_texts = [dataset[i]['text'] for i in range(b*batch_size, (b+1)*batch_size)]
            
            batch_texts = [t for t in batch_texts if t and len(t.strip()) > 10]
            if len(batch_texts) == 0:
                continue

            inputs = tokenizer(batch_texts, return_tensors='pt', truncation=True, max_length=batch_len, padding=True)
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)

            with torch.no_grad():
                _ = model(input_ids, attention_mask=attention_mask)

            torch.cuda.synchronize()
            start = time.time()
            with torch.no_grad():
                _ = model(input_ids, attention_mask=attention_mask)
            torch.cuda.synchronize()
            times.append(time.time() - start)

        if len(times) == 0:
            continue

        avg_time = np.mean(times)
        results[batch_size] = {'avg_time_s': avg_time, 'seq_len': batch_len, 'batch_size': batch_size}
        print(f"len: {batch_len} Batch {batch_size}: avg time {avg_time*1000:.2f}ms")
    
    return results
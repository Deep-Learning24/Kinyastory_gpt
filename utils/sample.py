'''
    code by TaeHwan Jung(@graykode),
    Modified by Claude Kwizera (@ckwizera)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
import torch
import torch.nn.functional as F
from tqdm import trange


def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)


def sample_sequence(model, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0,
                    device='cuda', sample=True):
    context_input_ids, context_attention_mask = context
    if start_token is None:
        assert context_input_ids is not None, 'Specify exactly one of start_token and context!'
        context_input_ids = torch.tensor(context_input_ids, device=device, dtype=torch.long).unsqueeze(0)
        context_input_ids = context_input_ids.repeat(batch_size, 1)
    else:
        assert context_input_ids is None, 'Specify exactly one of start_token and context!'
        context_input_ids = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)

    prev = context_input_ids
    output = context_input_ids
    
    past = None
    with torch.no_grad():
        for i in trange(length):
            logits= model(context_input_ids, context_attention_mask)
            #logits, past = model(prev, past=past)
            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, k=top_k)
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            output = torch.cat((output, prev), dim=1)
    return output

import torch


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        # crop the context_size
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]  # getting last time step

        probs = torch.softmax(logits, dim=-1)

        idx_next = torch.argmax(probs, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def get_model_details(model):
    """Returns Models Parametersand Size"""
    params = sum(p.numel() for p in model.parameters())

    total_bytes = params * 4
    total_size = total_bytes / (1024 * 1024)
    total_size = round(total_size, 2)
    return params, total_size

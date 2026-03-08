import torch


def format_to_alpaca(entry):
    """Formats Input Text response to alpaca Format"""
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


def collate(
    batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device="cpu"
):

    # longest sequence in batch
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, target_lst = [], []

    for item in batch:
        new_item = item.copy()

        # endoftext token
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

        # truncate last for input
        inputs = torch.tensor(padded[:-1])
        # shift to 1 right for targets
        targets = torch.tensor(padded[1:])

        # Replace all padding index token by ignore_index except the 1st one
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # Truncate for max sequence length if required
        if allowed_max_length:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        target_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(target_lst).to(device)
    return inputs_tensor, targets_tensor

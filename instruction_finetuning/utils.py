import logging

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


def loader_accuracy(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break

    return correct_predictions / num_examples


def text_to_token_ids(text, tokenizer, device):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = (
        torch.tensor(encoded).unsqueeze(0).to(device)
    )  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def batch_loss(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def loader_loss(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = batch_loss(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def plot_values(
    epochs_seen, examples_seen, train_values, val_values, label="loss", plot=False
):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # Create a second x-axis for examples seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(examples_seen, train_values, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig(f"{label}-plot.pdf")
    if plot:
        plt.show()


def calc_loss_batch(input_batch, target_batch, model, device):
    """Returns Cross Entropy loss"""
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


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


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context, set_train=True):
    model.eval()
    context_size = model.position_embeddings.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer, device=device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded, max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    if set_train:
        model.train()


def generate(
    model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None
):
    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits
            )

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if (
            idx_next == eos_id
        ):  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx


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


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
    device,
    cfg,
    writer=None,
):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    logging.basicConfig(
        filename="runtime.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    # Training Loop
    for epoch in range(num_epochs):
        model.train()

        pbar = tqdm(train_loader, desc=f"Epoch:{epoch + 1}/{num_epochs}")
        for batch_idx, (input_batch, target_batch) in enumerate(pbar):
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            pbar.set_postfix({"Loss": loss.item(), "Tokens Seen": tokens_seen})

            if writer:
                writer.add_scalar("Training Loss", loss.item(), global_step)
                writer.add_scalar("Tokens Seen", tokens_seen, global_step)
            # Evaluation step:
        # if global_step % eval_freq == 0:
        train_loss, val_loss = evaluate_model(
            model, train_loader, val_loader, device, eval_iter
        )

        train_perplexity = torch.exp(torch.tensor(train_loss))
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        track_tokens_seen.append(tokens_seen)
        print(
            f"Epoch:{epoch + 1} (Step:{global_step:06d}) "
            f"Train Loss:{train_loss:.3f},Val Loss:{val_loss:.3f},"
            f"Tokens Seen:{tokens_seen}, Train Preplex:{round(train_perplexity.item())}"
        )
        logging.info(
            f"Epoch:{epoch + 1} (Step:{global_step:06d}) "
            f"Train Loss:{train_loss:.3f},Val Loss:{val_loss:.3f},"
            f"Tokens Seen:{tokens_seen}, Train Preplex:{round(train_perplexity.item())}"
        )
        if writer:
            writer.add_scalar("train_loss_epoch", train_loss, epoch)
            writer.add_scalar("Validation_loss", val_loss, epoch)
            writer.add_scalar("Perplexity", round(train_perplexity.item()), epoch)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(start_context, tokenizer, device),
            max_new_tokens=25,
            context_size=cfg["context_length"],
            top_k=50,
            temperature=1,
        )

        torch.save(model.state_dict(), f"weights_{epoch + 1}.pth")
        generated_text = token_ids_to_text(token_ids, tokenizer)

        print(generated_text)
    torch.save(model.state_dict(), "weights_checkpoint.pth")
    return train_losses, val_losses, track_tokens_seen

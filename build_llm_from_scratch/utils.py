import matplotlib.pyplot as plt
import logging
import torch
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm


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


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def get_model_details(model):
    """Returns Models Parametersand Size"""
    params = sum(p.numel() for p in model.parameters())

    total_bytes = params * 4
    total_size = total_bytes / (1024 * 1024)
    total_size = round(total_size, 2)
    return params, total_size


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


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.position_embeddings.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded, max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(
        MaxNLocator(integer=True)
    )  # only show integer labels on x-axis

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig("loss-plot.pdf")
    plt.show()


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
        generate_and_print_sample(model, tokenizer, device, start_context)
    torch.save(model.state_dict(), "weights.pth")
    return train_losses, val_losses, track_tokens_seen

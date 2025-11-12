import matplotlib.pyplot as plt
import torch

inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your
        [0.55, 0.87, 0.66],  # Journey
        [0.57, 0.85, 0.64],  # Starts
        [0.22, 0.58, 0.33],  # with
        [0.77, 0.25, 0.10],  # one
        [0.05, 0.80, 0.55],  # step
    ]
)


def softmax_basic(x):
    """Basic softmax function"""
    return torch.exp(x) / torch.exp(x).sum(dim=0)


words = ["Your", "journey", "starts", "with", "one", "step"]
query = inputs[1]  # query is word journey

attention_score = torch.empty(inputs.shape[0])
print(attention_score)
"""
attention_score:
[ 2.3257e-38,  4.5642e-41, -6.3288e+12,  4.5640e-41,  2.3010e-38,4.5642e-41]
"""
for i, x_i in enumerate(inputs):
    print(f"------{words[i]}------")
    print(f"attention_score_{i} = x_{i}  {x_i} . query  {query}")
    attention_score[i] = torch.dot(x_i, query)
    print(
        f"attention_score_{i} = {attention_score[i]}"
    )  # dot product between each input embedding vector and query vector
print(f"final_attention_score = {attention_score}")

print(f"Basic_softmax_score: {softmax_basic(attention_score)}")
print(f"Basic_softmax_sum: {softmax_basic(attention_score).sum()}")

attention_weights = torch.softmax(attention_score, dim=0)
print(f"Torch Softmax:{attention_weights}")

attention_scores = inputs @ inputs.T


attention_weights = torch.softmax(attention_scores, dim=-1)
print(attention_weights)

context_vectors = attention_weights @ inputs

print(context_vectors)

# Basic concepts for creating LLM's from Scratch.

- Learning about Tokenization 
  - Word based
  - Byte-pair encoding

- Learning about Embeddings
  - A multi-dimensional vector space representing tokens
  - Tokens with similar context are closer
  - Catches and preserves the context
  - positional_embeddings preservers context w.r.t Position of token in Input

## Attention Mechanisms

- Simple Attention
  - Suppose we have a input = ["Hi","I","Am","Good"]
  - For each token or we call Query we will have a embedding of dim (n)
  - For calculating Attention we will have to calculate the dot product between query and each input token.
  - Dot product quantifies how much aligned are two vectors in the space.
  - Dot product determines the extent to which elements of input attend to each other or values to each other.
  - Higher the dot product -> Higher similarity and higher attention scores.

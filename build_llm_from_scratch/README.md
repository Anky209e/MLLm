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

- Self Attention
  - For self attention we want trainable weights that we can update and improve attention.
  - We have Three different weight matrix
    - W_q = Query
    - W_k = Key
    - W_v = Value
  - Input Dimension of these matrices need to be same as Embedding Dimension of out Input vectors.
  - Now for calculating:
    - Keys = Input . W_k
    - Queries = Input . W_q
    - Values = Input . W_v
  - Now we need to calculate Attention score:
    - attention_score = Queries . Transpose(Keys)
  - For calculating Attention weights We will have to scale these values and use softmax.
  - Before using Softmax We will scale then by root(key.shape[-1]) .
  - Dividing the attention Scores by root of embedding Dimension before putting them in softmax helps with:
    - Reduces the peaks of softmax and model won't be very overconfident:
    - Reduces Variance if we divide by the root of Dimension.
  - Finally we will calculate context vectors which is dot product of Attention_weights and Values

- Causal Attention
  - For Causal Attention we Mask the Values which we do not need or deemed to be not required.
  - We create a mask and replace values above diagonal with -inf.
  - Why -inf ?
    - We want to make sure that during softmax the other values are not effected by removed or extra values.
    - Prevents Data leakage

- Multi-Head Attention
  - We have multiple 2heads (n) of Causal Attention.
  - We will have (n) no of Q,K,V and we will process them parallel.
  - In the end we will have n no. of context vectors which we will concatenate and that will be our final context vector for input.

- Layer Normalisation
  - While Doing backprop on out network The gradients and become too small or large.
  - Its called as vanishing and exploding gradients which can result in unstable training dynamics.
  - Layer Normalisation prevents this.
  - As training Proceeds Input to each layer can change i.e internal covariate shift.
  - This delays convergence and layer Normalisation prevents this.

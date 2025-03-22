# Self-Attention-mechanism
This project contains original Self Attention Mechanism written in a Jupyter notebook (runnable on colab)


feat: add core SelfAttention module implementation

- Implement PyTorch module for scaled dot-product self-attention mechanism
- Initialize learnable query/key/value projection matrices
- Add dimension configuration parameters (row_dim, col_dim) for flexible tensor operations
- Include scaled similarity score calculation with √d_k normalization
- Implement attention probability computation via softmax
- Add value aggregation step for context-aware embeddings
- Document class and methods with shape annotations and references to "Attention Is All You Need" (Vaswani et al., 2017)

The module forms the foundation for transformer-based architectures and supports:
- Batch processing through ... in shape annotations
- Custom dimension configuration for non-standard input formats
- Numerical stability through gradient-safe scaling

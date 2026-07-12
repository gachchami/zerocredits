ZeroCredits is a token-efficient AI agent that classifies each task and routes it to the most suitable allowed Fireworks model.

## Runtime model policy

When the runner exposes only `minimax-m3` and `kimi-k2p7-code`:

- MiniMax M3 handles factual Q&A, math, sentiment, summarization, NER, and logic.
- Kimi K2 Code handles code debugging and code generation.
- MiniMax receives a larger provider completion budget because its hidden reasoning consumes completion tokens. This does not change the requested concise answer format.

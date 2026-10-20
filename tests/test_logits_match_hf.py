import pytest
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from quantlens.weights import load_pretrained

PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In the beginning, the universe was created.",
    "Machine learning models can be quantized to reduce memory usage.",
    "To be, or not to be, that is the question.",
    "Artificial intelligence is transforming every industry.",
    "Once upon a time in a land far away, there lived a king.",
    "Deep neural networks require careful optimization during training.",
    "The weather today is sunny with a mild breeze.",
    "Quantum computing leverages superposition and entanglement.",
    "She opened the book and began reading chapter one.",
    "Python is known for its readability and simplicity.",
    "Transformer architectures rely heavily on self-attention mechanisms.",
    "The capital of France is Paris, famous for the Eiffel Tower.",
    "He brewed a cup of black coffee before starting work.",
    "Linear algebra and calculus form the foundation of deep learning.",
    "The speed of light in a vacuum is approximately 300,000 kilometers per second.",
    "A journey of a thousand miles begins with a single step.",
    "Natural language processing has evolved rapidly in recent years.",
    "Music is the universal language of mankind.",
    "Exploring the depths of the ocean reveals extraordinary biodiversity.",
]

@pytest.mark.slow
def test_logits_match_hf():
    device = "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    hf_model.eval()

    custom_model = load_pretrained("gpt2").to(device)
    custom_model.eval()

    with torch.no_grad():
        for prompt in PROMPTS:
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)

            hf_outputs = hf_model(input_ids)
            hf_logits = hf_outputs.logits.to(torch.float32)

            custom_logits = custom_model(input_ids).to(torch.float32)

            torch.testing.assert_close(custom_logits, hf_logits, atol=1e-4, rtol=1e-4)

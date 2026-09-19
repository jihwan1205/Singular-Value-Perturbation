import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from svp.models.coconut import CoconutAdapter

SPECIALS = ["<|endoftext|>", "<|start-latent|>", "<|latent|>", "<|end-latent|>"]
WORDS = [str(i) for i in range(30)] + ["a", "b", "how", "many", "has", "and", "#", "<unk>"]


def tiny_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {w: i for i, w in enumerate(SPECIALS + WORDS)}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|endoftext|>", pad_token="<|endoftext|>",
                                   unk_token="<unk>", additional_special_tokens=SPECIALS[1:])


class TinyCoconut(CoconutAdapter):
    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer
        tok = tokenizer.convert_tokens_to_ids
        self.latent_token_id, self.start_token_id, self.end_token_id = (
            tok("<|latent|>"), tok("<|start-latent|>"), tok("<|end-latent|>"))


def make_gpt2_adapter(seed: int = 0) -> TinyCoconut:
    torch.manual_seed(seed)
    tokenizer = tiny_tokenizer()
    config = GPT2Config(vocab_size=len(tokenizer), n_positions=128, n_embd=32, n_layer=2, n_head=2,
                        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
                        bos_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
    return TinyCoconut(GPT2LMHeadModel(config).eval(), tokenizer)


def make_llama_adapter(seed: int = 0) -> TinyCoconut:
    torch.manual_seed(seed)
    tokenizer = tiny_tokenizer()
    config = LlamaConfig(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128,
                         bos_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
    adapter = TinyCoconut(LlamaForCausalLM(config).eval(), tokenizer)
    adapter.arch = "llama"
    return adapter


@pytest.fixture
def gpt2_adapter():
    return make_gpt2_adapter()


@pytest.fixture
def llama_adapter():
    return make_llama_adapter()


QUESTIONS = ["a has 3 and b has 4 how many", "how many 12 and 7", "a 5 b 6 how many", "2 and 2"]

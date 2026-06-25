"""Hidden test for Task 5 — build_rm + rm_step (shape & smoke)."""
import torch
from transformers import AutoTokenizer

from tasks.task5_reward_head import build_rm, rm_step


def test_build_rm_forward_shape():
    """The RM accepts (batch, seq) input_ids and returns (batch,)
    scalar scores when called through rm_step.forward path."""
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rm = build_rm(pad_token_id=tok.pad_token_id)
    rm.eval()

    # Tiny batch.
    ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    attn = torch.ones_like(ids)
    out = rm(input_ids=ids, attention_mask=attn).logits.squeeze(-1)
    assert out.shape == (2,), f"expected shape (2,); got {out.shape}"


def test_rm_step_returns_finite_loss():
    """rm_step on a fixture batch returns a finite scalar loss + two
    score tensors of shape (batch,)."""
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rm = build_rm(pad_token_id=tok.pad_token_id)
    rm.train()
    batch = {
        "chosen_ids":    torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]),
        "chosen_attn":   torch.ones((2, 5), dtype=torch.long),
        "rejected_ids":  torch.tensor([[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]),
        "rejected_attn": torch.ones((2, 5), dtype=torch.long),
    }
    loss, sc, sr = rm_step(rm, batch)
    assert loss.dim() == 0, f"loss should be scalar; got shape {loss.shape}"
    assert torch.isfinite(loss).item(), f"loss not finite: {loss.item()}"
    assert sc.shape == (2,), f"chosen scores shape {sc.shape} != (2,)"
    assert sr.shape == (2,), f"rejected scores shape {sr.shape} != (2,)"


if __name__ == "__main__":
    test_build_rm_forward_shape()
    test_rm_step_returns_finite_loss()
    print("task 5 (build_rm + rm_step) — all tests passed")

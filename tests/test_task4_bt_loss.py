"""Hidden test for Task 4 — bt_loss."""
import torch

from tasks.task4_bt_loss import bt_loss


def test_bt_loss_values():
    cs = torch.tensor([1.0,  2.0, -1.0])
    rs = torch.tensor([0.0,  3.0, -1.0])
    out = bt_loss(cs, rs)
    assert out.shape == (3,), f"bt_loss should return shape (batch,); got {out.shape}"
    expected = torch.tensor([0.313262, 1.313262, 0.693147])
    assert torch.allclose(out, expected, atol=1e-5), \
        f"bt_loss wrong: got {out}, expected {expected}"


def test_bt_loss_margin_sign():
    out = bt_loss(torch.tensor([0.0]), torch.tensor([1.0]))
    assert out.item() > 1.0, f"loss should be > 1 when margin = -1; got {out.item()}"


if __name__ == "__main__":
    test_bt_loss_values()
    test_bt_loss_margin_sign()
    print("task 4 (bt_loss) — all tests passed")

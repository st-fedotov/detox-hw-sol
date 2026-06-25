"""Hidden test for Task 2 — dpo_loss."""
import torch

from tasks.task2_dpo_loss import dpo_loss


def test_dpo_loss_values():
    torch.manual_seed(0)
    pcl = torch.tensor([-12.0, -8.0,  -6.0])
    prl = torch.tensor([-15.0, -7.0, -10.0])
    rcl = torch.tensor([-13.0, -9.0,  -7.0])
    rrl = torch.tensor([-14.0, -6.0, -11.0])

    losses, cr, rr = dpo_loss(pcl, prl, rcl, rrl, beta=0.1)
    assert losses.shape == (3,)
    assert cr.shape == (3,)
    assert rr.shape == (3,)
    expected_loss = torch.tensor([0.598139, 0.598139, 0.693147])
    assert torch.allclose(losses, expected_loss, atol=1e-4), \
        f"loss wrong: {losses}, expected {expected_loss}"
    assert torch.allclose(cr, torch.tensor([0.1, 0.1, 0.1])), \
        f"chosen_rewards wrong: {cr}"
    assert torch.allclose(rr, torch.tensor([-0.1, -0.1, 0.1])), \
        f"rejected_rewards wrong: {rr}"


def test_dpo_loss_rewards_detached():
    pcl = torch.tensor([-12.0], requires_grad=True)
    prl = torch.tensor([-15.0], requires_grad=True)
    rcl = torch.tensor([-13.0])
    rrl = torch.tensor([-14.0])
    _, cr, rr = dpo_loss(pcl, prl, rcl, rrl, beta=0.1)
    assert not cr.requires_grad, "chosen_rewards must be detached"
    assert not rr.requires_grad, "rejected_rewards must be detached"


if __name__ == "__main__":
    test_dpo_loss_values()
    test_dpo_loss_rewards_detached()
    print("task 2 (dpo_loss) — all tests passed")

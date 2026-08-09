import torch

from src.icm import ICMModule


def test_single_transition_curiosity_is_nonzero_and_icm_updates():
    torch.manual_seed(0)
    icm = ICMModule(input_dim=16, action_dim=4)
    optimizer = torch.optim.Adam(icm.parameters(), lr=1e-3)
    state = torch.zeros((1, 16), dtype=torch.float32)
    next_state = torch.ones((1, 16), dtype=torch.float32)
    action = torch.tensor([1])

    before = [parameter.detach().clone() for parameter in icm.parameters()]
    curiosity, forward_loss, inverse_loss = icm(state, next_state, action)

    assert curiosity.shape == (1,)
    assert curiosity.item() > 0.0

    optimizer.zero_grad()
    (forward_loss + inverse_loss).backward()
    optimizer.step()

    assert any(
        not torch.equal(old, new.detach()) for old, new in zip(before, icm.parameters())
    )

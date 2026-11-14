"""Check Intel GPU forward/backward correctness before training (no robot access)."""

import json

import torch


def main():
    assert torch.xpu.is_available(), "Intel GPU is unavailable"
    a = torch.ones(1024, 1024, device="xpu", requires_grad=True)
    results = []
    for _ in range(5):
        loss = (a @ a).square().mean()
        loss.backward()
        torch.xpu.synchronize()
        result = {"loss": loss.item(), "gradient_mean": a.grad.mean().item()}
        results.append(result)
        assert result == {"loss": 1048576.0, "gradient_mean": 4.0}, result
        a.grad = None
    print(json.dumps({"torch": torch.__version__, "device": torch.xpu.get_device_name(0),
                      "status": "passed", "iterations": results}, indent=2))


if __name__ == "__main__":
    main()

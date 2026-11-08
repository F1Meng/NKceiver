import torch


def integrated_state_contributions(
    model, m, p, present, reference_m, targets, steps=64, return_genes=False
):
    device = next(model.parameters()).device
    model.eval()
    mm = torch.as_tensor(m, dtype=torch.float32, device=device)
    pp = torch.as_tensor(p, dtype=torch.float32, device=device)
    mask = torch.as_tensor(present, dtype=torch.bool, device=device)
    m0 = torch.as_tensor(reference_m, dtype=torch.float32, device=device).expand_as(
        mm
    ) * mask.unsqueeze(-1)
    target = torch.as_tensor(targets, dtype=torch.long, device=device)
    gm = torch.zeros_like(mm)

    def probability(x):
        o = model(state_expression=x, state_present=mask)
        out = o if isinstance(o, torch.Tensor) else o.leaf_logits
        return out.softmax(-1).gather(1, target[:, None]).squeeze(1)

    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.no_grad():
            baseline = probability(m0)
            actual = probability(mm)
        for index in range(steps + 1):
            alpha = index / steps
            x = (m0 + alpha * (mm - m0)).detach().requires_grad_(True)
            with torch.enable_grad():
                value = probability(x).sum()
                (dm,) = torch.autograd.grad(value, (x,), allow_unused=True)
            weight = 0.5 if index in (0, steps) else 1.0
            if dm is not None:
                gm.add_(dm.detach(), alpha=weight / steps)
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    gene_contribution = (mm - m0) * gm
    contrib_m = gene_contribution.sum(-1)
    contrib_p = torch.zeros_like(pp)
    residual = actual - baseline - contrib_m.sum(1) - contrib_p.sum(1)
    result = {
        "P": contrib_p.cpu().numpy(),
        "M": contrib_m.cpu().numpy(),
        "reference_probability": baseline.cpu().numpy(),
        "target_probability": actual.cpu().numpy(),
        "completeness_error": residual.cpu().numpy(),
        "integration_steps": steps,
    }
    if return_genes:
        result["M_gene"] = gene_contribution.cpu().numpy()
    return result

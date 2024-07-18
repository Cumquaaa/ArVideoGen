def dummy_metric(output, target):
    return ((output - target) ** 2).mean().item()

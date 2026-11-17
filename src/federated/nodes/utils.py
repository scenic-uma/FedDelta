def compute_theoretical_flops(alg, n: int, epochs: int = None, batch_size: int = None) -> float:
    """
    Theoretical floating-point operations of one local training round of `alg` on n samples.
    FedDelta depends only on n; the gradient-based methods also on the epochs and batch size.
    """
    if hasattr(alg, "erank_diagnostics"):          # FedDelta
        return alg.flops_info(n)["total_flops"]
    return alg.flops_info(n, epochs, batch_size)["total_flops"]

def node_theoretical_time(
    flops: float,
    capacity: int,
    flops_per_unit: float = 1e6
) -> float:
    """
    Convert a FLOP count into a simulated wall-clock time based on node capacity.

    Node capacity is an integer in [1, 100] representing relative compute power:
      - capacity = 100  →  full speed (flops_per_unit * 100 FLOPs / second)
      - capacity = 1    →  minimum speed (flops_per_unit * 1 FLOPs / second)

    This abstraction makes algorithm comparisons hardware-agnostic: two nodes
    with the same capacity always take the same simulated time for the same
    number of FLOPs, regardless of the host machine.

    Args:
        flops:          Estimated FLOPs for the local training step.
        capacity:       Node compute capacity in [1, 100].
        flops_per_unit: Base compute speed in FLOPs/second per capacity unit.
                        Adjust this value to calibrate the simulation to realistic
                        wall-clock times (default: 1e6 FLOPs per capacity unit).

    Returns:
        Simulated elapsed time in seconds (float).

    Raises:
        ValueError: If capacity is outside [1, 100].
    """
    if not (1 <= capacity <= 100):
        raise ValueError(
            f"capacity must be in [1, 100], got {capacity}."
        )

    effective_speed = flops_per_unit * capacity  # FLOPs / second for this node
    return flops / effective_speed
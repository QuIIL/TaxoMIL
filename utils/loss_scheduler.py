import math
from typing import Tuple
import argparse

class PeriodicLossScheduler:
    def __init__(
            self,
            total_epochs: int = 100,
            ce_weight: float = 1.0,
            hier_max: float = 0.5,
            ital_max: float = 0.3,
            cl_max: float = 0.1,
            n_cycles: int = 10,
    ):
        self.total_epochs = max(1, int(total_epochs))
        self.ce_weight = float(ce_weight)
        self.hier_max = float(hier_max)
        self.ital_max = float(ital_max)
        self.cl_max = float(cl_max)
        self.n_cycles = max(1, int(n_cycles))

    @staticmethod
    def _periodic_with_decay(
            x: float,
            max_weight: float,
            n_cycles: int,
    ) -> float:
        if max_weight <= 0.0:
            return 0.0
        wave = math.sin(math.pi * n_cycles * x) ** 2
        return max_weight * wave

    @staticmethod
    def _periodic_with_decay_inverted(
            x: float,
            max_weight: float,
            n_cycles: int,
    ) -> float:
        if max_weight <= 0.0:
            return 0.0
        wave = math.cos(math.pi * n_cycles * x) ** 2
        return max_weight * wave

    def get_weights(self, epoch: int) -> dict[str, float]:
        if self.total_epochs <= 1:
            x = 1.0
        else:
            x = max(0.0, min(1.0, epoch / (self.total_epochs - 1)))

        w_ce = self.ce_weight   # fixed weight

        # HIER, CL
        w_hier = self._periodic_with_decay_inverted(
            x, self.hier_max, self.n_cycles
        )
        w_cl = self._periodic_with_decay_inverted(
            x, self.cl_max, self.n_cycles
        )
        # ITAL: sin²
        w_ital = self._periodic_with_decay(
            x, self.ital_max, self.n_cycles
        )
        return {
            'ce': w_ce,
            'hier': w_hier,
            'ital': w_ital,
            'cl': w_cl,
        }


    def __call__(self, epoch: int) -> Tuple[float, float, float, float]:
        weights = self.get_weights(epoch)
        return weights['ce'], weights['hier'], weights['ital'], weights['cl']


def main():
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Visualize PeriodicLossScheduler weights.")
    parser.add_argument("--total_epochs", type=int, default=100)
    parser.add_argument("--n_cycles", type=int, default=3) 
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--hier_max", type=float, default=0.3)
    parser.add_argument("--ital_max", type=float, default=0.3)
    parser.add_argument("--cl_max", type=float, default=0.1)

    parser.add_argument("--out", type=str, default="loss_weights_cycle3.png")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    sched = PeriodicLossScheduler(
        total_epochs=args.total_epochs,
        ce_weight=args.ce_weight,
        hier_max=args.hier_max,
        ital_max=args.ital_max,
        cl_max=args.cl_max,
        n_cycles=args.n_cycles,
    )

    epochs = list(range(args.total_epochs))
    w_ce, w_hier, w_ital, w_cl = [], [], [], []

    for e in epochs:
        w = sched.get_weights(e)
        w_ce.append(w["ce"])
        w_hier.append(w["hier"])
        w_ital.append(w["ital"])
        w_cl.append(w["cl"])

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, w_ce, label="CE (periodic)")
    plt.plot(epochs, w_hier, label="HIER")
    plt.plot(epochs, w_ital, label="ITAL")
    plt.plot(epochs, w_cl, label="CL")
    plt.xlabel("Epoch")
    plt.ylabel("Weight")
    plt.title(
        f"Loss Weights (n_cycles={args.n_cycles}, total_epochs={args.total_epochs})"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)

    if args.show:
        plt.show()

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

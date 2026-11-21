"""Exchange image evidence within a scene class, keeping all fitted weights fixed."""
import argparse
import numpy as np
import pandas as pd
import torch
from course.common import configure, table, dump
from experiment3.train import BASE, SEEDS
from experiment3.semantic_features import OUT, CACHE
from experiment3.semantic_train import SemanticHead


@torch.inference_mode()
def main(device):
    configure(42)
    meta = pd.read_csv(BASE / "data/images.csv")
    test = meta[meta.split == "test"]
    donor = np.arange(len(meta))
    rng = np.random.default_rng(840)
    for _, group in test.groupby("category"):
        cycle = rng.permutation(group.index.to_numpy())
        donor[cycle] = np.roll(cycle, 1)
    assert (
        (donor[test.index] != test.index)
        & (meta.category.to_numpy()[donor[test.index]] == test.category.to_numpy())
    ).all()
    table(
        OUT / "counterfactual_pairs.csv",
        dict(
            index=test.index,
            member=test.member,
            donor=donor[test.index],
            donor_member=meta.member.to_numpy()[donor[test.index]],
        ),
    )
    actual = np.load(OUT / "tokens_local_routed.npy", mmap_mode="r")
    routed = np.load(OUT / "routed_prior.npy", mmap_mode="r")
    array = [
        np.load(p, mmap_mode="r")
        for p in (CACHE / "block6.npy", CACHE / "block12.npy", BASE / "data/stage2.npy")
    ]
    results = {k: np.copy(actual) for k in ("swap_tokens", "swap_local", "swap_route")}
    for k in results:
        results[k][test.index] = 0
    for seed in SEEDS:
        ck = torch.load(
            OUT / f"tokens_local_routed_{seed}/best.pt",
            weights_only=True,
            map_location=device,
        )
        model = SemanticHead("tokens_local_routed").eval().to(device)
        model.load_state_dict(ck["model"])
        for start in range(0, len(test), 32):
            idx = test.index.to_numpy()[start : start + 32]
            for name in results:
                token_ids = donor[idx] if name == "swap_tokens" else idx
                local_ids = donor[idx] if name == "swap_local" else idx
                route_ids = donor[idx] if name == "swap_route" else idx
                layers = [
                    torch.from_numpy(a[i]).to(device, dtype=torch.float32)
                    for a, i in zip(array, (token_ids, token_ids, local_ids))
                ]
                p = torch.from_numpy(routed[route_ids]).to(device)
                q = model(layers[0], layers[1], p, layers[2]).cpu().numpy()
                results[name][idx] += q / len(SEEDS)
    for name, x in results.items():
        np.save(OUT / f"{name}.npy", x)
    dump(
        OUT / "counterfactual_protocol.json",
        dict(
            variant="tokens_local_routed",
            seed=840,
            replaced=[
                "both DINO token maps",
                "ResNet stage-2 grid",
                "image-routed prior",
            ],
            donors="one cyclic derangement per official test category",
            models=SEEDS,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(p.parse_args().device)

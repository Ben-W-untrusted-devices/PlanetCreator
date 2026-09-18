import numpy as np

from planetcreator.bake import face_distances
from planetcreator.dataset import augment_layers


def _dists(water):
    return face_distances(water, 0, 1.0, 100.0)


def test_augmented_distances_match_recomputed():
    rng = np.random.default_rng(0)
    water = rng.random((24, 24)) < 0.15
    layers = {"water": water, **_dists(water)}
    layers.update({f"ctx_{k}": v for k, v in _dists(water).items()})
    for k in range(4):
        for flip in (False, True):
            aug = augment_layers(layers, k, flip)
            fresh = _dists(aug["water"])
            for name, expected in fresh.items():
                assert np.allclose(aug[name], expected), (k, flip, name)
                assert np.allclose(aug["ctx_" + name], expected), (k, flip, "ctx_" + name)

from dataclasses import dataclass


@dataclass
class LayerWeight:
    idx: int
    weight: float


HYPERPARAMS = {
    "alpha": 10,  # content_cost_weight
    "beta": 40,  # style_cost_weight
    "content_cost_layer": -1,  # index of the layer to use for content cost
    "style_cost_layers": [LayerWeight(idx=1, weight=1)],
}

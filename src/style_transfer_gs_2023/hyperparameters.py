from dataclasses import dataclass


@dataclass
class LayerAndWeight:
    name: str
    weight: float


HYPERPARAMS = {
    "alpha": 10,  # content_cost_weight
    "beta": 40,  # style_cost_weight
    "content_cost_layer": [LayerAndWeight("block5_conv4", 1)],
    "style_cost_layers": [
        LayerAndWeight("block1_conv1", 0.2),
        LayerAndWeight("block2_conv1", 0.2),
        LayerAndWeight("block3_conv1", 0.2),
        LayerAndWeight("block4_conv1", 0.2),
        LayerAndWeight("block5_conv1", 0.2),
    ],
    "initial_noise": 0.25,  # fraction of noise to add between 0 and 1
}

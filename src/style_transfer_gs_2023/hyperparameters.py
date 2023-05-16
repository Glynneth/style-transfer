from dataclasses import dataclass


@dataclass
class LayerAndWeight:
    name: str
    weight: float


HYPERPARAMS = {
    "alpha": 1e4,  # content_cost_weight
    "beta": 1e-2,  # style_cost_weight
    "total_variation_weight": 30, # reduces high frequency components in generation
    "content_cost_layer": [LayerAndWeight("block5_conv2", 1)],
    "style_cost_layers": [
        LayerAndWeight("block1_conv1", 0.2),
        LayerAndWeight("block2_conv1", 0.2),
        LayerAndWeight("block3_conv1", 0.2),
        LayerAndWeight("block4_conv1", 0.2),
        LayerAndWeight("block5_conv1", 0.2),
    ],
    "initial_noise": 0.25,  # fraction of noise to add between 0 and 1
    "epochs": 10,
    "steps_per_epoch": 30,
    "learning_rate": 0.02
}

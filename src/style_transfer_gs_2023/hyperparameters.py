from dataclasses import dataclass


@dataclass
class LayerAndWeight:
    name: str
    weight: float


HYPERPARAMS = {
    "alpha": 1000,  # content_cost_weight
    "beta": 40,  # style_cost_weight
    "content_cost_layer": [LayerAndWeight("block5_conv2", 1)],
    "style_cost_layers": [
        LayerAndWeight("block1_conv1", 0.35),
        LayerAndWeight("block2_conv1", 0.3),
        LayerAndWeight("block3_conv1", 0.2),
        LayerAndWeight("block4_conv1", 0.1),
        LayerAndWeight("block5_conv1", 0.05),
    ],
    "initial_noise": 0.25,  # fraction of noise to add between 0 and 1
    "epochs": 2000,
    "num_output_imgs": 10,
    "learning_rate": 0.02
}

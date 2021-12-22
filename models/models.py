import torch
import torch.nn as nn

import datasets.houses_dataset
import options.linear_regression_options
from options.classification_options import ClassificationOptions


class Print(nn.Module):
    """"
    This model is for debugging purposes (place it in nn.Sequential to see tensor dimensions).
    """

    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(x.shape)
        return x


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        """START TODO: replace None with a Linear layer"""
        self.linear_layer = nn.Linear(1, 1)
        """END TODO"""

    def forward(self, x: torch.Tensor):
        """START TODO: forward the tensor x through the linear layer and return the outcome (replace None)"""
        x = self.linear_layer(x)
        """END TODO"""
        return x

class Classifier(nn.Module):
    def __init__(self, options: ClassificationOptions):
        super().__init__()
        """ START TODO: fill in all three layers. 
            Remember that each layer should contain 2 parts, a linear layer and a nonlinear activation function.
            Use options.hidden_sizes to store all hidden sizes, (for simplicity, you might want to 
            include the input and output as well).
        """

        """linear layer + ReLu function"""
        self.layer1 = nn.Sequential(
            nn.Linear(options.hidden_sizes[0], options.hidden_sizes[1]),
            nn.ReLU()
        )

        """linear layer + ReLu function"""
        self.layer2 = nn.Sequential(
            nn.Linear(options.hidden_sizes[1], options.hidden_sizes[2]),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
             nn.Linear(options.hidden_sizes[2], options.hidden_sizes[3]),
             nn.ReLU()
         )

        """linear layer + Softmax function
        the softmax function, only for this layer a classification itself (last layer)
        """
        """Print(),"""
        self.layer4 = nn.Sequential(
            nn.Linear(options.hidden_sizes[3], options.hidden_sizes[4])
        )
        """END TODO"""

    def forward(self, x: torch.Tensor):
        """START TODO: forward tensor x through all layers."""
        output_l1 = self.layer1(x)
        output_l2 = self.layer2(output_l1)
        output_l3 = self.layer3(output_l2)
        output_l4 = self.layer4(output_l3)
        """END TODO"""
        return output_l4


class ClassifierVariableLayers(nn.Module):
    def __init__(self, options: ClassificationOptions):
        super().__init__()
        self.layers = nn.Sequential()
        for i in range(len(options.hidden_sizes) - 1):
            self.layers.add_module(
                f"lin_layer_{i + 1}",
                nn.Linear(options.hidden_sizes[i], options.hidden_sizes[i + 1])
            )
            if i < len(options.hidden_sizes) - 2:
                self.layers.add_module(
                    f"relu_layer_{i + 1}",
                    nn.ReLU()
                )
            else:
                self.layers.add_module(
                    f"softmax_layer",
                    nn.Softmax(dim=1)
                )
        print(self)

    def forward(self, x: torch.Tensor):
        x = self.layers(x)
        return x
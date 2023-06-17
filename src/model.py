"""Policy model.

Convolutional neural network policy model.
The models represent the agent's policy and map states
to actions.

"""
import torch
import torch.nn as nn

from src.utils import eval


class ConvBlock(nn.Module):
    """Convolutional block.

    Attributes:
        conv_block:
    """

    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            hidden_channels: int = None,
            dropout_rate: float = 0.0
        ) -> None:
        super().__init__()

        hidden_channels = hidden_channels or out_channels

        conv_config = {"kernel_size": 3, "stride": 1, "padding": "same"}

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=hidden_channels, 
                **conv_config
            ),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
            nn.Dropout(p=dropout_rate)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (1, field_size, field_size).

        Returns:
            Tensor of same shape as input.
        """
        x = self.conv_block(x)
        return x


class Model(nn.Module):
    """Simple convolutional neural network"""

    num_actions = 4

    def __init__(self, args) -> None:
        """Initializes a Model instance."""
        super().__init__()

        field_size = args.field_size
        num_layers = args.num_layers
        num_channels = args.num_channels
        dropout_rate = args.dropout_rate

        is_policy_gradient = args.algorithm == "policy_gradient"

        blocks = [
            ConvBlock(
                in_channels=num_channels, 
                out_channels=num_channels,
                dropout_rate=dropout_rate
            )
            for _ in range(num_layers)
        ]

        self.conv_block = nn.Sequential(
            nn.GroupNorm(1, 1),
            ConvBlock(in_channels=1, out_channels=num_channels),
            *blocks
        )

        in_features_dense = num_channels * field_size * field_size

        self.dense_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_features_dense, out_features=self.num_actions),
            nn.Softmax(dim=-1) if is_policy_gradient else nn.Identity(),
        )

    @eval
    @torch.no_grad()
    def predict(self, state: torch.Tensor) -> int:
        """Predicts action for given state.

        Args:
            state: Tensor representing state of playing field.

        Returns:
            The action represented by an integer.
        """
        prediction = self(state)
        action = torch.argmax(prediction, dim=-1).item()
        return action

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor.
        """
        x = torch.unsqueeze(input=x, dim=1)  # Add channel dimension.
        x = self.conv_block(x)
        x = self.dense_block(x)
        return x
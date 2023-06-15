"""Policy model.

Convolutional neural network with residual connections.
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
            hidden_channels: int = None
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
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (1, field_size, field_size).

        Returns:
            Tensor of same shape as input.
        """
        x = x + self.conv_block(x)
        return x


class Model(nn.Module):
    """Simple convolutional neural network"""

    num_actions = 4

    def __init__(self, args) -> None:
        """Initializes a Model instance."""
        super().__init__()

        field_size = args.field_size
        # num_layers = args.num_layers
        is_policy_gradient = args.algorithm == "policy_gradient"
        hidden_channels = 32

        self.conv_block = nn.Sequential(
            nn.GroupNorm(1, 1),
            ConvBlock(in_channels=1, out_channels=hidden_channels),
            ConvBlock(in_channels=hidden_channels, out_channels=hidden_channels),
            ConvBlock(in_channels=hidden_channels, out_channels=hidden_channels),
            ConvBlock(in_channels=hidden_channels, out_channels=hidden_channels),
            # nn.LayerNorm((1, field_size, field_size)),
            # nn.Conv2d(in_channels=1, out_channels=hidden_channels, kernel_size=3, stride=1, padding="same"),
            # nn.GELU(),
            # nn.LayerNorm((hidden_channels, field_size, field_size)),
            # nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding="same"),
            # nn.GELU(),
            # nn.LayerNorm((hidden_channels, field_size, field_size)),
            # nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding="same"),
            # nn.GELU(),
            # nn.LayerNorm((hidden_channels, field_size, field_size)),
        )

        in_features_dense = hidden_channels * field_size * field_size

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
        x = torch.unsqueeze(input=x, dim=1)  # Add dimension to represent the channel.
        x = self.conv_block(x)
        x = self.dense_block(x)
        return x
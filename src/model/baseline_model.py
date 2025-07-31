import torch
from torch import nn
from torch.nn import Sequential


class MFM(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return torch.max(x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:])


class PrintShape(nn.Module):
    def forward(self, x):
        print(f"Shape: {x.shape}")
        return x


class BaselineModel(nn.Module):
    """
    Simple MLP
    """

    def __init__(self):
        """
        Args:
            n_feats (int): number of input features.
            n_class (int): number of classes.
            fc_hidden (int): number of hidden features.
        """
        super().__init__()
        self.ConvPart = Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1),     # Layer 1
            MFM(),                                                                             # Layer 2
            nn.MaxPool2d(kernel_size=2, stride=2),                                             # Layer 3
            nn.Dropout2d(p=0.5), 
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1),               # Layer 4
            MFM(),                                                                             # Layer 5
            nn.BatchNorm2d(32),                                                                # Layer 6
            nn.Dropout2d(p=0.5),
            nn.Conv2d(in_channels=32, out_channels=96, kernel_size=3, stride=1, padding=1),               # Layer 7
            MFM(),                                                                             # Layer 8

            nn.MaxPool2d(kernel_size=2, stride=2),                                             # Layer 9
            nn.BatchNorm2d(48),                                                                # Layer 10
            nn.Dropout2d(p=0.5),
            nn.Conv2d(in_channels=48, out_channels=96, kernel_size=1, stride=1),               # Layer 11
            MFM(),                                                                             # Layer 12
            nn.BatchNorm2d(48),                                                                # Layer 13
            nn.Conv2d(in_channels=48, out_channels=128, kernel_size=3, stride=1, padding=1),              # Layer 14
            MFM(),                                                                             # Layer 15
            nn.MaxPool2d(kernel_size=2, stride=2),                                             # Layer 16
            nn.Dropout2d(p=0.5), 
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1),              # Layer 17
            MFM(),                                                                             # Layer 18
            nn.BatchNorm2d(64),                                                                # Layer 19
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),               # Layer 20
            MFM(),                                                                             # Layer 21
            nn.BatchNorm2d(32),                                                                # Layer 22
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1),               # Layer 23
            MFM(),                                                                             # Layer 24
            nn.BatchNorm2d(32),                                                                # Layer 25
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),               # Layer 26
            MFM(),                                                                             # Layer 27
            nn.MaxPool2d(kernel_size=2, stride=2),                                             # Layer 28
            nn.Dropout2d(p=0.5),
        )

        self.LinearPart = Sequential(
            nn.Linear(53 * 37 * 32, 160),
            nn.Dropout(p=0.5),
            MFM(),
            nn.BatchNorm1d(80),
            nn.Linear(80, 2)
        )

    def forward(self, data_object, **batch):
        """
        Model forward method.

        Args:
            data_object (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """
        data_object = data_object.squeeze(1)
        data_object = self.ConvPart(data_object)
        data_object = data_object.view(data_object.size(0), -1)
        data_object = self.LinearPart(data_object)
        return {"logits": data_object}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info

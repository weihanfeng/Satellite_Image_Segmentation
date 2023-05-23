import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torchvision


class DoubleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, feature_nums=[64, 128, 256, 512]):
        super().__init__()
        self.down_conv = nn.ModuleList()
        self.up_conv = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down convolution
        for feature_num in feature_nums:
            self.down_conv.append(DoubleConvolution(in_channels, feature_num))
            in_channels = feature_num

        # Up convolution
        for feature_num in reversed(feature_nums):
            self.up_conv.append(
                nn.ConvTranspose2d(
                    feature_num * 2, feature_num, kernel_size=2, stride=2
                )
            )
            self.up_conv.append(DoubleConvolution(feature_num * 2, feature_num))

        # Bottom convolution
        self.bottom_conv = DoubleConvolution(feature_nums[-1], feature_nums[-1] * 2)

        # Output convolution
        self.output_conv = nn.Conv2d(feature_nums[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Down convolution and save skip connections to be used in up convolution
        skip_connections = []
        for down_conv in self.down_conv:
            x = down_conv(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottom_conv(x)
        skip_connections = skip_connections[::-1]  # reverse the list

        # Up convolution
        for idx in range(0, len(self.up_conv), 2):
            x = self.up_conv[idx](x)
            skip_connection = skip_connections[
                idx // 2
            ]  # because we have 2 layers in up convolution for each skip connection
            if x.shape != skip_connection.shape:
                # bilinear interpolation to match the shape of skip connection
                # due to maxpooling, the shape of x may be smaller than skip connection due to floor division
                # e.g. input shape (also skip connection) = [161, 161], output shape = [80, 80], but up_conv shape = [160, 160]
                x = nn.functional.interpolate(
                    x,
                    size=skip_connection.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                )
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.up_conv[idx + 1](concat_skip)

        return self.output_conv(x)


class ImageSegmentationModel(nn.Module):
    def __init__(
        self,
        model = UNet,
        in_channels=3,
        out_channels=5,
        num_classes=5,
        feature_nums=[64, 128, 256, 512],
        learning_rate=0.001,
        optimizer=torch.optim.Adam,
        loss_fn=nn.CrossEntropyLoss(),
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.num_classes = num_classes
        self.optimizer = optimizer(self.model.parameters(), lr=self.learning_rate)
        self.scaler = torch.cuda.amp.GradScaler()

    def forward(self, x):
        return self.model(x)

    def _get_prediction_result(self, pred, target):
        """Get prediction result
        Args:
            pred (torch.Tensor): prediction
            target (torch.Tensor): target
        Returns:
            loss (torch.Tensor): loss
            iou (torch.Tensor): IoU
            pred_labels (torch.Tensor): predicted labels
        """
        loss = self.loss_fn(pred, target)
        pred_labels = torch.argmax(pred, dim=1)
        pred_labels = F.one_hot(pred_labels, num_classes=self.num_classes).permute(0, 3, 1, 2)
        iou = self._compute_iou(pred_labels, target)

        return loss, iou, pred_labels

    def _compute_iou(self, pred, target):
        """Compute IoU
        Args:
            pred (torch.Tensor): prediction
            target (torch.Tensor): target
        Returns:
            iou (torch.Tensor): IoU
        """
        intersection = torch.logical_and(pred, target)
        union = torch.logical_or(pred, target)
        iou = torch.sum(intersection) / torch.sum(union)

        return iou
    
    def train_val_epoch(self, loader, mode):
        """A training and validation epoch
        Args:
            loader (DataLoader): DataLoader
            mode (str): train or validation
        Returns:
            train_loss (float): training loss
            train_iou (float): training IoU
        """
        self.model.train() if mode == "train" else self.model.eval()

        progress_bar = tqdm(loader, desc=mode)
        total_loss = 0.0
        total_iou = 0.0
        for batch_idx, (data, targets) in enumerate(progress_bar):
            data = data.to(device=self.device)
            targets = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2)
            targets = targets.float().to(device=self.device)

            # forward
            with torch.cuda.amp.autocast(): # cast tensors to a smaller memory footprint
                predictions = self.model(data)
                loss, iou, _ = self._get_prediction_result(predictions, targets)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward() # scale loss to prevent underflow when using low precision floats
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # get result for current batch
            total_iou += iou.item()
            total_loss += loss.item()
            ongoing_iou = total_iou/(batch_idx+1)
            ongoing_loss = total_loss/(batch_idx+1)

            # update tqdm progress bar
            progress_bar.set_postfix({"loss": ongoing_loss, "IoU": ongoing_iou})
            progress_bar.update()

        epoch_loss = total_loss/len(loader)
        epoch_iou = total_iou/len(loader)

        return epoch_loss, epoch_iou


class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """

        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class UNetWithResnet50Encoder(nn.Module):
    DEPTH = 6

    def __init__(self, n_classes=5):
        super().__init__()
        resnet = torchvision.models.resnet.resnet50(pretrained=True)
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)
        up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 3, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)

    def forward(self, x, with_output_feature_map=False):
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (UNetWithResnet50Encoder.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{UNetWithResnet50Encoder.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])
        output_feature_map = x
        x = self.out(x)
        del pre_pools
        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x
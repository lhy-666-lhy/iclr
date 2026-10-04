import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy
from pathlib import Path

from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint
import pdb

def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules

class PointNetEncoderXYZRGB(nn.Module):
    """Encoder for Pointcloud"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1024,
        use_layernorm: bool = False,
        final_norm: str = "none",
        use_projection: bool = True,
        **kwargs,
    ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256, 512]
        cprint("pointnet use_layernorm: {}".format(use_layernorm), "cyan")
        cprint("pointnet use_final_norm: {}".format(final_norm), "cyan")

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )

        if final_norm == "layernorm":
            self.final_projection = nn.Sequential(nn.Linear(block_channel[-1], out_channels),
                                                  nn.LayerNorm(out_channels))
        elif final_norm == "none":
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x

class RobotTorpologyEncoder(nn.Module):
    """Encoder for topology point cloud [x, y, z, self_id, neighbor_id].

    Before MLP, scales id channels: (x,y,z,a,b) → (x,y,z,0.1*a,0.1*b).
    """

    # Scale factor for link-id channels (dims 3, 4). xyz unchanged.
    ID_SCALE = 0.1

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 1024,
        use_layernorm: bool = False,
        final_norm: str = "none",
        use_projection: bool = True,
        id_scale: float = 0.1,
        **kwargs,
    ):
        """_summary_

        Args:
            in_channels (int): feature size of input (must be 5)
            id_scale (float): multiply self_id / neighbor_id by this before MLP
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256]
        self.id_scale = float(id_scale)
        cprint("[RobotTorpologyEncoder] use_layernorm: {}".format(use_layernorm), "cyan")
        cprint("[RobotTorpologyEncoder] use_final_norm: {}".format(final_norm), "cyan")
        cprint("[RobotTorpologyEncoder] id_scale (dims 3,4): {}".format(self.id_scale), "cyan")

        assert in_channels == 5, cprint(
            f"RobotTorpologyEncoder only supports 5 channels, but got {in_channels}", "red"
        )

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )

        if final_norm == "layernorm":
            self.final_projection = nn.Sequential(nn.Linear(block_channel[-1], out_channels),
                                                  nn.LayerNorm(out_channels))
        elif final_norm == "none":
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[RobotTorpologyEncoder] not use projection", "yellow")

        VIS_WITH_GRAD_CAM = False
        if VIS_WITH_GRAD_CAM:
            self.gradient = None
            self.feature = None
            self.input_pointcloud = None
            self.mlp[0].register_forward_hook(self.save_input)
            self.mlp[6].register_forward_hook(self.save_feature)
            self.mlp[6].register_backward_hook(self.save_gradient)

    def forward(self, x):
        # x: (B, N, 5) = (x, y, z, self_id, neighbor_id)
        # → (x, y, z, id_scale*a, id_scale*b); upstream of this encoder needs no grad.
        x = torch.cat([x[..., :3], x[..., 3:5] * self.id_scale], dim=-1)
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x

    def save_gradient(self, module, grad_input, grad_output):
        """
        for grad-cam
        """
        self.gradient = grad_output[0]

    def save_feature(self, module, input, output):
        """
        for grad-cam
        """
        if isinstance(output, tuple):
            self.feature = output[0].detach()
        else:
            self.feature = output.detach()

    def save_input(self, module, input, output):
        """
        for grad-cam
        """
        self.input_pointcloud = input[0].detach()

class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud"""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1024,
        use_layernorm: bool = False,
        final_norm: str = "none",
        use_projection: bool = True,
        **kwargs,
    ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256]
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), "cyan")
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), "cyan")

        assert in_channels == 3, cprint(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )

        if final_norm == "layernorm":
            self.final_projection = nn.Sequential(nn.Linear(block_channel[-1], out_channels),
                                                  nn.LayerNorm(out_channels))
        elif final_norm == "none":
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")

        VIS_WITH_GRAD_CAM = False
        if VIS_WITH_GRAD_CAM:
            self.gradient = None
            self.feature = None
            self.input_pointcloud = None
            self.mlp[0].register_forward_hook(self.save_input)
            self.mlp[6].register_forward_hook(self.save_feature)
            self.mlp[6].register_backward_hook(self.save_gradient)

    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x

    def save_gradient(self, module, grad_input, grad_output):
        """
        for grad-cam
        """
        self.gradient = grad_output[0]

    def save_feature(self, module, input, output):
        """
        for grad-cam
        """
        if isinstance(output, tuple):
            self.feature = output[0].detach()
        else:
            self.feature = output.detach()

    def save_input(self, module, input, output):
        """
        for grad-cam
        """
        self.input_pointcloud = input[0].detach()

class RobotGeometryEncoder(nn.Module):
    """Encoder for Pointcloud"""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1024,
        use_layernorm: bool = False,
        final_norm: str = "none",
        use_projection: bool = True,
        **kwargs,
    ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256]
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), "cyan")
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), "cyan")

        assert in_channels == 3, cprint(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )

        if final_norm == "layernorm":
            self.final_projection = nn.Sequential(nn.Linear(block_channel[-1], out_channels),
                                                  nn.LayerNorm(out_channels))
        elif final_norm == "none":
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")

        VIS_WITH_GRAD_CAM = False
        if VIS_WITH_GRAD_CAM:
            self.gradient = None
            self.feature = None
            self.input_pointcloud = None
            self.mlp[0].register_forward_hook(self.save_input)
            self.mlp[6].register_forward_hook(self.save_feature)
            self.mlp[6].register_backward_hook(self.save_gradient)

    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x

    def save_gradient(self, module, grad_input, grad_output):
        """
        for grad-cam
        """
        self.gradient = grad_output[0]

    def save_feature(self, module, input, output):
        """
        for grad-cam
        """
        if isinstance(output, tuple):
            self.feature = output[0].detach()
        else:
            self.feature = output.detach()

    def save_input(self, module, input, output):
        """
        for grad-cam
        """
        self.input_pointcloud = input[0].detach()

def _third_party_root() -> Path:
    # vision → model → backbone → spacepolicy → SPACE → policy → repo root
    return Path(__file__).resolve().parents[6] / "third_party"

class ScaledIdTopologyWrapper(nn.Module):
    """Apply topology id_scale then forward backbone (for PointNet++ / PT)."""

    def __init__(self, backbone: nn.Module, id_scale: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.id_scale = float(id_scale)

    def forward(self, x):
        # x: [B,N,5] → scale id channels like RobotTorpologyEncoder
        x = torch.cat([x[..., :3], x[..., 3:5] * self.id_scale], dim=-1)
        return self.backbone(x)

def build_robot_geometry_encoder(
    backbone: str = "pointnet",
    in_channels: int = 3,
    out_channels: int = 128,
    use_layernorm: bool = True,
    final_norm: str = "layernorm",
    use_projection: bool = True,
    **kwargs,
) -> nn.Module:
    """Ablation-4: robot geometry PC encoder only (scene path untouched)."""
    import sys

    bb = str(backbone).lower().strip()
    if bb in ("pointnet", "pn", "simple", "simple_pointnet"):
        return RobotGeometryEncoder(
            in_channels=in_channels,
            out_channels=out_channels,
            use_layernorm=use_layernorm,
            final_norm=final_norm,
            use_projection=use_projection,
            **kwargs,
        )
    tp = _third_party_root()
    if str(tp) not in sys.path:
        sys.path.insert(0, str(tp))
    from pc_backbone import build_robot_pc_encoder  # type: ignore

    return build_robot_pc_encoder(
        backbone=bb,
        in_channels=in_channels,
        out_channels=out_channels,
        use_layernorm=use_layernorm,
        final_norm=final_norm,
        use_projection=use_projection,
        pointnet_cls=RobotGeometryEncoder,
        **kwargs,
    )

def build_robot_topology_encoder(
    backbone: str = "pointnet",
    in_channels: int = 5,
    out_channels: int = 128,
    use_layernorm: bool = True,
    final_norm: str = "layernorm",
    use_projection: bool = True,
    id_scale: float = 0.1,
    **kwargs,
) -> nn.Module:
    """Ablation-4: robot topology PC encoder only (scene path untouched)."""
    import sys

    bb = str(backbone).lower().strip()
    if bb in ("pointnet", "pn", "simple", "simple_pointnet"):
        return RobotTorpologyEncoder(
            in_channels=in_channels,
            out_channels=out_channels,
            use_layernorm=use_layernorm,
            final_norm=final_norm,
            use_projection=use_projection,
            id_scale=id_scale,
            **kwargs,
        )
    tp = _third_party_root()
    if str(tp) not in sys.path:
        sys.path.insert(0, str(tp))
    from pc_backbone import build_robot_pc_encoder  # type: ignore

    core = build_robot_pc_encoder(
        backbone=bb,
        in_channels=in_channels,
        out_channels=out_channels,
        use_layernorm=use_layernorm,
        final_norm=final_norm,
        use_projection=use_projection,
        pointnet_cls=None,
        **kwargs,
    )
    return ScaledIdTopologyWrapper(core, id_scale=id_scale)

class SPACEEncoder(nn.Module):

    def __init__(
        self,
        observation_space: Dict,
        img_crop_shape=None,
        out_channel=256,
        state_mlp_size=(64, 64),
        state_mlp_activation_fn=nn.ReLU,
        pointcloud_encoder_cfg=None,
        use_pc_color=False,
        pointnet_type="pointnet",
    ):
        super().__init__()
        self.imagination_key = "imagin_robot"
        self.state_key = "agent_pos"
        self.point_cloud_key = "point_cloud"
        self.rgb_image_key = "image"
        self.n_output_channels = out_channel

        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_shape = observation_space[self.state_key]
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None

        cprint(f"[SPACEEncoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[SPACEEncoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[SPACEEncoder] imagination point shape: {self.imagination_shape}", "yellow")

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        # Official / default scene PC backbone: Simple PointNet only.
        if pointnet_type != "pointnet":
            cprint(
                f"[SPACEEncoder] WARNING: scene pointnet_type={pointnet_type!r} "
                f"forced to official simple 'pointnet' (robot ablation uses geo/topo only).",
                "yellow",
            )
            pointnet_type = "pointnet"
            self.pointnet_type = pointnet_type
        if use_pc_color:
            pointcloud_encoder_cfg.in_channels = 6
            self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
        else:
            pointcloud_encoder_cfg.in_channels = 3
            self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
        cprint("[SPACEEncoder] scene pc backbone: simple pointnet (official)", "cyan")

        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels += output_dim
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))

        cprint(f"[SPACEEncoder] output dim: {self.n_output_channels}", "red")

    def forward(self, observations: Dict) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
        if self.use_imagined_robot:
            img_points = observations[self.imagination_key][..., :points.shape[-1]]  # align the last dim
            points = torch.concat([points, img_points], dim=1)

        # points = torch.transpose(points, 1, 2)   # B * 3 * N
        # points: B * 3 * (N + sum(Ni))
        pn_feat = self.extractor(points)  # B * out_channel

        state = observations[self.state_key]
        state_feat = self.state_mlp(state)  # B * 64
        final_feat = torch.cat([pn_feat, state_feat], dim=-1)
        return final_feat

    def output_shape(self):
        return self.n_output_channels


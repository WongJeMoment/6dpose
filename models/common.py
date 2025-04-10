import torch
from torch import nn, einsum
from torch.autograd import Variable
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn.functional as F


class ConvLSTM1D(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.Gates = nn.Conv1d(in_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=kernel_size // 2,)
        self.prev_state = None
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.Gates.weight)

    def reset(self):
        self.prev_state = None

    def forward(self, input_):
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if self.prev_state is None:
            state_size = [batch_size, self.hidden_channels] + list(spatial_size)
            self.prev_state = (
                Variable(torch.zeros(state_size, device=input_.device)),
                Variable(torch.zeros(state_size, device=input_.device)),
            )

        prev_hidden, prev_cell = self.prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        self.prev_state = (hidden, cell)
        return hidden


class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    From: https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py
    """

    def __init__(self, in_channels, hidden_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.Gates = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=kernel_size // 2,
        )
        self.prev_state = None
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.Gates.weight)

    def reset(self):
        self.prev_state = None

    def forward(self, input_):
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if self.prev_state is None:
            state_size = [batch_size, self.hidden_channels] + list(spatial_size)
            self.prev_state = (
                Variable(torch.zeros(state_size, device=input_.device)),
                Variable(torch.zeros(state_size, device=input_.device)),
            )

        prev_hidden, prev_cell = self.prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        self.prev_state = (hidden, cell)
        return hidden


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_convs=3,
        kernel_size=3,
        stride=1,
        padding=1,
        downsample=True,
        dilation=1,
    ):
        super(ConvBlock, self).__init__()
        self.modules = []

        c_in = in_channels
        c_out = out_channels
        for i in range(n_convs):
            self.modules.append(
                nn.Conv2d(
                    in_channels=c_in,
                    out_channels=c_out,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    dilation=dilation,
                )
            )
            self.modules.append(nn.BatchNorm2d(num_features=out_channels))
            self.modules.append(nn.LeakyReLU(0.1))
            c_in = c_out

        if downsample:
            self.modules.append(
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.modules.append(nn.ReLU())
            # self.modules.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.model = nn.Sequential(*self.modules)
        self.init_weights()

    def init_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.model(x)

# 用于构建可插拔的神经网络层
# 通用的“卷积模块积木”，方便快速搭建神经网络，同时支持 2D/3D、普通/反卷积、是否归一化、是否激活
class BasicConv_IN(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, IN=True, relu=True, **kwargs):
        super(BasicConv_IN, self).__init__()

        self.relu = relu
        self.use_in = IN
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_in:
            x = self.IN(x)
        if self.relu:
            x = nn.LeakyReLU()(x)#, inplace=True)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.dropout = nn.Dropout(dropout)

        self.reattn_weights = nn.Parameter(torch.randn(heads, heads))

        self.reattn_norm = nn.Sequential(
            Rearrange('b h i j -> b i j h'),
            nn.LayerNorm(heads),
            Rearrange('b i j h -> b h i j')
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, pad_mask=None):
        b, n, _, h = *x.shape, self.heads
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # attention

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        if pad_mask is not None:
            # print('dots:', dots.shape)
            # print('pad_mask:', pad_mask.shape)
            dots += pad_mask
        attn = dots.softmax(dim=-1)
        # print('attn:', attn[0].shape)
        # print('attn:', attn[1][0])
        # print('attn:', attn[1][1])
        # input()
        attn = self.dropout(attn)

        # re-attention

        attn = einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)
        attn = self.reattn_norm(attn)

        # aggregate and out

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x, pad_mask=None):
        for attn, ff in self.layers:
            x = attn(x, pad_mask) + x
            x = ff(x) + x
        return x


def extract_glimpse(input, size, offsets, centered=False, normalized=False, mode="nearest", padding_mode="zeros"):
    W, H = input.size(-1), input.size(-2)

    if normalized and centered:
        offsets = (offsets + 1) * offsets.new_tensor([W / 2, H / 2])
    elif normalized:
        offsets = offsets * offsets.new_tensor([W, H])
    elif centered:
        raise ValueError("Invalid parameter that offsets centered but not normlized")

    h, w = size
    xs = torch.arange(0, w, dtype=input.dtype, device=input.device) - (w - 1) / 2.0
    ys = torch.arange(0, h, dtype=input.dtype, device=input.device) - (h - 1) / 2.0

    # vy, vx = torch.meshgrid(ys, xs, indexing="ij")
    vy, vx = torch.meshgrid(ys, xs)
    grid = torch.stack([vx, vy], dim=-1)
    offsets_grid = offsets[:, None, None, :] + grid[None, ...]
    offsets_grid = (offsets_grid - offsets_grid.new_tensor([W / 2, H / 2])) / offsets_grid.new_tensor([W / 2, H / 2])

    return torch.nn.functional.grid_sample(input, offsets_grid.float(), mode=mode, align_corners=True, padding_mode=padding_mode)


def disparity_regression(x, maxdisp):
    assert len(x.shape) == 2
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp)
    return torch.sum(x * disp_values, 1, keepdim=True)


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2).mean(-1).mean(-1)
    return cost

# refimg_fea: 参考图像的特征（通常是左图）。
# targetimg_fea: 目标图像的特征（通常是右图）。
# maxdisp: 最大视差范围（原始分辨率下）。
# num_groups: 分组数量，用于分组相关性计算。
# patch_pos: 当前参考图的 patch 中心位置坐标。
# ds_rate: 特征图的下采样率。
def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups, patch_pos, ds_rate):
    BN, C, P, _ = refimg_fea.shape    # 24 * C * 31 * 31
    B, _, H, W = targetimg_fea.shape  # 4 * C * 31 * 31
    maxdisp_ds = maxdisp // ds_rate   # 下采样后的最大视差

    # 空代价体，准备每个视差位置填上相似度。
    volume = refimg_fea.new_zeros([BN, num_groups, maxdisp_ds])  # 24 * 8 * 160//4
    # 距离边界检查 + 初始化
    dist_to_boarder = patch_pos[:, 0].max()   # patch 最左边距离图像边界的最大值
    off = torch.zeros_like(patch_pos)         # 每次视差迭代都会左移，off 记录累计偏移
    # target_copy是把右图特征扩展成与refimg_fea一样数量的tensor，用于提取相应位置的patch
    target_copy = targetimg_fea.unsqueeze(1).repeat(1, BN // B, 1, 1, 1).reshape(-1, C, H, W)
    # 遍历视差，构建cost volume
    for i in range(maxdisp_ds):
        # 如果当前视差已经让 patch 移出图像边界，那就不用继续了。
        if i > dist_to_boarder:
            break
        if i > 0:
            # 横向往左移动（因为 stereo 匹配假设视差是水平方向的）；
            off[:, 0] -= 1. * ds_rate
            # 从右图上抽取与左图 patch 对齐的区域（用 patch_pos + 偏移）；
            target_patch = extract_glimpse(target_copy, (P, P), patch_pos + off + 0.5)
            # 左右特征之间按组点乘计算相似度，填入当前视差层。
            volume[:, :, i] = groupwise_correlation(refimg_fea, target_patch, num_groups)
        else:
            target_patch = extract_glimpse(target_copy, (P, P), patch_pos + off + 0.5)
            volume[:, :, i] = groupwise_correlation(refimg_fea, target_patch, num_groups)

    volume = volume.contiguous()
    return volume


def context_upsample(disp_low, up_weights):
    b, c, h, w = disp_low.shape

    disp_unfold = F.unfold(disp_low.reshape(b, c, h, w), 3, 1, 1).reshape(b, -1, h, w)
    disp_unfold = F.interpolate(disp_unfold, (h * 4, w * 4), mode='nearest').reshape(b, 9, h * 4, w * 4)

    disp = (disp_unfold * up_weights).sum(1)

    return disp
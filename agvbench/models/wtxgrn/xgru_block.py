import math
import einops
import torch
import torch.nn.functional as F
from torch import nn
from .wtxgrn_util import DropPath, SequenceConv2d, WTSequenceConv2d
from .multi_scan import MultiScan


def bias_linspace_init_(param: torch.Tensor, start: float = 3.4, end: float = 6.0) -> torch.Tensor:
    """Linearly spaced bias init across dimensions."""
    assert param.dim() == 1, f"param must be 1-dimensional (typically a bias), got {param.dim()}"
    n_dims = param.shape[0]
    init_vals = torch.linspace(start, end, n_dims)
    with torch.no_grad():
        param.copy_(init_vals)
    return param


def small_init_(param: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Fills the input Tensor with values according to the method described in Transformers without Tears: Improving
    the Normalization of Self-Attention - Nguyen, T. & Salazar, J. (2019), using a normal distribution.
    Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py.
    """
    std = math.sqrt(2 / (5 * dim))
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param


def wang_init_(param: torch.Tensor, dim: int, num_blocks: int):
    """ Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py. """
    std = 2 / num_blocks / math.sqrt(dim)
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param


class LinearHeadwiseExpand(nn.Module):
    """
    This is a structured projection layer that projects the input to a higher dimension.
    It only allows integer up-projection factors, i.e. the output dimension is a multiple of the input dimension.
    """

    def __init__(self, dim, head_dim, bias=False):
        super().__init__()
        assert dim % head_dim == 0
        self.dim = dim
        self.head_dim = head_dim

        dim_per_head = dim // head_dim
        self.weight = nn.Parameter(torch.empty(head_dim, dim_per_head, dim_per_head))
        if bias:
            self.bias = nn.Parameter(torch.empty(dim))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight.data, mean=0.0, std=math.sqrt(2 / 5 / self.weight.shape[-1]))
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(x, "... (nh d) -> ... nh d", nh=self.head_dim)
        x = einops.einsum(
            x,
            self.weight,
            "... nh d, nh out_d d -> ... nh out_d",
        )
        x = einops.rearrange(x, "... nh out_d -> ... (nh out_d)")
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self):
        return (
            f"dim={self.dim}, "
            f"num_heads={self.head_dim}, "
            f"bias={self.bias is not None}, "
        )


class CausalConv1d(nn.Module):
    """
    Implements causal depthwise convolution of a time series tensor.
    Input:  Tensor of shape (B,T,F), i.e. (batch, time, feature)
    Output: Tensor of shape (B,T,F)

    Args:
        dim: number of features in the input tensor
        kernel_size: size of the kernel for the depthwise convolution
        bias: whether to use bias in the depthwise convolution

    channel_mixing: whether to use channel mixing (i.e. groups=1) or not (i.e. groups=dim)
                    If True, it mixes the convolved features across channels.
                    If False, all the features are convolved independently.
    """

    def __init__(self, dim, kernel_size=4, bias=True):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.bias = bias
        # padding of this size assures temporal causality.
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=self.pad,
            groups=dim,
            bias=bias,
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # conv requires dim first
        x = einops.rearrange(x, "b l d -> b d l")
        # causal conv1d
        x = self.conv(x)
        x = x[:, :, :-self.pad]
        # back to dim last
        x = einops.rearrange(x, "b d l -> b l d")
        return x


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False. """

    def __init__(
            self,
            ndim: int = -1,
            weight: bool = True,
            bias: bool = False,
            eps: float = 1e-5,
            residual_weight: bool = True,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(ndim)) if weight else None
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps
        self.residual_weight = residual_weight
        self.ndim = ndim
        self.reset_parameters()

    @property
    def weight_proxy(self) -> [torch.Tensor, None]:
        if self.weight is None:
            return None
        if self.residual_weight:
            return 1.0 + self.weight
        else:
            return self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            normalized_shape=(self.ndim,),
            weight=self.weight_proxy,
            bias=self.bias,
            eps=self.eps,
        )

    def reset_parameters(self):
        if self.weight_proxy is not None:
            if self.residual_weight:
                nn.init.zeros_(self.weight)
            else:
                nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class MultiHeadLayerNorm(LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, "Input must be 4D tensor (B, NH, S, DH)"
        B, NH, S, DH = x.shape

        gn_in_1 = x.transpose(1, 2)  # (B, S, NH, DH)
        gn_in_2 = gn_in_1.reshape(B * S, NH * DH)  # (B * S, NH * DH)
        out = F.group_norm(
            gn_in_2,
            num_groups=NH,
            weight=self.weight_proxy,
            bias=self.bias,
            eps=self.eps,
        )  # .to(x.dtype)
        # (B * S), (NH * DH) -> (B, S, NH, DH) -> (B, NH, S, DH)
        out = out.view(B, S, NH, DH).transpose(1, 2)
        return out


def parallel_stabilized_simple(
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        rgate_preact: torch.Tensor,
        ugate_preact: torch.Tensor,
        lower_triangular_matrix: torch.Tensor = None,
        stabilize_rowwise: bool = True,
) -> torch.Tensor:
    """
    This is the mGRU cell in parallel form.
    This version is stabilized. We control the range of exp() arguments by
    ensuring that they are always smaller than 0.0 by subtracting the maximum.

    Args:
        :param queries: (torch.Tensor) (B, NH, S, DH)
        :param keys: (torch.Tensor) (B, NH, S, DH)
        :param values: (torch.Tensor) (B, NH, S, DH)
        :param rgate_preact: (torch.Tensor) (B, NH, S, 1)
        :param ugate_preact: (torch.Tensor) (B, NH, S, 1)
        :param lower_triangular_matrix: (torch.Tensor) (S,S). Defaults to None.
        :param stabilize_rowwise: (bool) Wether to stabilize the combination matrix D rowwise (take maximum per row).
            Alternative: Subtract the maximum over all rows. Defaults to True.

    Returns:
        torch.Tensor: (B, NH, S, DH), h_tilde_state
    """

    B, NH, S, DH = queries.shape
    _dtype, _device = queries.dtype, queries.device

    coef1 = ugate_preact + rgate_preact - ugate_preact * rgate_preact
    coef2 = 1. - ugate_preact

    # reset gate matrix
    logs_coef1 = F.logsigmoid(coef1)  # (B, NH, S, 1)
    if lower_triangular_matrix is None or S < lower_triangular_matrix.size(-1):
        ltr = torch.tril(torch.ones((S, S), dtype=torch.bool, device=_device))
    else:
        ltr = lower_triangular_matrix
    assert ltr.dtype == torch.bool, f"lower_triangular_matrix must be of dtype bool, got {ltr.dtype}"

    log_coef1_cumsum = torch.cat(
        [
            torch.zeros((B, NH, 1, 1), dtype=_dtype, device=_device),
            torch.cumsum(logs_coef1, dim=-2),
        ],
        dim=-2,
    )  # (B, NH, S+1, 1)
    # for each batch/head this is a matrix of shape (S+1, S+1) containing the cumsum of the log reset gate values
    # in the second dimension (colum dimension). Each row has the same is a copy of the first row.
    # First entry of each row is zero.
    rep_log_coef1_cumsum = log_coef1_cumsum.repeat(1, 1, 1, S + 1)  # (B, NH, S+1, S+1)
    # Now in each row cut off / subtract the reset gate values of the later timesteps
    # where col j > row i
    _log_coef1_matrix = rep_log_coef1_cumsum - rep_log_coef1_cumsum.transpose(-2, -1)  # (B, NH, S+1, S+1)
    # Causal masking & selection of the correct submatrix, such that forgetgate at timestep t is not applied
    # to the input at timestep t
    log_coef1_matrix = torch.where(ltr, _log_coef1_matrix[:, :, 1:, 1:], -float("inf"))  # (B, NH, S, S)

    # gate decay matrix D (combination of reset gate and update gate)
    log_D_matrix = log_coef1_matrix + coef2.transpose(-2, -1)  # (B, NH, S, S)
    # D matrix stabilization
    if stabilize_rowwise:
        max_log_D, _ = torch.max(log_D_matrix, dim=-1, keepdim=True)  # (B, NH, S, 1)
    else:
        max_log_D = torch.max(log_D_matrix.view(B, NH, -1), dim=-1, keepdim=True)[0].unsqueeze(-1)
        # (B, NH, 1, 1)
    log_D_matrix_stabilized = log_D_matrix - max_log_D  # (B, NH, S, S)
    D_matrix = torch.exp(log_D_matrix_stabilized)  # (B, NH, S, S)

    keys_scaled = keys / math.sqrt(DH)

    # combination matrix C
    qk_matrix = queries @ keys_scaled.transpose(-2, -1)  # (B, NH, S, S)
    h_state = (qk_matrix * D_matrix) @ values  # (B, NH, S, DH)

    return h_state


class MatricGRUCell(nn.Module):
    def __init__(self, dim, num_heads, norm_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.rgate = nn.Linear(3 * dim, num_heads)
        self.ugate = nn.Linear(3 * dim, num_heads)
        self.outnorm = MultiHeadLayerNorm(ndim=dim, weight=True, bias=norm_bias)
        self.causal_mask_cache = {}
        self.reset_parameters()

    def reset_parameters(self):
        self.outnorm.reset_parameters()
        # forget gate initialization
        torch.nn.init.zeros_(self.rgate.weight)
        bias_linspace_init_(self.rgate.bias, start=3.0, end=6.0)
        # input gate initialization
        torch.nn.init.zeros_(self.ugate.weight)
        torch.nn.init.normal_(self.ugate.bias, mean=0.0, std=0.1)

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
    ) -> torch.Tensor:
        B, S, _ = q.shape  # (B, S, H)

        ru_gate_input = torch.cat([q, k, v], dim=-1)
        q = q.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)
        k = k.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)
        v = v.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)

        _, _, NH, DH = q.shape

        q = q.transpose(1, 2)  # (B, NH, S, DH)
        k = k.transpose(1, 2)  # (B, NH, S, DH)
        v = v.transpose(1, 2)  # (B, NH, S, DH)

        # compute reset and update gate pre-activations
        rgate_preact = self.rgate(ru_gate_input)  # (B, S, NH)
        rgate_preact = rgate_preact.transpose(-1, -2).unsqueeze(-1)  # (B, NH, S, 1)
        ugate_preact = self.ugate(ru_gate_input)  # (B, S, NH)
        ugate_preact = ugate_preact.transpose(-1, -2).unsqueeze(-1)  # (B, NH, S, 1)#

        # cache causal mask to avoid memory allocation in every iteration
        if S in self.causal_mask_cache:
            causal_mask = self.causal_mask_cache[(S, str(q.device))]
        else:
            causal_mask = torch.tril(torch.ones(S, S, dtype=torch.bool, device=q.device))
            self.causal_mask_cache[(S, str(q.device))] = causal_mask

        h_state = parallel_stabilized_simple(
            queries=q,
            keys=k,
            values=v,
            rgate_preact=rgate_preact,
            ugate_preact=ugate_preact,
            lower_triangular_matrix=causal_mask,
        )  # (B, NH, 1 DH), ((B, NH, DH, DH), (B, NH, DH, 1), (B, NH, 1, 1))

        h_state_norm = self.outnorm(h_state)  # (B, NH, S, DH)
        h_state_norm = h_state_norm.transpose(1, 2).reshape(B, S, -1)  # (B, NH, S, DH) -> (B, S, NH, DH) -> (B, S, H)

        return h_state_norm


class ViGLayer(nn.Module):
    def __init__(
            self,
            dim,
            if_flip=False,
            expansion=2,
            num_heads=4,
            proj_bias=True,
            norm_bias=True,
            conv_bias=True,
            conv_kernel_size=4,
            conv_kind="1d",
            seqlens=None,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.if_flip = if_flip
        self.expansion = expansion
        self.num_heads = num_heads
        self.proj_bias = proj_bias
        self.conv_kernel_size = conv_kernel_size
        self.conv_kind = conv_kind

        inner_dim = expansion * dim
        head_dim = inner_dim // self.num_heads

        self.proj_up = nn.Linear(
            in_features=dim,
            out_features=inner_dim,
            bias=proj_bias,
        )
        self.q_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            head_dim=head_dim,
            bias=proj_bias,
        )
        self.k_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            head_dim=head_dim,
            bias=proj_bias,
        )
        self.v_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            head_dim=head_dim,
            bias=proj_bias,
        )

        if conv_kind == "1d":
            self.conv = CausalConv1d(
                dim=inner_dim,
                kernel_size=conv_kernel_size,
                bias=conv_bias,
            )
        elif conv_kind == "2d":
            assert conv_kernel_size % 2 == 1, \
                f"same output shape as input shape is required -> even kernel sizes not supported"
            self.conv = SequenceConv2d(
                in_channels=inner_dim,
                out_channels=inner_dim,
                kernel_size=conv_kernel_size,
                padding=conv_kernel_size // 2,
                groups=inner_dim,
                bias=conv_bias,
                seqlens=seqlens,
            )
        elif conv_kind == 'wt2d':
            assert conv_kernel_size % 2 == 1, \
                f"same output shape as input shape is required -> even kernel sizes not supported"
            self.conv = WTSequenceConv2d(
                in_channels=inner_dim,
                out_channels=inner_dim,
                kernel_size=conv_kernel_size,
                bias=conv_bias,
                seqlens=seqlens,
                wt_levels=2
            )
        elif conv_kind == 'none':
            pass
        else:
            raise NotImplementedError
        self.gru_cell = MatricGRUCell(
            dim=inner_dim,
            num_heads=self.num_heads,
            norm_bias=norm_bias,
        )
        self.learnable_skip = nn.Parameter(torch.ones(inner_dim))

        self.proj_down = nn.Linear(
            in_features=inner_dim,
            out_features=dim,
            bias=proj_bias,
        )
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape

        if self.if_flip:
            x = x.flip(dims=[1])

        # up-projection
        x_inner = self.proj_up(x)

        x_gru = x_inner

        # mgru branch
        if self.conv_kind != "none":
            x_gru_conv_act = F.silu(self.conv(x_gru))
        else:
            x_gru_conv_act = x_gru

        q = self.q_proj(x_gru_conv_act)
        k = self.k_proj(x_gru_conv_act)
        v = self.v_proj(x_gru)
        h_state = self.gru_cell(q=q, k=k, v=v)
        h_state_skip = h_state + (self.learnable_skip * x_gru_conv_act)

        # down-projection
        x = self.proj_down(h_state_skip)

        # reverse alternating flip
        if self.if_flip:
            x = x.flip(dims=[1])

        return x

    def reset_parameters(self):
        # init inproj
        small_init_(self.proj_up.weight, dim=self.dim)
        if self.proj_up.bias is not None:
            nn.init.zeros_(self.proj_up.bias)
        # init outproj (original mLSTM uses num_blocks=1)
        wang_init_(self.proj_down.weight, dim=self.dim, num_blocks=1)
        if self.proj_down.bias is not None:
            nn.init.zeros_(self.proj_down.bias)

        nn.init.ones_(self.learnable_skip)

        def _init_qkv_proj(qkv_proj: LinearHeadwiseExpand):
            # use the embedding dim instead of the inner embedding dim
            small_init_(qkv_proj.weight, dim=self.dim)
            if qkv_proj.bias is not None:
                nn.init.zeros_(qkv_proj.bias)

        _init_qkv_proj(self.q_proj)
        _init_qkv_proj(self.k_proj)
        _init_qkv_proj(self.v_proj)

        self.gru_cell.reset_parameters()


class ViGBlock(nn.Module):
    def __init__(
            self,
            dim,
            if_flip=False,
            expansion=2,
            num_heads=4,
            drop_path=0.0,
            conv_kind="1d",
            conv_kernel_size=3,
            proj_bias=True,
            norm_bias=True,
            seqlens=None,
    ):
        super().__init__()

        self.drop_path = DropPath(drop_prob=drop_path)
        self.norm = LayerNorm(ndim=dim, weight=True, bias=norm_bias)
        self.layer = ViGLayer(
            dim=dim,
            if_flip=if_flip,
            expansion=expansion,
            num_heads=num_heads,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel_size,
            seqlens=seqlens,
            norm_bias=norm_bias,
            proj_bias=proj_bias,
        )
        self.reset_parameters()

    def _forward_path(self, x):
        x = self.norm(x)
        x = self.layer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop_path(x, self._forward_path)
        return x

    def reset_parameters(self):
        self.layer.reset_parameters()
        self.norm.reset_parameters()


# bidirectional xLSTM
class ViGBlockPair(nn.Module):
    def __init__(
            self,
            dim,
            drop_path=0.0,
            expansion=2,
            num_heads=4,
            conv_kind="1d",
            conv_kernel_size=3,
            proj_bias=True,
            norm_bias=True,
            seqlens=None,
    ):
        super().__init__()
        self.rowwise_from_top_left = ViGBlock(
            dim=dim,
            if_flip=False,
            expansion=expansion,
            num_heads=num_heads,
            drop_path=drop_path,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel_size,
            proj_bias=proj_bias,
            norm_bias=norm_bias,
            seqlens=seqlens
        )
        self.rowwise_from_bot_right = ViGBlock(
            dim=dim,
            if_flip=True,
            expansion=expansion,
            num_heads=num_heads,
            drop_path=drop_path,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel_size,
            proj_bias=proj_bias,
            norm_bias=norm_bias,
            seqlens=seqlens
        )

    def forward(self, x):
        x = self.rowwise_from_top_left(x)
        x = self.rowwise_from_bot_right(x)
        return x


class MultiDirectionGRULayer(nn.Module):
    def __init__(
            self,
            dim,
            expansion=2,
            num_heads=4,
            directions="default",
            proj_bias=True,
            norm_bias=True,
            conv_bias=True,
            conv_kernel_size=4,
            conv_kind="2d",
            seqlens=(8, 8),
    ):
        super().__init__()

        if directions == "default":
            directions = MultiScan.GL_DIRECTIONS
        assert isinstance(directions, (list, tuple))
        for i in range(len(directions)):
            assert directions[i] in MultiScan.TOTAL_DIRECTIONS

        assert dim % num_heads == 0
        self.dim = dim
        self.expansion = expansion
        self.num_heads = num_heads
        self.proj_bias = proj_bias
        self.conv_kernel_size = conv_kernel_size
        self.conv_kind = conv_kind
        self.directions = directions

        inner_dim = expansion * dim
        head_dim = inner_dim // self.num_heads

        self.proj_up = nn.Linear(in_features=dim, out_features=inner_dim, bias=proj_bias)

        for i in range(len(self.directions)):
            q_proj = LinearHeadwiseExpand(dim=inner_dim, head_dim=head_dim, bias=proj_bias)
            setattr(self, f"q_proj_{i}", q_proj)
            k_proj = LinearHeadwiseExpand(dim=inner_dim, head_dim=head_dim, bias=proj_bias)
            setattr(self, f"k_proj_{i}", k_proj)
            v_proj = LinearHeadwiseExpand(dim=inner_dim, head_dim=head_dim, bias=proj_bias)
            setattr(self, f"v_proj_{i}", v_proj)

            if conv_kind == "1d":
                conv = CausalConv1d(dim=inner_dim, kernel_size=conv_kernel_size, bias=conv_bias)
            elif conv_kind == "2d":
                assert conv_kernel_size % 2 == 1, \
                    f"same output shape as input shape is required -> even kernel sizes not supported"
                conv = SequenceConv2d(
                    in_channels=inner_dim,
                    out_channels=inner_dim,
                    kernel_size=conv_kernel_size,
                    padding=conv_kernel_size // 2,
                    groups=inner_dim,
                    bias=conv_bias,
                    seqlens=seqlens,
                )
            elif conv_kind == 'wt2d':
                assert conv_kernel_size % 2 == 1, \
                    f"same output shape as input shape is required -> even kernel sizes not supported"
                conv = WTSequenceConv2d(
                    in_channels=inner_dim,
                    out_channels=inner_dim,
                    kernel_size=conv_kernel_size,
                    bias=conv_bias,
                    seqlens=seqlens,
                    wt_levels=2
                )
            elif conv_kind == 'none':
                conv = None
            else:
                raise NotImplementedError
            setattr(self, f"conv_{i}", conv)

            gru_cell = MatricGRUCell(dim=inner_dim, num_heads=self.num_heads, norm_bias=norm_bias)
            setattr(self, f'grucell_{i}', gru_cell)

        self.multiscan = MultiScan(directions=self.directions, mode="add", seqlens=seqlens)
        self.learnable_skip = nn.Parameter(torch.ones(inner_dim))
        self.proj_down = nn.Linear(in_features=inner_dim, out_features=dim, bias=proj_bias)
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape

        # up-projection
        x_inner = self.proj_up(x)
        xs = self.multiscan.multi_scan(x_inner)  # [[B,L,D], [B,L,D], ..., [B,L,D]]
        outs = []  # [[B,L,D], [B,L,D], ..., [B,L,D]]
        for i, x in enumerate(xs):
            # x = rearrange(x, 'b d l -> b l d')
            x_conv_act = x
            if self.conv_kind != "none":
                x_conv_act = F.silu(getattr(self, f"conv_{i}")(x))

            q = getattr(self, f"q_proj_{i}")(x_conv_act)
            k = getattr(self, f"k_proj_{i}")(x_conv_act)
            v = getattr(self, f"v_proj_{i}")(x)

            h = getattr(self, f"grucell_{i}")(q=q, k=k, v=v)

            outs.append(h)
        outs = self.multiscan.multi_reverse(outs)
        h_state = self.multiscan(outs)

        h_state_skip = h_state + (self.learnable_skip * x_inner)
        # down-projection
        x = self.proj_down(h_state_skip)

        return x

    def reset_parameters(self):
        # init inproj
        small_init_(self.proj_up.weight, dim=self.dim)
        if self.proj_up.bias is not None:
            nn.init.zeros_(self.proj_up.bias)
        # init outproj (original mLSTM uses num_blocks=1)
        wang_init_(self.proj_down.weight, dim=self.dim, num_blocks=1)
        if self.proj_down.bias is not None:
            nn.init.zeros_(self.proj_down.bias)

        nn.init.ones_(self.learnable_skip)

        def _init_qkv_proj(qkv_proj: LinearHeadwiseExpand):
            # use the embedding dim instead of the inner embedding dim
            small_init_(qkv_proj.weight, dim=self.dim)
            if qkv_proj.bias is not None:
                nn.init.zeros_(qkv_proj.bias)

        for i in range(len(self.directions)):
            _init_qkv_proj(getattr(self, f"q_proj_{i}"))
            _init_qkv_proj(getattr(self, f"k_proj_{i}"))
            _init_qkv_proj(getattr(self, f"v_proj_{i}"))

            getattr(self, f"grucell_{i}").reset_parameters()


class MultiDirectionGRUBlock(nn.Module):
    def __init__(
            self,
            dim,
            directions="default",
            expansion=2,
            num_heads=4,
            drop_path=0.0,
            conv_kind="2d",
            conv_kernel_size=3,
            proj_bias=True,
            norm_bias=True,
            seqlens=(8, 8),
    ):
        super().__init__()

        self.drop_path = DropPath(drop_prob=drop_path)
        self.norm = LayerNorm(ndim=dim, weight=True, bias=norm_bias)

        self.layer = MultiDirectionGRULayer(
            dim=dim,
            expansion=expansion,
            num_heads=num_heads,
            directions=directions,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel_size,
            seqlens=seqlens,
            norm_bias=norm_bias,
            proj_bias=proj_bias,
        )

        self.reset_parameters()

    def _forward_path(self, x):
        x = self.norm(x)
        x = self.layer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop_path(x, self._forward_path)
        return x

    def reset_parameters(self):
        self.layer.reset_parameters()
        self.norm.reset_parameters()





import torch.nn.functional as F

from utils.utils import *


def make_divisible(v, divisor):
    # Function ensures all layers have a channel number that is divisible by 8
    # https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    return math.ceil(v / divisor) * divisor


class Flatten(nn.Module):
    # Use after nn.AdaptiveAvgPool2d(1) to remove last 2 dimensions
    def forward(self, x):
        return x.view(x.size(0), -1)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class FeatureConcat(nn.Module):
    def __init__(self, layers, dual_view=False):
        super(FeatureConcat, self).__init__()
        self.layers = layers  # layer indices
        self.multiple = len(layers) > 1  # multiple layers flag
        self.dual_view = dual_view

        self.layers_to_warp = [i for i in self.layers if i > 0] if self.dual_view else []
        assert len(self.layers_to_warp) in [0,1]

        self.inner_count = 0

    def forward(self, x, outputs, outputs_sub=[], H_img_bev=None, writer=None): # VISMODE1113
        ### outputs_sub is given when we are using both bev and original view. 
        ### Need to concat features from both views at layers that are dual_view_accept_layers (self.dual_view==True)
        if len(outputs_sub) == 0 or not self.dual_view:
            return torch.cat([outputs[i] for i in self.layers], 1) if self.multiple else outputs[self.layers[0]]
        else:
            assert len(self.layers_to_warp) == 1
            layer_warp = self.layers_to_warp[0]
            assert layer_warp in [36, 61], layer_warp   # for yolov3-spp
            grid_shape = outputs[layer_warp].shape
            height = grid_shape[2]
            width = grid_shape[3]
            batch_size = grid_shape[0]
            # xs = torch.linspace(-1, 1, width)   # if align_corners==True, the corner pixel center is -1 / 1
            # ys = torch.linspace(-1, 1, height)  # if align_corners==False, the corner of corner pixels is -1 / 1, thererfore the corner pixel center is not -1 / 1
            xs = torch.arange(width) + 0.5
            ys = torch.arange(height) + 0.5
            # xs = (torch.arange(width) + 0.5) / width * 2 - 1
            # ys = (torch.arange(height) + 0.5) / height * 2 - 1
            
            base_grid = torch.stack(
                torch.meshgrid([xs, ys])).transpose(1, 2).to(device=outputs[layer_warp].device, dtype=outputs[layer_warp].dtype)  # 2xHxW
            base_grid = torch.unsqueeze(base_grid, dim=0).permute(0, 2, 3, 1).expand(batch_size, -1, -1, -1)  # BxHxWx2
            grid_flat = base_grid.reshape(batch_size, -1, 2)    # B*N*2
            grid_flat_homo = torch.nn.functional.pad(grid_flat, (0, 1), "constant", 1.0)    # B*N*3

            H_to_use = H_img_bev[:, 1] if layer_warp == 36 else H_img_bev[:, 2] # B*3*3

            grid_warped_flat_homo = torch.matmul(
                H_to_use.unsqueeze(1), grid_flat_homo.unsqueeze(-1))    # B*1*3*3 x B*N*3*1 => B*N*3*1
            grid_warped_flat_homo = torch.squeeze(grid_warped_flat_homo, dim=-1)    # B*N*3

            grid_warped_flat = grid_warped_flat_homo[..., :-1] / grid_warped_flat_homo[..., -1:]    # B*N*2
            grid_warped = grid_warped_flat.view(batch_size, height, width, 2)   # B*H*W*2

            height_ori = outputs_sub[layer_warp].shape[2]
            width_ori = outputs_sub[layer_warp].shape[3]
            grid_warped[..., 0] = grid_warped[..., 0] / width_ori * 2 - 1
            grid_warped[..., 1] = grid_warped[..., 1] / height_ori * 2 - 1

            feature_warped = torch.nn.functional.grid_sample(outputs_sub[layer_warp], grid_warped, mode='bilinear', padding_mode='zeros', align_corners=False)

            if writer is not None: # VISMODE1113
                print("intm feature shape ori",layer_warp, outputs_sub[layer_warp].shape)
                # writer.add_images("feat_ori_%d"%layer_warp, outputs_sub[layer_warp][:,:3], self.inner_count)
                # writer.add_images("feat_warped_ori_%d"%layer_warp, feature_warped[:,:3], self.inner_count)
                for ii in range(batch_size):
                    writer.add_image("feat_ori_%d/%d"%(layer_warp, ii), outputs_sub[layer_warp][ii,:3], self.inner_count)
                    writer.add_image("feat_warped_ori_%d/%d"%(layer_warp, ii), feature_warped[ii,:3], self.inner_count)
                for layer_i in self.layers:
                    print("intm feature shape bev ",layer_i, outputs_sub[layer_i].shape)
                    # writer.add_images("feat_bev_%d_%d"%(layer_warp, layer_i), outputs[layer_i][:,:3], self.inner_count)
                    for ii in range(batch_size):
                        writer.add_image("feat_bev_%d_%d/%d"%(layer_warp, layer_i, ii), outputs[layer_i][ii,:3], self.inner_count)
                self.inner_count = self.inner_count + 1

            if self.multiple:
                return torch.cat([outputs[i] for i in self.layers] + [feature_warped], 1) # i > 0 means it is absolute (positive) index. outputs_sub only accept absolute index. 
            else:
                assert self.layers[0] > 0   # i > 0 means it is absolute (positive) index. outputs_sub only accept absolute index. 
                return torch.cat([outputs[self.layers[0]], feature_warped] , 1)

            # if self.multiple:

            #     return torch.cat([outputs[i] for i in self.layers] + [outputs_sub[i] for i in self.layers if i > 0], 1) # i > 0 means it is absolute (positive) index. outputs_sub only accept absolute index. 
            # else:
            #     assert self.layers[0] > 0   # i > 0 means it is absolute (positive) index. outputs_sub only accept absolute index. 
            #     return torch.cat([outputs_[self.layers[0]] for outputs_ in [outputs, outputs_sub]], 1)


class WeightedFeatureFusion(nn.Module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, layers, weight=False):
        super(WeightedFeatureFusion, self).__init__()
        self.layers = layers  # layer indices
        self.weight = weight  # apply weights boolean
        self.n = len(layers) + 1  # number of layers
        if weight:
            self.w = nn.Parameter(torch.zeros(self.n), requires_grad=True)  # layer weights

    def forward(self, x, outputs):
        # Weights
        if self.weight:
            w = torch.sigmoid(self.w) * (2 / self.n)  # sigmoid weights (0-1)
            x = x * w[0]

        # Fusion
        nx = x.shape[1]  # input channels
        for i in range(self.n - 1):
            a = outputs[self.layers[i]] * w[i + 1] if self.weight else outputs[self.layers[i]]  # feature to add
            na = a.shape[1]  # feature channels

            # Adjust channels
            if nx == na:  # same shape
                x = x + a
                # # VISMODE1113 divide by 2
                # x = x / 2
            elif nx > na:  # slice input
                x[:, :na] = x[:, :na] + a  # or a = nn.ZeroPad2d((0, 0, 0, 0, 0, dc))(a); x = x + a
                # # VISMODE1113 divide by 2
                # x[:, :na] = x[:, :na] / 2
            else:  # slice feature
                x = x + a[:, :nx]
                # # VISMODE1113 divide by 2
                # x = x / 2

        return x


class MixConv2d(nn.Module):  # MixConv: Mixed Depthwise Convolutional Kernels https://arxiv.org/abs/1907.09595
    def __init__(self, in_ch, out_ch, k=(3, 5, 7), stride=1, dilation=1, bias=True, method='equal_params'):
        super(MixConv2d, self).__init__()

        groups = len(k)
        if method == 'equal_ch':  # equal channels per group
            i = torch.linspace(0, groups - 1E-6, out_ch).floor()  # out_ch indices
            ch = [(i == g).sum() for g in range(groups)]
        else:  # 'equal_params': equal parameter count per group
            b = [out_ch] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            ch = np.linalg.lstsq(a, b, rcond=None)[0].round().astype(int)  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([nn.Conv2d(in_channels=in_ch,
                                          out_channels=ch[g],
                                          kernel_size=k[g],
                                          stride=stride,
                                          padding=k[g] // 2,  # 'same' pad
                                          dilation=dilation,
                                          bias=bias) for g in range(groups)])

    def forward(self, x):
        return torch.cat([m(x) for m in self.m], 1)


# Activation functions below -------------------------------------------------------------------------------------------
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.sigmoid(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sx = torch.sigmoid(x)  # sigmoid(ctx)
        return grad_output * (sx * (1 + x * (1 - sx)))


class MishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.mul(torch.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sx = torch.sigmoid(x)
        fx = F.softplus(x).tanh()
        return grad_output * (fx + x * sx * (1 - fx * fx))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class MemoryEfficientMish(nn.Module):
    def forward(self, x):
        return MishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class HardSwish(nn.Module):  # https://arxiv.org/pdf/1905.02244.pdf
    def forward(self, x):
        return x * F.hardtanh(x + 3, 0., 6., True) / 6.


class Mish(nn.Module):  # https://github.com/digantamisra98/Mish
    def forward(self, x):
        return x * F.softplus(x).tanh()

import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Misc
# 下面两个是图像质量评估指标的MSE和PSNR, 具体可以参考: https://zhuanlan.zhihu.com/p/150865007
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
# 将元素大小限制在255 * 0, 255 * 1之间
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            # 2^{0到max_freq等间距取N_freqs个点}
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            # 0到2^{max_freq}等间距取N_freqs个点
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        # 构建公式(4), 保存到embed_fns中(这种写法非常nice, 感觉挺简洁的)
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        # 这里是不改变第一个维度的拼接, 从[x,y,z]变成[[x1,x2,...], [y1,y2,...], [z1,z2,...]]
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    """
    根据(5.1 Positional encoding)构建一个编码器
    :param multires: log2 of max freq for positional encoding (2D direction)
    :param i: 是否启用位置编码(0 use, -1 not use)
    :return: 编码器, 输出的维数
    """
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """
        Init NeRF Model
        :param D: 网络的层数(深度)
        :param W: 每层网络的通道数(宽度)
        :param input_ch: input channel(3D坐标)
        :param input_ch_views: input view channel(2D观察方向, 用3D的方向向量表示)
        :param output_ch: output channel(rgb, alpha)
        :param skips: skip connection layer index
        :param use_viewdirs: use view directions or not(If True, use viewing direction of a point in space in model.)
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        # nn.ModuleList 是一个储存不同 module, 并自动将每个 module 的 parameters 添加到网络之中的容器
        # 但是,  nn.ModuleList 并没有定义一个网络, 它只是将不同的模块储存在一起, 这些模块之间并没有什么先后顺序可言（所以也没有forward函数）
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            # TODO 这是什么样的一个情况?
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        # torch.split 将输入x(应该是经过位置编码的x?)切分成input_ch和input_ch_views长的两块
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts

        # 根据论文, 输入3D坐标, 首先通过具有8个完全连接层的MLP, 其中有一个skip connection
        # 对skip的说明: We follow the DeepSDF architecture and include a skip connection that
        #   concatenates this input to the fifth layer’s activation.
        #   更详细的介绍: https://www.analyticsvidhya.com/blog/2021/08/all-you-need-to-know-about-skip-connections
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            # 通过附加层(additional layer)输出volume density(alpha)和256维的feature vector
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)

            # 然后该特征向量和相机射线的方向连接起来, 传递到一个额外的完全连接层（128通道）
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)
            # 得到与视图相关的RGB颜色
            rgb = self.rgb_linear(h)
            # 最终输出(rgb, alpha)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears + 1]))

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear + 1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears + 1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear + 1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear + 1]))


# Ray helpers
def get_rays(H, W, K, c2w):
    """
    获取(4 Volume Rendering with Radiance Fields)中的o 和 d
    :param H: image H
    :param W: image W
    :param K: 相机内参
    :param c2w: 相机外参(相机到世界)
    :return: rays_o and rays_d
    """
    # i, j的shape都是[W, H]
    # i
    # [0,......]
    # [1,......]
    # [W-1,....]
    # j
    # [0,..,H-1]
    # [0,..,H-1]
    # [0,..,H-1]
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
                          torch.linspace(0, H - 1, H))  # pytorch's meshgrid has indexing='ij'
    # 转置成[H, W]
    i = i.t()
    j = j.t()
    # 构造网格点坐标矩阵, 下面最里面的维度是[i,j,-1], shape为[H,W,3]
    # 这里应该是从image plane 到了 camera plane, 我认为[i,j,-1]是每条光线的方向向量
    dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)

    # torch.sum的dim参数可以参考: https://blog.csdn.net/qq_31239371/article/details/123668936
    # 下面torch.sum(vector,dim=-1)是对最里面的维度做累加
    # 因为H,W,1,3 * 3,3得到了H,W,3,3, 对最里面的维度做累加就能够恢复成旋转到世界坐标系的[H,W,3]
    # 这里涉及到了广播机制, 可以参考: https://zhuanlan.zhihu.com/p/86997775
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                       -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    """
    获取(4 Volume Rendering with Radiance Fields)中的o 和 d, 但这里使用了numpy的float32
    :param H: image H
    :param W: image W
    :param K: 相机内参
    :param c2w: 相机外参(相机到世界)
    :return: rays_o and rays_d
    """
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                    -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


# 把光线原点移动到near平面的原因为: (A Additional Implementation Details)
# Our dataset of real images contains content that can exist anywhere between the closest point and infinity,
# so we use normalized device coordinates to map the depth range of these points into [−1, 1].
# This shifts all the ray origins to the near plane of the scene,
# maps the perspective rays of the camera to parallel rays in the transformed volume,
# and uses disparity (inverse depth) instead of metric depth, so all coordinates are now bounded.
# TODO 三个疑问:【这是对论文的疑问】
#  1. 没有深度图, 如何做到归一化深度
#  2. 为什么归一化之后光线原点移动到了near平面(光线原点应该就是相机原点?)
#  3. 在论文公式推导部分, camera space to NDC space的投影矩阵怎么来的
def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """
    translate rays from camera space to NDC space
    :param H: image H
    :param W: image W
    :param focal: camera focal
    :param near: near clipping plane
    :param rays_o: rays_o in camera space
    :param rays_d: rays_d in camera space
    :return: rays_o and rays_d in NDC space
    """
    # 下面公式的推导见论文(C NDC ray space derivation)

    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


# TODO 下面这个函数, 以及论文对应部分都有点看不懂
#  不过我估计是根据权重，在权重大的地方多采样，权重小的地方少采样
# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    """
    Sample N_samples samples from bins with distribution defined by weights.
    :param bins: z_vals_mid(可以认为是采样点)
    :param weights: 公式(5)中的权重
    :param N_samples: the number of samples to draw from the distribution
    :param det: deterministic or not
    :param pytest:
    :return: samples: the sampled samples
    """
    # Get pdf(概率密度函数)
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    # Get cdf(累积分布函数)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples, 均匀/随机采样
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF, 逆变换采样
    # 把tensor变成内存连续分布形式
    u = u.contiguous()
    # 用高维的searchsorted算子去寻找坐标值的索引, 返回和u一样大小的tensor, 其元素是CDF中大于等于u的索引
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    # lower + 线性插值
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

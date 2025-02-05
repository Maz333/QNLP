import torch
import torch.nn as nn


def torch_dct_2d(image_tensor):
    """
    使用 PyTorch 的 FFT 模块实现 2D DCT
    :param image_tensor: 输入图像张量 (H, W)
    :return: DCT 转换后的图像张量
    """
    # 对行进行 DCT (类似离散傅里叶变换)
    dct_rows = torch.fft.fft(image_tensor, dim=0, norm="ortho")
    # 对列进行 DCT
    dct_2d = torch.fft.fft(dct_rows, dim=1, norm="ortho")
    return dct_2d.real  # 返回实部，因为 DCT 是实值变换


def extract_dct_features_torch(image_tensor, block_size=8, num_features=4):
    """
    使用 PyTorch 提取图像的 DCT 特征
    :param image_tensor: 输入图像张量 (C, H, W)
    :param block_size: DCT 分块大小
    :param num_features: 每个块保留的 DCT 特征数量
    :return: DCT 特征张量
    """
    H, W = image_tensor.shape[-2:]  # 获取图像高度和宽度
    dct_features = []

    for i in range(0, H, block_size):
        for j in range(0, W, block_size):
            block = image_tensor[..., i : i + block_size, j : j + block_size]
            if block.shape[-2:] != (block_size, block_size):
                continue
            # 对每个块计算 DCT
            dct_block = torch_dct_2d(block)
            # 提取前 num_features 个特征
            dct_features.append(dct_block.flatten()[:num_features])

    return torch.cat(dct_features)


class DCTFeatureExtractor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dim_reduction = nn.Linear(input_dim, output_dim)

    def forward(self, imgs_tensors, block_size=8, num_features=4):
        dct_features_list = []
        for img_tensor in imgs_tensors:
            # 提取单张图像的 DCT 特征 (仅使用第一个通道)
            dct_features = extract_dct_features_torch(
                img_tensor[0], block_size, num_features
            )
            dct_features_list.append(dct_features)

        # 将所有特征堆叠成批量张量
        dct_features_tensor = torch.stack(dct_features_list)
        return self.dim_reduction(dct_features_tensor)

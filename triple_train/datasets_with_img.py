from torch.utils.data import Dataset
import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F


class FakeNewsDataset(Dataset):
    def __init__(self, mode, datasets, tokenize, path, img_path):
        """
        初始化 FakeNewsDataset。
        :param mode: 模式 'train' 或 'test'
        :param datasets: 数据集名称
        :param tokenize: 传入的 tokenizer 函数
        :param path: 数据集 CSV 文件路径
        :param img_path: 图像所在路径
        """
        assert mode in ["train", "test"]
        self.mode = mode
        self.img_path = img_path

        self.df = pd.read_csv(path + datasets + "_" + mode + ".csv").fillna("")
        self.len = len(self.df)
        self.tokenizer = tokenize

    def __getitem__(self, idx):
        """
        获取数据集中的一个样本。
        :param idx: 索引
        :return: (tokens_tensor, segments_tensor, image_tensor, label_tensor)
        """
        statement = self.df.iloc[idx]["content"]
        label = self.df.iloc[idx]["label"]
        img = self.df.iloc[idx]["image"]
        label_tensor = torch.tensor(label)

        word_pieces = ["[CLS]"]
        statement = self.tokenizer.tokenize(statement)
        if len(statement) > 100:
            statement = statement[:100]
        word_pieces += statement + ["[SEP]"]
        len_st = len(word_pieces)

        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)

        segments_tensor = torch.tensor([0] * len_st, dtype=torch.long)

        image = Image.open(self.img_path + img)
        # 将所有图片转换为 RGB
        image = image.convert("RGB")
        image_tensor = self.pad_image(image)  # 确保使用统一的填充

        return (tokens_tensor, segments_tensor, image_tensor, label_tensor)

    def __len__(self):
        return self.len

    def pad_image(self, image, target_size=(224, 224)):
        """
        对图片进行零填充，使其大小一致。
        :param image: 输入图片，PIL.Image 对象
        :param target_size: 目标大小 (height, width)
        :return: 填充后的张量
        """
        # 使用 Resampling.LANCZOS 重采样并调整大小
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        return transforms.ToTensor()(image)


def create_mini_batch(samples):
    """
    处理一个批次的数据。
    :param samples: 批次数据
    :return: (tokens_tensors, segments_tensors, masks_tensors, imgs_tensors, label_ids)
    """
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    imgs_tensors = [s[2] for s in samples]

    # 动态计算最大高度和宽度
    max_h = max(img.shape[1] for img in imgs_tensors)  # 高度
    max_w = max(img.shape[2] for img in imgs_tensors)  # 宽度

    # 对每张图片进行填充
    imgs_tensors = [
        F.pad(img, (0, max_w - img.shape[2], 0, max_h - img.shape[1]))
        for img in imgs_tensors
    ]
    imgs_tensors = torch.stack(imgs_tensors)

    if samples[0][3] is not None:
        label_ids = torch.stack([s[3] for s in samples])
    else:
        label_ids = None

    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, batch_first=True)

    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)

    return tokens_tensors, segments_tensors, masks_tensors, imgs_tensors, label_ids

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


# TODO Start: Inherit from torch.utils.data.Dataset #
class LandScapeDataset(Dataset):
    # TODO End #
    def __init__(self, mode="train"):
        self.mode = mode
        if mode == "train" or mode == "val":
            with open(f"./data/{mode}/file.txt", 'r') as file:
                entries = file.read().strip().split('\n')[1:]
                table = [entry.split(",") for entry in entries]
                self.images = [row[0] for row in table]
                self.gt = [(eval(row[1]), eval(row[2]), eval(row[3])) for row in table]
        elif mode == "test":
            with open(f"./data/{mode}/file.txt", 'r') as file:
                self.images = file.read().strip().split('\n')[1:]

    def __len__(self):
        # TODO Start: Return length of current dataset #
        return len(self.images)
        # TODO End #

    def __getitem__(self, idx):
        """
        :param idx: index to get.
        :return: ret_dict: {"image": torch.Tensor (3, 192, 256), "label": torch.Tensor (3, )}
        """

        file_name = self.images[idx]

        # TODO Start: Use Image from PIL to load image, then resize it to (w/4, h/4) #
        image = Image.open(f"./data/{self.mode}/imgs/{file_name}")
        image = image.resize((image.width // 4, image.height // 4))  # Resize to (w/4, h/4)
        # TODO End #

        array = np.array(image)
        # TODO Start: What is this line doing? #
        array = array.transpose((2, 0, 1))  # From (192, 256, 3) to (3, 192, 256)
        # 当处理图像数据时，通常会使用 (height, width, channels) 的顺序
        # 但某些模型或操作可能需要 (channels, height, width) 的顺序
        # TODO End #

        ret_dict = {
            "image": torch.tensor(array),
        }

        if self.mode != "test":
            ret_dict["label"] = torch.tensor(np.array(self.gt[idx]))

        # Normalize ret_dict["image"] from [0, 255] to [-1, 1]
        ret_dict["image"] = (ret_dict["image"] / 255) * 2 - 1

        return ret_dict

import math
import numpy as np
import torch
import cv2
import random

from ..builder import PIPELINES

@PIPELINES.register_module()
class TMemoryBank(object):
    def __init__(self, num_classes, small_index,  memory_size = 5, p_sample = 1, p_resize = 0.1):
        self.num_classes = num_classes
        self.small_index = small_index
        self.memory_size = memory_size
        self.p_sample = p_sample
        self.p_resize = p_resize

        if self.memory_size != 0:
            self.class_object_store_img = {}
            self.class_object_store_target = {}
            for i in range(self.num_classes):
                self.class_object_store_img[i] = {}
                self.class_object_store_target[i] = {}

    def _add_image_to_image(self, img, img2, gt, gt2):
        _, h_img, w_img = img.size()
        _, h_img2, w_img2 = img2.size()

        x_offset = random.randint(0, w_img - w_img2)
        y_offset = random.randint(0, h_img - h_img2)

        mask = gt2 == -1
        unmask = ~mask

        img2 = torch.mul(img2, unmask) + torch.mul(img[:, y_offset:y_offset + h_img2, x_offset:x_offset + w_img2], mask)
        gt2 = torch.mul(gt2, unmask) + torch.mul(gt[:, y_offset:y_offset + h_img2, x_offset:x_offset + w_img2], mask)

        img[:, y_offset:y_offset + h_img2, x_offset:x_offset + w_img2] = img2
        gt[:, y_offset:y_offset + h_img2, x_offset:x_offset + w_img2] = gt2

        return img, gt

    def _resize_image_and_gt(self, image, gt, p_resize):
        _, h, w = image.size()
        h_scale = int(h * p_resize)
        w_scale = int(w * p_resize)

        resized_image = image.numpy().transpose((1, 2, 0))
        resized_gt = np.squeeze(gt.numpy())

        resized_image = cv2.resize(resized_image, (w_scale, h_scale), interpolation=1)
        resized_gt = cv2.resize(resized_gt, (w_scale, h_scale), interpolation=0)

        resized_image = resized_image.transpose((2, 0, 1))
        resized_gt = np.expand_dims(resized_gt, axis=0)

        resized_image = torch.tensor(resized_image, device=image.device)
        resized_gt = torch.tensor(resized_gt, device=gt.device)
        return resized_image, resized_gt

    def _transform(self, image, gt, target):
        _, h, w = image.size()
        new_gt = torch.full((1, h, w), -1, dtype=torch.int32, device=image.device)
        for i in range(len(gt[0])):
            new_gt[int(gt[0][i]), int(gt[1][i]), int(gt[2][i])] = target
        return image, new_gt

    def __call__(self, results):
        unique_class_index = torch.unique(results['gt_semantic_seg'].data)
        unique_class_index = unique_class_index.tolist()
        unique_class_index = [value for value in unique_class_index if value != 255]

        unique_small_index = list(set(unique_class_index).intersection(set(self.small_index)))
        unique_not_small_index = [x for x in self.small_index if x not in unique_class_index]

        if len(unique_small_index) != 0:
            for i in unique_small_index:
                class_i_index = torch.where(results['gt_semantic_seg'].data == i)
                if len(self.class_object_store_target[i]) == self.memory_size:
                    del_num = random.randint(0, self.memory_size - 1)
                    self.class_object_store_target[i][del_num] = class_i_index
                    self.class_object_store_img[i][del_num] = results['img'].data
                else:
                    self.class_object_store_img[i][len(self.class_object_store_target[i])] = results['img'].data
                    self.class_object_store_target[i][len(self.class_object_store_target[i])] = class_i_index
        if len(unique_not_small_index) != 0:
            random_index = random.sample(range(len(unique_not_small_index)), math.ceil(len(unique_not_small_index) * self.p_sample))
            random_index = [unique_not_small_index[i] for i in random_index]
            for i in random_index:
                if len(self.class_object_store_target[i]) == 0:
                    continue
                else:
                    add_num = random.randint(0, len(self.class_object_store_target[i]) - 1)
                    temp_img = self.class_object_store_img[i][add_num]
                    temp_gt = self.class_object_store_target[i][add_num]

                    temp_img, temp_gt = self._transform(temp_img, temp_gt, i)
                    temp_img, temp_gt = self._resize_image_and_gt(temp_img, temp_gt, self.p_resize)
                    temp_img, temp_gt = self._add_image_to_image(results['img'].data, temp_img, results['gt_semantic_seg'].data, temp_gt)

                    results['img'].data[:,:,:] = temp_img
                    results['gt_semantic_seg'].data[:,:,:] = temp_gt
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(num_classes={self.num_classes}, ' \
                    f'memory_size={self.memory_size})'
        return repr_str
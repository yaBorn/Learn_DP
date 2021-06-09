"""

"""
import math
import os

from PIL import Image
import h5py
import numpy as np
import torch
import torch.utils.data as data

np.set_printoptions(threshold=np.inf)


class Tools:
    def __init__(self):
        pass

    @staticmethod
    def psnr(img1, img2):
        """
        compute the psnr
        :param img1: img1
        :param img2: img2
        :return:
        """
        diff = img1 - img2
        diff = diff.flatten('C')
        rmse = math.sqrt(np.mean(diff ** 2.))
        psnr = 20 * math.log10(1.0 / rmse)
        return psnr

    @staticmethod
    def convert_rgb_to_ycbcr(img):
        if type(img) == np.ndarray:
            y = 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
            cb = 128. + (-37.945 * img[:, :, 0] - 74.494 * img[:, :, 1] + 112.439 * img[:, :, 2]) / 256.
            cr = 128. + (112.439 * img[:, :, 0] - 94.154 * img[:, :, 1] - 18.285 * img[:, :, 2]) / 256.
            return np.array([y, cb, cr]).transpose([1, 2, 0])
        elif type(img) == torch.Tensor:
            if len(img.shape) == 4:
                img = img.squeeze(0)
            y = 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
            cb = 128. + (-37.945 * img[0, :, :] - 74.494 * img[1, :, :] + 112.439 * img[2, :, :]) / 256.
            cr = 128. + (112.439 * img[0, :, :] - 94.154 * img[1, :, :] - 18.285 * img[2, :, :]) / 256.
            return torch.cat([y, cb, cr], 0).permute(1, 2, 0)
        else:
            raise Exception('Unknown Type', type(img))

    @staticmethod
    def convert_ycbcr_to_rgb(img):
        if type(img) == np.ndarray:
            r = 298.082 * img[:, :, 0] / 256. + 408.583 * img[:, :, 2] / 256. - 222.921
            g = 298.082 * img[:, :, 0] / 256. - 100.291 * img[:, :, 1] / 256. - 208.120 * img[:, :, 2] / 256. + 135.576
            b = 298.082 * img[:, :, 0] / 256. + 516.412 * img[:, :, 1] / 256. - 276.836
            return np.array([r, g, b]).transpose([1, 2, 0])
        elif type(img) == torch.Tensor:
            if len(img.shape) == 4:
                img = img.squeeze(0)
            r = 298.082 * img[0, :, :] / 256. + 408.583 * img[2, :, :] / 256. - 222.921
            g = 298.082 * img[0, :, :] / 256. - 100.291 * img[1, :, :] / 256. - 208.120 * img[2, :, :] / 256. + 135.576
            b = 298.082 * img[0, :, :] / 256. + 516.412 * img[1, :, :] / 256. - 276.836
            return torch.cat([r, g, b], 0).permute(1, 2, 0)
        else:
            raise Exception('Unknown Type', type(img))

    def make_train_h5(self, training_root, save_path, input_size=33, label_size=21, scale_factor=3, flag=0):
        '''
        make training data(h5 file)
        :param training_root: the dir of traning dataset
        :param save_path: name of ht file
        :param input_size: the input img size for training (set to be 33*33, default)
        :param label_size: the label size for training (set to be 21*21, default)
        :param scale_factor: (set to be 3, default)
        :param flag: training mode,0 is ycrcb's y channel; else is rgb channel
        :return:
        '''
        stride = 14
        padding = (input_size - label_size) // 2

        data = []
        label = []

        for (root, dir, files) in os.walk(training_root):
            for file in files:
                filepath = root + '/' + file
                image = Image.open(filepath).convert('RGB')
                if flag == 0:
                    image = np.array(image)
                    image = Tools.convert_rgb_to_ycbcr(image).astype(np.uint8)
                    image = np.clip(image, a_min=0, a_max=255)
                else:
                    image = np.array(image)
                image = image[:, :, 0:3]
                im_label = self.__modcrop(image, scale_factor)
                (hei, wid, channel) = im_label.shape
                # scale to 1 / s
                im_label = Image.fromarray(im_label)
                im_input = im_label.resize((wid // scale_factor, hei // scale_factor), resample=Image.BICUBIC)
                # scale to s
                im_input = im_input.resize((im_input.width * scale_factor, im_input.height * scale_factor), resample=Image.BICUBIC)

                # low resolution for input
                im_input = np.array(im_input)
                im_input = im_input.astype('float32')
                # high resolution for label
                im_label = np.array(im_label)
                im_label = im_label.astype('float32')

                if flag == 0:
                    for x in range(0, hei - input_size + 1, stride):
                        for y in range(0, wid - input_size + 1, stride):
                            sub_im_input = im_input[x:x + input_size, y:y + input_size, :]
                            sub_im_label = im_label[x + padding:x + padding + label_size, y + padding: y + padding + label_size, :]
                            sub_im_input = sub_im_input.reshape([input_size, input_size, 3])
                            sub_im_label = sub_im_label.reshape([label_size, label_size, 3])
                            sub_im_input_y = sub_im_input[:, :, 0]
                            sub_im_label_y = sub_im_label[:, :, 0]
                            data.append(sub_im_input_y)
                            label.append(sub_im_label_y)
                else:
                    for x in range(0, hei - input_size + 1, stride):
                        for y in range(0, wid - input_size + 1, stride):
                            sub_im_input = im_input[x:x + input_size, y:y + input_size, :]
                            sub_im_label = im_label[x + padding:x + padding + label_size, y + padding: y + padding + label_size, :]
                            sub_im_input = sub_im_input.reshape([input_size, input_size, 3])
                            sub_im_label = sub_im_label.reshape([label_size, label_size, 3])
                            data.append(sub_im_input)
                            label.append(sub_im_label)
        data = np.asarray(data)
        label = np.asarray(label)
        with h5py.File(save_path, 'w') as hf:
            hf.create_dataset('input', data=data)
            hf.create_dataset('label', data=label)

    def make_eval_h5(self, training_root, save_path, scale_factor=3, flag=0):
        h5_file = h5py.File(save_path, 'w')
        i = 0
        lr_group = h5_file.create_group('input')
        hr_group = h5_file.create_group('label')
        for (root, dir, files) in os.walk(training_root):
            for file in files:
                filepath = root + '/' + file
                image = Image.open(filepath).convert('RGB')
                if flag == 0:
                    image = np.array(image)
                    image = Tools.convert_rgb_to_ycbcr(image).astype(np.uint8)
                    image = np.clip(image, a_min=0, a_max=255)
                else:
                    image = np.array(image)
                # 有些图片是带alpha通道的4通道矩阵，这里我们只用3个通道，故用0:3
                image = image[:, :, 0:3]
                im_label = self.__modcrop(image, scale_factor)
                (hei, wid, channel) = im_label.shape
                # scale to 1 / s
                im_label = Image.fromarray(im_label)
                im_input = im_label.resize((wid // scale_factor, hei // scale_factor), resample=Image.BICUBIC)
                # scale to s
                im_input = im_input.resize((im_input.width * scale_factor, im_input.height * scale_factor), resample=Image.BICUBIC)
                im_input = np.array(im_input)
                im_label = np.array(im_label)
                if flag == 0:
                    im_input = im_input[:, :, 0]
                    im_label = im_label[:, :, 0]
                lr_group.create_dataset(str(i), data=im_input)
                hr_group.create_dataset(str(i), data=im_label)
                i += 1
        h5_file.close()

    def __modcrop(self, imgs, modulo):
        '''
        crop the image to make the H and W be integer multiples of 3
        :param imgs:
        :param modulo:
        :return:
        '''
        if np.size(imgs.shape) == 3:
            (sheight, swidth, _) = imgs.shape
            sheight = sheight - np.mod(sheight, modulo)
            swidth = swidth - np.mod(swidth, modulo)
            imgs = imgs[0:sheight, 0:swidth, :]

        else:
            (sheight, swidth) = imgs.shape
            sheight = sheight - np.mod(sheight, modulo)
            swidth = swidth - np.mod(swidth, modulo)
            imgs = imgs[0:sheight, 0:swidth]

        return imgs


class DataFromH5File(data.Dataset):
    def __init__(self, filepath, status='train'):
        h5file = h5py.File(filepath, 'r')
        self.status = status
        if self.status == 'train':
            self.label = h5file['label']
            self.input = h5file['input']
        else:
            self.h5_file = h5file

    def __getitem__(self, idx):
        if self.status == 'train':
            train_label = torch.from_numpy(self.label[idx]).float()
            train_input = torch.from_numpy(self.input[idx]).float()
        else:
            f1 = np.asarray(self.h5_file['label'][str(idx)])
            f2 = np.asarray(self.h5_file['input'][str(idx)])
            train_label = torch.from_numpy(f1).float()
            train_input = torch.from_numpy(f2).float()
        return train_input, train_label

    def __len__(self):
        if self.status == 'train':
            assert self.label.shape[0] == self.input.shape[0], "Wrong data length"
            return self.input.shape[0]
        else:
            count = len(self.h5_file['input'])
            return count
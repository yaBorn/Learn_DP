"""
    SRCNN 测试
"""
import os

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from pylab import *
import pytorch_ssim
from SRCNNStructure import SRCNN
from utils import Tools


def modcrop(imgs, modulo):
    '''
    因为双三次线性插值要对图像进行缩放，本函数将图像大小变为缩放的整数倍
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


'''
    遇到的问题，conv1:卷积核size为9，步长为1;conv2:卷积核size为1，步长为1;conv3:卷积核size为5，步长为1
    直接测试时，对于一个33×33的图像，经过conv1后尺寸为25×25，经过conv2后尺寸为25×25，经过conv3后尺寸为21×21
    相当于经过SRCNN后输出图像的高与宽均分别少了12。这样，在计算PSNR时比较的图像尺寸就不一致。而Pytorch的模型一旦保存
    后，模型结构里的属性不能修改，因此我需要人为的给测试图像加Padding，使输入输出的图像尺寸保持一致。
'''
if __name__ == '__main__':
    flag = 0  # flag用来分测试模式，为0时，只训练Y通道；非0时训练RGB三通道
    dirPath = os.path.abspath('.')
    testFilePath = dirPath + '\Test\Set5\\butterfly_GT.bmp'
    if flag == 0:
        dicname = '\\SRCNN_set5_{}_best.pt'.format('y')
    else:
        dicname = '\\SRCNN_set5_{}_best.pt'.format('rgb')
    SRCNN_dict = dirPath + dicname
    scale = 3   # 放大倍数
    img = Image.open(testFilePath).convert('RGB')
    img1 = np.array(img)
    if flag == 0:
        SRCNN = SRCNN().cuda()
        img1 = Tools.convert_rgb_to_ycbcr(img1).astype(np.uint8)
    else:
        SRCNN = SRCNN(channels_number=3).cuda()
    SRCNN.load_state_dict(torch.load(SRCNN_dict))
    img2 = modcrop(img1, 3)
    img2 = Image.fromarray(img2)
    img3 = img2.resize((img2.width//scale, img2.height//scale), resample=Image.BICUBIC)
    img4 = img3.resize((img3.width * scale, img3.height * scale), resample=Image.BICUBIC)
    img4 = np.array(img4)
    img2 = np.array(img2)
    img3 = np.array(img3)
    if flag == 0:
        img5 = img4[:, :, 0]
        img5 = np.expand_dims(img5, 2)
        img_cb = img4[:, :, 1]
        img_cr = img4[:, :, 2]
    else:
        img5 = img4
    # 导入训练测试
    img_t = torch.from_numpy(img5.astype('float32')).cuda() / 255
    img_t = img_t.permute(2, 0, 1)
    img_t = img_t.unsqueeze(0)
    with torch.no_grad():
        # conv1卷积
        img_t = F.pad(img_t, (6, 6, 6, 6))
        img_H = F.relu(SRCNN.conv1(img_t))
        # conv2卷积
        img_H = F.relu(SRCNN.conv2(img_H))
        # conv3卷积
        img_H = SRCNN.conv3(img_H)
    img_h = torch.clamp(img_H.cpu(), min=0, max=1)
    if flag == 0:
        img_y = torch.squeeze(img_h).numpy()
        img_y = (img_y * 255).astype(np.uint8)
        img6 = np.array([img_y, img_cb, img_cr])
        img6 = Tools.convert_ycbcr_to_rgb(img6.transpose([1, 2, 0]))/255
        img2 = Tools.convert_ycbcr_to_rgb(img2).astype(np.uint8)
        img3 = Tools.convert_ycbcr_to_rgb(img3).astype(np.uint8)
        img4 = Tools.convert_ycbcr_to_rgb(img4).astype(np.uint8)
        img6 = np.clip(img6, a_min=0.0, a_max=1.0)
        img2 = np.clip(img2, a_min=0, a_max=255)
        img3 = np.clip(img3, a_min=0, a_max=255)
        img4 = np.clip(img4, a_min=0, a_max=255)
    else:
        img6 = torch.squeeze(img_h).numpy()
        img6 = img6.transpose(1, 2, 0)
    # 计算下PSNR（峰值信噪比）,PSNR越大越好
    psnr1 = Tools.psnr(img6, img2/255)
    psnr2 = Tools.psnr(img4/255, img2/255)
    ss6 = torch.from_numpy(img6).permute(2, 0, 1).unsqueeze(0).float()
    ss2 = torch.from_numpy(img2 / 255).permute(2, 0, 1).unsqueeze(0).float()
    ss4 = torch.from_numpy(img4 / 255).permute(2, 0, 1).unsqueeze(0).float()
    ssim1 = pytorch_ssim.ssim(ss6, ss2, window_size=10).item()
    ssim2 = pytorch_ssim.ssim(ss4, ss2, window_size=10).item()
    print("trainingResult psnr:%.4f dB, ssim: %.4f" % (psnr1, ssim1))
    print("bicubic2 psnr:%.4f dB, ssim: %.4f" % (psnr2, ssim2))
    # 画图对比
    trainPicture = figure()
    ax1 = trainPicture.add_subplot(221)
    title('origin')
    ax2 = trainPicture.add_subplot(222)
    title('bicubic1')
    ax3 = trainPicture.add_subplot(223)
    title('bicubic2')
    ax4 = trainPicture.add_subplot(224)
    title('trainingResult')
    ax1.imshow(img2)
    ax2.imshow(img3)
    ax3.imshow(img4)
    ax4.imshow(img6)
    trainPicture.show()

"""
    SRCNN 训练
"""
import copy
import os

import pytorch_ssim
import torch
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm

import SRCNNStructure as Net
from utils import Tools, DataFromH5File


# 训练函数，返回本次训练的loss
def train(img_L, originimg_H, model, device, optimizer, loss_fn, flag):
    '''
    1、将numpy形式的img_L变为torch,且由于是openCV读取，其格式为height*width*channel,不符合torch传
    入数据为channel*height*width的格式，故需要在变成torch前将其转置一下；
    2、img_lt为插值后低分辨率tensor，3×33×33，originimg_ht为原始高分辨率图片tensor,3×21×21；
    :param img_L: 经过bicubic插值图片
    :param originimg_H: 原始图片
    :param model: SRCNN网络
    :param device: 选用GPU或者CPU
    :param optimizer: SGD
    :param loss_fn: MSE Loss
    :return: 每次训练的loss数值
    '''
    if flag == 0:
        img_L = img_L.unsqueeze(3)
        originimg_H = originimg_H.unsqueeze(3)
    img_L = (img_L / 255).permute(0, 3, 1, 2)
    originimg_H = (originimg_H / 255).permute(0, 3, 1, 2)
    img_L = img_L.to(device)
    originimg_H = originimg_H.to(device)
    model.to(device)
    loss_fn.to(device)
    # 训练
    pred = model(img_L)
    loss = loss_fn(pred, originimg_H)
    # SGD
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def valid(valinput, vallabel, model, device, flag):
    model.to(device)
    if flag == 0:
        valinput = valinput.unsqueeze(3)
        vallabel = vallabel.unsqueeze(3)
    valinput = (valinput / 255).permute(0, 3, 1, 2).to(device)
    vallabel = (vallabel / 255).permute(0, 3, 1, 2)
    # 经过SRCNN后，图片长宽会比原来各小12，所以要加一个大小为6的padding
    img = F.pad(valinput, (6, 6, 6, 6))
    img = F.relu(model.conv1(img))
    img = F.relu(model.conv2(img))
    pred = model.conv3(img)
    sr = torch.clamp(pred.cpu(), min=0, max=1)
    ssim = pytorch_ssim.ssim(sr.cpu(), vallabel.cpu(), window_size=10).item()
    np_pred = torch.squeeze(sr).detach().numpy()
    np_vallabel = torch.squeeze(vallabel).numpy()
    psnr = Tools.psnr(np_pred, np_vallabel)
    return psnr, ssim


if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    flag = 0  # flag用来控制训练模式，为0时，只训练Y通道；非0时训练RGB三通道
    # trainSetPath:E:\design\pythonworkspace\SRCNN\Train\
    trainSetPath = os.path.abspath('.') + '\Train\\'
    evalSetPath = os.path.abspath('.') + '\Test\\Set5\\'
    '''
    一、数据预处理，将训练图片变成HDF5数据集，需要经历几步:
        1、对传入训练的图片进行去边处理
        2、scale_factor的默认值为3，进行scale_factor分之一的缩小，再按scale_factor倍进行双三次线性插值放大
        3、将图片进行分割裁剪，变成33×33的小块，压入h5数据集
    '''
    if flag == 0:
        train_outputFile = 'training_91_image_patches(y).h5'
        eval_outputFile = 'set5_image_patches(y).h5'
        model = Net.SRCNN()
    else:
        train_outputFile = 'training_91_image_patches(rgb).h5'
        eval_outputFile = 'set5_image_patches(rgb).h5'
        model = Net.SRCNN(channels_number=3)
    if not os.path.isfile(train_outputFile):
        tools = Tools()
        Tools.make_train_h5(tools, trainSetPath, train_outputFile, flag=flag)
    if not os.path.isfile(eval_outputFile):
        tools = Tools()
        Tools.make_eval_h5(tools, evalSetPath, eval_outputFile, flag=flag)
    # 导入h5文件做数据集
    trainset = DataFromH5File(train_outputFile)
    evalset = DataFromH5File(eval_outputFile, status='eval')
    train_loader = data.DataLoader(dataset=trainset, batch_size=16, shuffle=True, pin_memory=True)
    eval_loader = data.DataLoader(dataset=evalset, batch_size=1)
    # 二、设置训练参数，准备训练

    num_epochs = 3
    lr = 1e-5
    conv3_params = list(map(id, model.conv3.parameters()))
    base_params = filter(lambda p: id(p) not in conv3_params, model.parameters())
    optimizer = torch.optim.Adam([
        {'params': base_params},
        {'params': model.conv3.parameters(), 'lr': lr}
    ], lr * 10)
    loss_fn = torch.nn.MSELoss()

    best_weight = copy.deepcopy(model.state_dict)
    bestepoch = 1  # 测试结果最好的一次
    best_psnr = 0  # 测试结果最好的psnr
    best_ssim = 0  # 测试结果最好的psnr

    for epoch in range(num_epochs):
        # 进行训练
        batchnum = 0  # 训练batch数目
        validnum = 0  # 验证集图片个数
        loss_epo = 0  # 每次迭代损失
        psnr = 0  # 每次迭代验证集psnr
        ssim = 0  # 每次迭代验证集ssim
        train_bar = tqdm(train_loader)
        model.train()
        for train_data, train_label in train_bar:
            loss_bat = train(train_data, train_label, model, device, optimizer, loss_fn, flag)
            loss_epo += loss_bat
            batchnum += 1
            train_bar.set_description(desc='flag%d---Train[%d/%d] Loss:%.5f' % (flag, (epoch + 1), num_epochs, loss_epo / batchnum))
        model.eval()
        with torch.no_grad():
            eval_bar = tqdm(eval_loader)
            for data in eval_bar:
                eval_data, eval_label = data
                psnr_per, ssim_per = valid(eval_data, eval_label, model, device, flag)
                psnr += psnr_per
                ssim += ssim_per
                validnum += 1
                eval_bar.set_description('psnr: %.5f dB, ssim: %.5f' % (psnr / validnum, ssim / validnum))
            if (psnr / validnum) > best_psnr and (ssim / validnum) > best_ssim:
                best_epoch = (epoch + 1)
                best_psnr = (psnr / validnum)
                best_ssim = (ssim / validnum)
                best_weight = copy.deepcopy(model.state_dict())
    # 训练结束，保存训练模型
    if flag == 0:
        torch.save(best_weight, "SRCNN_set5_y_best.pt")
    else:
        torch.save(best_weight, "SRCNN_set5_rgb_best.pt")
    print("模型保存成功！最佳数据——epoch{}, psnr: {:.5f}dB, ssim: {:.5f}".format(best_epoch, best_psnr, best_ssim))

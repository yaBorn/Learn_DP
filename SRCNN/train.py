"""
    SRCNN ѵ��
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


# ѵ�����������ر���ѵ����loss
def train(img_L, originimg_H, model, device, optimizer, loss_fn, flag):
    '''
    1����numpy��ʽ��img_L��Ϊtorch,��������openCV��ȡ�����ʽΪheight*width*channel,������torch��
    ������Ϊchannel*height*width�ĸ�ʽ������Ҫ�ڱ��torchǰ����ת��һ�£�
    2��img_ltΪ��ֵ��ͷֱ���tensor��3��33��33��originimg_htΪԭʼ�߷ֱ���ͼƬtensor,3��21��21��
    :param img_L: ����bicubic��ֵͼƬ
    :param originimg_H: ԭʼͼƬ
    :param model: SRCNN����
    :param device: ѡ��GPU����CPU
    :param optimizer: SGD
    :param loss_fn: MSE Loss
    :return: ÿ��ѵ����loss��ֵ
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
    # ѵ��
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
    # ����SRCNN��ͼƬ������ԭ����С12������Ҫ��һ����СΪ6��padding
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
    flag = 0  # flag��������ѵ��ģʽ��Ϊ0ʱ��ֻѵ��Yͨ������0ʱѵ��RGB��ͨ��
    # trainSetPath:E:\design\pythonworkspace\SRCNN\Train\
    trainSetPath = os.path.abspath('.') + '\Train\\'
    evalSetPath = os.path.abspath('.') + '\Test\\Set5\\'
    '''
    һ������Ԥ������ѵ��ͼƬ���HDF5���ݼ�����Ҫ��������:
        1���Դ���ѵ����ͼƬ����ȥ�ߴ���
        2��scale_factor��Ĭ��ֵΪ3������scale_factor��֮һ����С���ٰ�scale_factor������˫�������Բ�ֵ�Ŵ�
        3����ͼƬ���зָ�ü������33��33��С�飬ѹ��h5���ݼ�
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
    # ����h5�ļ������ݼ�
    trainset = DataFromH5File(train_outputFile)
    evalset = DataFromH5File(eval_outputFile, status='eval')
    train_loader = data.DataLoader(dataset=trainset, batch_size=16, shuffle=True, pin_memory=True)
    eval_loader = data.DataLoader(dataset=evalset, batch_size=1)
    # ��������ѵ��������׼��ѵ��

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
    bestepoch = 1  # ���Խ����õ�һ��
    best_psnr = 0  # ���Խ����õ�psnr
    best_ssim = 0  # ���Խ����õ�psnr

    for epoch in range(num_epochs):
        # ����ѵ��
        batchnum = 0  # ѵ��batch��Ŀ
        validnum = 0  # ��֤��ͼƬ����
        loss_epo = 0  # ÿ�ε�����ʧ
        psnr = 0  # ÿ�ε�����֤��psnr
        ssim = 0  # ÿ�ε�����֤��ssim
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
    # ѵ������������ѵ��ģ��
    if flag == 0:
        torch.save(best_weight, "SRCNN_set5_y_best.pt")
    else:
        torch.save(best_weight, "SRCNN_set5_rgb_best.pt")
    print("ģ�ͱ���ɹ���������ݡ���epoch{}, psnr: {:.5f}dB, ssim: {:.5f}".format(best_epoch, best_psnr, best_ssim))

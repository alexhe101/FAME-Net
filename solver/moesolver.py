#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-13 23:04:48
LastEditTime: 2020-12-03 22:02:20
@Description: file content
'''
import math
import cv2
from torchvision.utils import make_grid
import numpy as np

def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result

import os, importlib, torch, shutil
from solver.basesolver import BaseSolver
from utils.utils import maek_optimizer, make_loss, calculate_psnr, calculate_ssim, save_config, save_net_config,qnr,D_s,D_lambda,cpsnr,cssim,no_ref_evaluate
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import numpy as np
from importlib import import_module
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
from tensorboardX import SummaryWriter
from utils.config import save_yml
import torch.nn.functional as F
from torchvision import transforms
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
class CVLoss(nn.Module):
    def __init__(self, loss_weight=1.0,reduction='mean'):
        super(CVLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
    def forward(self,logits):
        # print(torch.mean(logits,dim=1).shape)
        cv = torch.std(logits,dim=1)/torch.mean(logits,dim=1)
        # print(cv)
        return self.loss_weight*torch.mean(cv**2)
class Solver(BaseSolver):
    def __init__(self, cfg):
        super(Solver, self).__init__(cfg)
        self.init_epoch = self.cfg['schedule']
        
        net_name = self.cfg['algorithm'].lower()
        lib = importlib.import_module('model.' + net_name)
        net = lib.Net

        assert (self.cfg['data']['n_colors']==4)
        self.model = net(
            num_channels=self.cfg['data']['n_colors'], 
            base_filter=32,
            args = self.cfg
        )
        self.optimizer = maek_optimizer(self.cfg['schedule']['optimizer'], cfg, self.model.parameters())
        self.loss = make_loss(self.cfg['schedule']['loss'])
        self.gate_loss = CVLoss()
        self.mask_loss = make_loss(self.cfg['schedule']['loss'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,1000,5e-8)
        self.log_name = self.cfg['algorithm'] + '_' + str(self.cfg['data']['upsacle']) + '_' + str(self.timestamp)
        # save log
        self.writer = SummaryWriter(self.cfg['log_dir']+ str(self.log_name))
        save_net_config(self.log_name, self.model)
        save_yml(cfg, os.path.join(self.cfg['log_dir'] + str(self.log_name), 'config.yml'))
        save_config(self.log_name, 'Train dataset has {} images and {} batches.'.format(len(self.train_dataset), len(self.train_loader)))
        save_config(self.log_name, 'Val dataset has {} images and {} batches.'.format(len(self.val_dataset), len(self.val_loader)))
        save_config(self.log_name, 'Model parameters: '+ str(sum(param.numel() for param in self.model.parameters())))

    def train(self): 
        with tqdm(total=len(self.train_loader), miniters=1,
                desc='Initial Training Epoch: [{}/{}]'.format(self.epoch, self.nEpochs)) as t:
            idx = math.pow(0.99, self.epoch - 1)
            para = 1* (idx)
            gate_cof = 1
            epoch_loss = 0
            for iteration, batch in enumerate(self.train_loader, 1):
                ms_image, lms_image, pan_image, mask_gt, file = Variable(batch[0]), Variable(batch[1]), Variable(
                    batch[2]), Variable(batch[3]), (batch[4])

                if self.cuda:
                    ms_image, lms_image, pan_image, mask_gt = ms_image.cuda(self.gpu_ids[0]), lms_image.cuda(
                        self.gpu_ids[0]), pan_image.cuda(self.gpu_ids[0]), mask_gt.cuda(self.gpu_ids[0])
                self.optimizer.zero_grad()
                self.model.train()
                y,mask,lf_gate,hf_gate,dec_gate = self.model(lms_image, lms_image, pan_image)
                total_lowgate_loss = self.gate_loss(lf_gate)
                total_highgate_loss = self.gate_loss(hf_gate)
                total_decodergate_loss = self.gate_loss(dec_gate)
                mask_loss = self.mask_loss(mask_gt,mask)
                gl = total_lowgate_loss+total_highgate_loss+total_decodergate_loss
                loss = (self.loss(y, ms_image)+para*mask_loss+gl)
                if self.cfg['schedule']['use_YCbCr']:
                    y_vgg = torch.unsqueeze(y[:,3,:,:], 1)
                    y_vgg_3 = torch.cat([y_vgg, y_vgg, y_vgg], 1)
                    pan_image_3 = torch.cat([pan_image, pan_image, pan_image], 1)
                    vgg_loss = self.vggloss(y_vgg_3, pan_image_3)
                epoch_loss += loss.data
                t.set_postfix_str("Batch loss {:.4f}".format(loss.item()))
                t.update()

                loss.backward()
                if self.cfg['schedule']['gclip'] > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg['schedule']['gclip']
                    )
                self.optimizer.step()
            torch.cuda.empty_cache() 
            self.scheduler.step()
            self.records['Loss'].append(epoch_loss / len(self.train_loader))
            self.writer.add_image('image1', ms_image[0], self.epoch)
            self.writer.add_image('image2', y[0], self.epoch)
            self.writer.add_image('image3', pan_image[0], self.epoch)
            save_config(self.log_name, 'Initial Training Epoch {}: Loss={:.4f}'.format(self.epoch, self.records['Loss'][-1]))
            self.writer.add_scalar('Loss_epoch', self.records['Loss'][-1], self.epoch)
    def eval(self):
        with tqdm(total=len(self.test_loader), miniters=1,
                desc='Val Epoch: [{}/{}]'.format(self.epoch, self.nEpochs)) as t1:
            psnr_list, ssim_list,qnr_list,d_lambda_list,d_s_list = [], [],[],[],[]

            for iteration, batch in enumerate(self.test_loader, 1):
                ms_image, lms_image, pan_image, bms_image, file = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3]), (batch[4])
                if self.cuda:
                    ms_image, lms_image, pan_image, bms_image = ms_image.cuda(self.gpu_ids[0]), lms_image.cuda(self.gpu_ids[0]), pan_image.cuda(self.gpu_ids[0]), bms_image.cuda(self.gpu_ids[0])

                self.model.eval()
                with torch.no_grad():
                    y,mask,lf_gate,hf_gate,dec_gate = self.model(lms_image, bms_image, pan_image)

                    loss = self.loss(y, ms_image)

                batch_psnr, batch_ssim,batch_qnr,batch_D_lambda,batch_D_s = [], [],[],[],[]
                fake_img = y[:,:,:,:]
                for c in range(y.shape[0]):
                    if not self.cfg['data']['normalize']:
                        predict_y = (y[c, ...].cpu().numpy().transpose((1, 2, 0))) * 255
                        ground_truth = (ms_image[c, ...].cpu().numpy().transpose((1, 2, 0))) * 255
                        pan = (pan_image[c, ...].cpu().numpy().transpose((1, 2, 0))) * 255
                        l_ms = (lms_image[c, ...].cpu().numpy().transpose((1, 2, 0))) * 255
                        f_img = (fake_img[c, ...].cpu().numpy().transpose((1, 2, 0))) * 255
                    else:          
                        predict_y = (y[c, ...].cpu().numpy().transpose((1, 2, 0)) + 1) * 127.5
                        ground_truth = (ms_image[c, ...].cpu().numpy().transpose((1, 2, 0)) + 1) * 127.5
                        pan = (pan_image[c, ...].cpu().numpy().transpose((1, 2, 0)) + 1) * 127.5
                        l_ms = (lms_image[c, ...].cpu().numpy().transpose((1, 2, 0)) + 1) * 127.5
                        f_img  =(fake_img[c, ...].cpu().numpy().transpose((1, 2, 0)) + 1) * 127.5
                    psnr = cpsnr(predict_y, ground_truth)
                    ssim = cssim(predict_y,ground_truth,255)
                    l_ms = np.uint8(l_ms)
                    pan = np.uint8(pan)
                    c_D_lambda, c_D_s, QNR = no_ref_evaluate(f_img,pan,l_ms)
                    batch_psnr.append(psnr)
                    batch_ssim.append(ssim)
                    batch_qnr.append(QNR)
                    batch_D_s.append(c_D_s)
                    batch_D_lambda.append(c_D_lambda)
                avg_psnr = np.array(batch_psnr).mean()
                avg_ssim = np.array(batch_ssim).mean()
                avg_qnr = np.array(batch_qnr).mean()
                avg_d_lambda = np.array(batch_D_lambda).mean()
                avg_d_s = np.array(batch_D_s).mean()
                psnr_list.extend(batch_psnr)
                ssim_list.extend(batch_ssim)
                qnr_list.extend(batch_qnr)
                d_s_list.extend(batch_D_s)
                d_lambda_list.extend(batch_D_lambda)
                t1.set_postfix_str('n:Batch loss: {:.4f}, PSNR: {:.4f}, SSIM: {:.4f},QNR:{:.4F} DS:{:.4f},D_L:{:.4F}'.format(loss.item(), avg_psnr, avg_ssim,avg_qnr,avg_d_s,avg_d_lambda))
                t1.update()
            self.records['Epoch'].append(self.epoch)
            self.records['PSNR'].append(np.array(psnr_list).mean())
            self.records['SSIM'].append(np.array(ssim_list).mean())
            self.records['QNR'].append(np.array(qnr_list).mean())
            self.records['D_lamda'].append(np.array(d_lambda_list).mean())
            self.records['D_s'].append(np.array(d_s_list).mean())
            save_config(self.log_name, 'Val Epoch {}: PSNR={:.4f}, SSIM={:.6f},QNR={:.4f}, DS:{:.4f},D_L:{:.4F}'.format(self.epoch, self.records['PSNR'][-1],
                                                                    self.records['SSIM'][-1],self.records['QNR'][-1],self.records['D_s'][-1],self.records['D_lamda'][-1]))
            self.writer.add_scalar('PSNR_epoch', self.records['PSNR'][-1], self.epoch)
            self.writer.add_scalar('SSIM_epoch', self.records['SSIM'][-1], self.epoch)
            self.writer.add_scalar('QNR_epoch', self.records['QNR'][-1], self.epoch)
            self.writer.add_scalar('D_s_epoch', self.records['D_s'][-1], self.epoch)
            self.writer.add_scalar('D_lamda_epoch', self.records['D_lamda'][-1], self.epoch)
    def check_gpu(self):
        self.cuda = self.cfg['gpu_mode']
        torch.manual_seed(self.cfg['seed'])
        if self.cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")
        if self.cuda:
            torch.cuda.manual_seed(self.cfg['seed'])
            cudnn.benchmark = True
              
            gups_list = self.cfg['gpus']
            self.gpu_ids = []
            for str_id in gups_list:
                gid = int(str_id)
                if gid >=0:
                    self.gpu_ids.append(gid)

            torch.cuda.set_device(self.gpu_ids[0]) 
            self.loss = self.loss.cuda(self.gpu_ids[0])
            self.model = self.model.cuda(self.gpu_ids[0])
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_ids) 

    def check_pretrained(self):
        checkpoint = os.path.join(self.cfg['pretrain']['pre_folder'], self.cfg['pretrain']['pre_sr'])
        if os.path.exists(checkpoint):
            self.model.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage)['net'],strict=False)
            self.epoch = torch.load(checkpoint, map_location=lambda storage, loc: storage)['epoch']
            if self.epoch > self.nEpochs:
                raise Exception("Pretrain epoch must less than the max epoch!")
        else:
            raise Exception("Pretrain path error!")

    def save_checkpoint(self,epoch):
        super(Solver, self).save_checkpoint()
        self.ckp['net'] = self.model.state_dict()
        self.ckp['optimizer'] = self.optimizer.state_dict()
        if not os.path.exists(self.cfg['checkpoint'] + '/' + str(self.log_name)):
            os.mkdir(self.cfg['checkpoint'] + '/' + str(self.log_name))
        torch.save(self.ckp, os.path.join(self.cfg['checkpoint'] + '/' + str(self.log_name), 'latest.pth'))

        if self.cfg['save_best']:
            if self.records['SSIM'] != [] and self.records['SSIM'][-1] == np.array(self.records['SSIM']).max():
                shutil.copy(os.path.join(self.cfg['checkpoint'] + '/' + str(self.log_name), 'latest.pth'),
                            os.path.join(self.cfg['checkpoint'] + '/' + str(self.log_name), 'bestSSIM.pth'))
            if self.records['PSNR'] !=[] and self.records['PSNR'][-1]==np.array(self.records['PSNR']).max():
                shutil.copy(os.path.join(self.cfg['checkpoint'] + '/' + str(self.log_name), 'latest.pth'),
                            os.path.join(self.cfg['checkpoint'] + '/' + str(self.log_name), 'bestPSNR.pth'))
            if self.records['QNR'] !=[] and self.records['QNR'][-1]==np.array(self.records['QNR']).max():
                shutil.copy(os.path.join(self.cfg['checkpoint'] + '/' + str(self.log_name), 'latest.pth'),
                            os.path.join(self.cfg['checkpoint'] + '/' + str(self.log_name), 'bestQNR.pth'))
            if self.records['D_lamda'] !=[] and self.records['D_lamda'][-1]==np.array(self.records['D_lamda']).min():
                shutil.copy(os.path.join(self.cfg['checkpoint'] + '/' + str(self.log_name), 'latest.pth'),
                            os.path.join(self.cfg['checkpoint'] + '/' + str(self.log_name), 'best_lamda.pth'))
            if self.records['D_s'] !=[] and self.records['D_s'][-1]==np.array(self.records['D_s']).min():
                shutil.copy(os.path.join(self.cfg['checkpoint'] + '/' + str(self.log_name), 'latest.pth'),
                            os.path.join(self.cfg['checkpoint'] + '/' + str(self.log_name), 'bestD_s.pth'))
    def run(self):
        self.check_gpu()
        if self.cfg['pretrain']['pretrained']:
            self.check_pretrained()
        try:
            while self.epoch <= self.nEpochs:
                self.train()
                if self.epoch % 5 == 0:
                    self.eval()
                    self.save_checkpoint(epoch=self.epoch)
                self.epoch += 1
        except KeyboardInterrupt:
            self.save_checkpoint(epoch=self.epoch)
        save_config(self.log_name, 'Training done.')
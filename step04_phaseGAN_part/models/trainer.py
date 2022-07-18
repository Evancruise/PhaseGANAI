import itertools
import os
import numpy as np
import math
import tensorflow as tf
#import Covariance_estimation_network.mvg_distributions as mvg_dist
#from keras import backend as K
from torch import nn
import torchvision.transforms as transforms
import scipy.io as io
from collections import OrderedDict
from abc import ABC
from models.Fresnel_and_phase_retrieval import Fresnel_propagation

from models.networks import PRNet, NN_model, cov_model #, WNet, WNet_DC, SRResNet
#from models.networks_v2 import PRNet, NN_model, cov_model #, WNet, WNet_DC, SRResNet
from models.prop import Phase_retrieval
#from models.prop import Phase_retrieval
from models.discriminator import NLayerDiscriminator
from models.initialization import init_weights
from dataset.Dataset2channel_v import *
#from dataset.Dataset2channel import *
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

class TrainModel(ABC):
    def __init__(self,opt):
        self.opt = opt
        self.RAYS = 128
        self.weights_A_phase = torch.zeros(self.RAYS*self.RAYS,1) #torch.rand(128 * 128, 1) - 0.5
        self.weights_A_absorption = torch.zeros(self.RAYS*self.RAYS,1) #torch.rand(128 * 128, 1) - 0.5
        self.weights_B_I1 = torch.zeros(self.RAYS*self.RAYS,1) #torch.rand(128 * 128, 1) - 0.5
        self.weights_B_I2 = torch.zeros(self.RAYS*self.RAYS,1) #torch.rand(128 * 128, 1) - 0.5
        self.batchnorm = nn.BatchNorm2d(self.RAYS*self.RAYS,affine=True)
        self.model_init_type = 'normal'
        self.load_path = opt.load_path
        self.load_pretrained_model_path = 'pretrained_model/'
        self.save_path = opt.run_path
        self.run_name = opt.run_name + '_' + self.opt.model_A + '_' + self.opt.model_B
        self.load_run_name = opt.run_name + '_' + self.opt.model_A + '_' + self.opt.model_B
        self.batch_size = opt.batch_size
        self.initial_alpham = 0.5
        self.alpham = 0
        self.padding = 2
        self.choice = opt.choice
        self.lambda_GA = opt.lambda_GA
        self.lambda_GB = opt.lambda_GB
        self.lambda_A = opt.lambda_A
        self.lambda_B = opt.lambda_B
        self.model_A = opt.model_A
        self.model_B = opt.model_B
        self.lambda_fscA = opt.lambda_FSCA
        self.lambda_fscB = opt.lambda_FSCB
        self.pretrained = not opt.no_pretrain
        self.num_epochs = opt.num_epochs
        self.lr_g = opt.lr_g
        self.lr_d = opt.lr_d
        self.lr_g_test = opt.lr_g
        self.lr_d_test = opt.lr_d
        self.beta1 = opt.beta1
        self.beta1_test = opt.beta1
        self.clip_max = opt.clip_max
        self.E = opt.E
        # self.pxs = opt.pxs
        self.z = opt.z
        
        #self.z1 = 0.009
        #self.z2 = 0.078
        self.c = opt.c #2.9979e8              
        self.h = opt.h #4.13566e-15        
        #self.wavelength = self.c*self.h/(self.E*1e3) #12.4 / self.E * 1e-10
        #self.wavelength = 3.1e-11 #12.4 / self.E * 1e-10
        self.images_mean,self.images_std,self.reals_mean,self.reals_std,self.imags_mean,self.imags_std = opt.image_stats
        self.model_tt_type = 'normal'
        self.criterioncrossentropy = nn.CrossEntropyLoss()
        self.criterionBSE = nn.BCELoss()
        self.criterionHEB = nn.HingeEmbeddingLoss()
        self.criterionCycle = nn.L1Loss()
        self.criterionCycle2 = nn.MSELoss()
        self.isTrain = not opt.isTest
        self.adjust_lr_epoch = opt.adjust_lr_epoch
        self.batchnorm = nn.BatchNorm2d(1, affine=False)
        self.batchnorm2 = nn.BatchNorm2d(4, affine=False)
        self.log_note = opt.log_note
        self.save_run = F"{self.save_path}/{self.run_name}"
        self.save_log = F"{self.save_run}/log.txt"
        self.save_stats = F"{self.save_run}/stats.txt"
        self.savefig = opt.savefig
        self.load_run = F"{self.load_pretrained_model_path}/{self.load_run_name}"
        self.load_log = F"{self.load_run}/log.txt"
        self.load_stats = F"{self.load_run}/stats.txt"
        self.loss_names = ['D_A', 'G_A', 'D_B', 'G_B']
        self.loss_names_test = ['D_A_test', 'G_A_test', 'D_B_test', 'G_B_test']
        
        self.real_A = torch.zeros((self.batch_size, 2, self.RAYS*self.padding, self.RAYS* self.padding))
        self.real_B = torch.zeros((self.batch_size, len(self.z), self.RAYS*self.padding, self.RAYS* self.padding))
        self.real_A_lr = torch.zeros((self.batch_size, 2, self.RAYS*self.padding, self.RAYS* self.padding))
        self.real_B_lr = torch.zeros((self.batch_size, len(self.z), self.RAYS*self.padding, self.RAYS* self.padding))
        # self.crop = np.linspace(int(self.RAYS/2)+1,int(self.RAYS*3/2),self.RAYS)
        
        self.index_names = ['index_train', 'index_val']
        self.pm_names = ['psnr_A', 'psnr_B', 'ssim_A', 'ssim_B', 'rmse_A', 'rmse_B']
        self.pm_names_test = ['psnr_A_test', 'psnr_B_test', 'ssim_A_test', 'ssim_B_test', 'rmse_A_test', 'rmse_B_test']
        
        self.img_names = ['real_A', 'fake_B', 'rec_A', 'real_B', 'fake_A', 'rec_B', 'real_A', 'fake_B', 'rec_A', 'real_B', 'fake_A', 'rec_B', 'real_B_intensity', 'real_B_intensity', 'fake_B_intensity', 'fake_B_intensity', 'rec_B_intensity', 'rec_B_intensity']

    def get_current_performance_metrics(self, phase):
        pm_list = OrderedDict()
        if phase == 'train':
            for name in self.pm_names:
                if isinstance(name, str):
                    pm_list[name] = float(
                        getattr(self, name))
        else:
            for name in self.pm_names_test:
                if isinstance(name, str):
                    pm_list[name] = float(
                        getattr(self, name))

        return pm_list
        
    def get_current_losses(self, phase):
        errors_list = OrderedDict()
        if phase == 'train':
            for name in self.loss_names:
                if isinstance(name, str):
                    errors_list[name] = float(
                        getattr(self, 'loss_' + name))
        else:
            for name in self.loss_names_test:
                if isinstance(name, str):
                    errors_list[name] = float(
                        getattr(self, 'loss_' + name))
                    
        return errors_list
    
    def im2double(self, im):
         min_val = np.min(im.ravel())
         max_val = np.max(im.ravel())
         out = (im.astype('float') - min_val) / (max_val - min_val) * 255.0
         return out
    
    def im2double_torch(self, im):
         min_val = torch.min(im)
         max_val = torch.max(im)
         out = (im - min_val) / (max_val - min_val) * 255.0
         return out
         
    def save_parameters(self):
        with open(self.save_log, "w+") as f:
            f.write(
                "Training log for {}\r\n\n{}\nlr_g:{}\nlr_d:{}\nlambda_GA:{}\nlambda_GB:{}\nlambda_A:{}\nlambda_B:{}\nlambda_fscA:{}\nlambda_fscB:{}\nbeta1:{}\nclip_max:{}\n\n".format(
                    self.run_name, self.log_note, self.lr_g, self.lr_d, self.lambda_GA, self.lambda_GB, self.lambda_A, self.lambda_B, self.lambda_fscA,self.lambda_fscB,self.beta1, self.clip_max))

    def print_current_losses(self, epoch, iters, losses):
        message = 'Epoch [{}/{}], Step [{}/{}]'.format(epoch+1, self.num_epochs, iters+1, self.total_step)
        for name, loss in losses.items():
            message += ', {:s}: {:.3f}'.format(name, loss)
        print(message)
        with open(self.save_log, "a") as f:
            print(message,file=f)
    
    def print_current_losses_and_performance_metrics(self, epoch, iters, losses, performance_metrics):
        message = 'Epoch [{}/{}], Step [{}/{}]'.format(epoch+1, self.num_epochs, iters+1, self.total_step)
        for name, loss in losses.items():
            message += ', {:s}: {:.3f}'.format(name, loss)
        for name, pr in performance_metrics.items():
            message += ', {:s}: {:.3f}'.format(name, pr)
            
        print(message)
        with open(self.save_log, "a") as f:
            print(message,file=f)

    def adjust_learning_rate(self, epoch, optimizer, initial_lr):
        """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
        lr = initial_lr * (0.01 ** (epoch // self.adjust_lr_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    def adjust_alpha_iterative(self, epoch):
        self.alpham = self.initial_alpham - (0.005 * epoch)
            
    def update_learning_rate(self,epoch,phase):
        if phase == 'train':
            self.adjust_learning_rate(epoch, self.optimizer_G,self.lr_g)
            self.adjust_learning_rate(epoch, self.optimizer_D,self.lr_d)
            self.adjust_alpha_iterative(epoch)

    def load_data(self):
        print('start loading data....')
        if self.isTrain:
            train_dataset = Dataset2channel(self.load_path, 'train', recursive=False, load_data=False,
                                            data_cache_size=1, transform=transforms.ToTensor())
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            train_loader = []
        test_dataset = Dataset2channel(self.load_path + '/test', 'test', recursive=False, load_data=False,
                                       data_cache_size=1, transform=transforms.ToTensor())
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=self.batch_size, shuffle=False)
        
        print('len of train_dataset:', len(train_dataset))
        print('len of test_dataset:', len(test_dataset))
        print('finish loading data.')
        print('len of train_loader:', len(train_loader))
        print('len of test_loader:', len(test_loader))
        
        return train_loader, train_dataset, test_loader, test_dataset

    def NN_model(self, num_out=1, model_name='SLNN'):
        
        model = NN_model(n_out = num_out, model_name='SLNN')
        
        return model
    
    def get_model(self, num_out=1, model_name='WNet'):
        
        model = PRNet(pretrained=self.pretrained, batch_size=self.batch_size , n_out=num_out, model_name = model_name)
        model.eval()
        if self.pretrained is not True:
            init_weights(model, self.model_init_type, init_gain=0.03)
        return model

    def get_NLdnet(self, num_input=1):
        dnet = NLayerDiscriminator(num_input)
        dnet.eval()
        init_weights(dnet, self.model_init_type, init_gain=0.02)
        return dnet

    def create_dir_if_not_exist(self, path):
        if os.path.exists(path):
            decision = input('This folder already exists. Continue training will overwrite the data. Proceed(y/n)?')
            if decision != 'y':
                exit()
            else:
                print('Warning: Overwriting folder: {}'.format(self.run_name))
        if not os.path.exists(path):
            print('Make directory in path:', path)
            os.makedirs(path)
    
    def cov_model(self, num_out=1, batch_size=16, model_name='Cov_reduction_network'):
        
        model = cov_model(n_out = num_out, batch_size=self.batch_size, model_name='Cov_reduction_network')
        
        return model

    def init_model(self, model_A_name, model_B_name):
        self.netG_A = self.get_model(num_out=1, model_name=model_A_name)
        self.netG_B = self.get_model(num_out=1, model_name=model_B_name)
        self.netG_A_cov_reduction = self.cov_model(num_out=1, batch_size=self.batch_size, model_name='Cov_estimation_network')
        self.netG_B_cov_reduction = self.cov_model(num_out=1, batch_size=self.batch_size, model_name='Cov_estimation_network')
        self.netD_A = self.get_NLdnet(1)
        self.netD_B = self.get_NLdnet(1)
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=self.lr_g, betas=(self.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=self.lr_d, betas=(self.beta1, 0.999))
        
        self.train_loader, self.train_dataset, self.test_loader, self.test_dataset = self.load_data()
        self.total_step = len(self.train_loader)
        self.total_step_test = len(self.test_loader)
        self.create_dir_if_not_exist(self.save_run)
        self.save_parameters()
        
        return self.total_step, self.total_step_test, self.run_name

    def optimize_parameters(self, model_A_name, model_B_name):
        self.init_model(model_A_name, model_B_name)

    def set_input(self, input):
        
        # r_index = torch.arange(0,self.batch_size)
        
        phase0, absorption0, I1, I2, phase_contact0, absorption_contact0, I1_0, I2_0 = [input[i].to(torch.float32) for i in range(8)]
        
        self.indexes = input[8]
        self.hypotheses = input[9]
        self.total_num = input[10]
        #self.total_num_test = input[10]
        self.image_type = input[11]
        self.recon_alg = input[12]
        
        for i in range(self.batch_size):
            
            self.real_A_lr[i,0,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)] = phase0[i,:,:]
            self.real_A_lr[i,1,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)] = absorption0[i,:,:]
            self.real_A[i,0,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)] = phase_contact0[i,:,:]
            self.real_A[i,1,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)] = absorption_contact0[i,:,:]
            self.real_B_lr[i,0,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)] = I1[i,:,:]
            self.real_B_lr[i,1,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)] = I2[i,:,:]
            self.real_B[i,0,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)] = I1_0[i,:,:]
            self.real_B[i,1,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)] = I2_0[i,:,:]
        
        '''
        real_A_re_rc = phase_contact[r_index][:, :].to(torch.float32)
        real_A_im_rc = absorption_contact[r_index][:, :].to(torch.float32)
        real_A = self.standard_channels(real_A_re_rc, real_A_im_rc)
        
        fig, ax = plt.subplots(2,int(self.batch_size))
        for i in range(int(self.batch_size)):
            ax[0,i].imshow(real_A[i,0,:,:])
            ax[1,i].imshow(real_A[i,1,:,:])
        plt.savefig('real_A.png')
        plt.close()
        
        real_B_re_rc = I1_0_reshape[r_index][:, :].to(torch.float32)
        real_B_im_rc = I2_0_reshape[r_index][:, :].to(torch.float32)
        real_B = self.standard_channels(real_B_re_rc, real_B_im_rc)
        
        fig, ax = plt.subplots(2,int(self.batch_size))
        for i in range(int(self.batch_size)):
            ax[0,i].imshow(real_B[i,0,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)])
            ax[1,i].imshow(real_B[i,1,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)])
        plt.savefig('real_B.png')
        plt.close()
        
        real_A_re_rc_lr = phase[r_index][:, :].to(torch.float32)
        real_A_im_rc_lr = absorption[r_index][:, :].to(torch.float32)
        real_A_lr = self.standard_channels(real_A_re_rc_lr, real_A_im_rc_lr)
        
        real_B_re_rc_lr = I1_reshape[r_index][:, :].to(torch.float32)
        real_B_im_rc_lr = I2_reshape[r_index][:, :].to(torch.float32)
        real_B_lr = self.standard_channels(real_B_re_rc_lr, real_B_im_rc_lr)
        '''
        
        '''
        prop_A, prop_A0 = Phase_retrieval(self.opt).fresnel_prop(real_A)
        
        retrieve_A = Phase_retrieval(self.opt).iterative_method(real_B, prop_A0, real_A)
        
        fig, ax = plt.subplots(4,2)
        ax[0,0].imshow(np.array(real_A[0,0,:,:]))
        ax[0,1].imshow(np.array(real_A[0,1,:,:]))
        ax[1,0].imshow(np.array(real_A_lr[0,0,:,:]))
        ax[1,1].imshow(np.array(real_A_lr[0,1,:,:]))
        ax[2,0].imshow(np.array(prop_A[0,0,:,:]))
        ax[2,1].imshow(np.array(prop_A[0,1,:,:]))
        ax[3,0].imshow(np.array(retrieve_A[0,0,:,:].detach().numpy()))
        ax[3,1].imshow(np.array(retrieve_A[0,1,:,:].detach().numpy()))
        plt.savefig('phi_A_rec.png')
        plt.close()
        '''
        
        # return index_train, hypothesis_train, real_A, real_A_lr, real_B, real_B_lr, total_num_train
        
        '''
    def val_input(self,input):
        
        # r_index_test = torch.arange(0,self.batch_size)
        
        phase0_test, absorption0_test, I1_test, I2_test, phase_contact0_test, absorption_contact0_test, I1_0_test, I2_0_test = [input[i].to(torch.float32) for i in range(8)]
        
        self.index_val = input[8]
        self.hypothesis_val = input[9]
        self.total_num_test = input[11]
        
        self.real_A_test = torch.zeros((self.batch_size, 2, self.RAYS*self.padding, self.RAYS* self.padding))
        self.real_B_test = torch.zeros((self.batch_size, 2, self.RAYS*self.padding, self.RAYS* self.padding))
        self.real_A_lr_test = torch.zeros((self.batch_size, 2, self.RAYS*self.padding, self.RAYS* self.padding))
        self.real_B_lr_test = torch.zeros((self.batch_size, 2, self.RAYS*self.padding, self.RAYS* self.padding))
        
        for i in range(self.batch_size):
            
            self.real_A_lr_test[i,0,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)] = phase0_test[i,:,:]
            self.real_A_lr_test[i,1,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)] = absorption0_test[i,:,:]
            self.real_A_test[i,0,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)] = phase_contact0_test[i,:,:]
            self.real_A_test[i,1,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)] = absorption_contact0_test[i,:,:]
            self.real_B_lr_test[i,0,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)] = I1_test[i,:,:]
            self.real_B_lr_test[i,1,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)] = I2_test[i,:,:]
            self.real_B_test[i,0,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)] = I1_0_test[i,:,:]
            self.real_B_test[i,1,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)] = I2_0_test[i,:,:]
        #return index_val, hypothesis_val, real_A_test, real_A_lr_test, real_B_test, real_B_lr_test, total_num_test
        '''
        
    def standard_channels(self, real_re_rc, real_im_rc):
        real_re = (real_re_rc - self.reals_mean) / self.reals_std
        real_im = (real_im_rc - self.imags_mean) / self.imags_std
        real = torch.cat((real_re, real_im), 1)
        return real

    def standard_forward_prop(self, fake_A):
        prop_A, prop_A0 = Fresnel_propagation(self.opt).forward_propagation(fake_A)
        #prop_A, prop_A0 = Phase_retrieval(self.opt).fresnel_prop(fake_A)
        return prop_A, prop_A0
    
    def standard_backward_prop(self, fake_B, prop_A0, real_A):
        retrieve = Fresnel_propagation(self.opt).backward_propagation(fake_B, prop_A0, real_A)
        #retrieve = Phase_retrieval(self.opt).iterative_method(fake_B, prop_A0, real_A)
        return retrieve

    def GANLoss(self, pred, is_real):
        target = self.logic_tensor(pred, is_real)
        loss = self.criterionBSE(pred, target)
        return loss
    
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window
    
    def compute_MSE(self, img1, img2):
        return ((img1 - img2) ** 2).mean()
    
    def compute_RMSE(self, img1, img2):
        
        #for i in range(10):
        #    img1[i,0,:,:] = self.im2double_torch(img1[i,0,:,:])
        #    img2[i,0,:,:] = self.im2double_torch(img2[i,0,:,:])
        
        if type(img1) == torch.Tensor:
            return torch.sqrt(self.compute_MSE(img1, img2)).item()
        else:
            return np.sqrt(self.compute_MSE(img1, img2))

    def compute_PSNR(self, img1, img2):
        data_range = 1
        
        if type(img1) == torch.Tensor:
            mse_ = self.compute_MSE(img1, img2)
            return 10 * torch.log10((data_range ** 2) / mse_).item()
        else:
            mse_ = self.compute_MSE(img1, img2)
            return 10 * np.log10((data_range ** 2) / mse_)

    def compute_SSIM(self, img1, img2):
        # referred from https://github.com/Po-Hsun-Su/pytorch-ssim
        
        #for i in range(10):
        #    img1[i,0,:,:] = self.im2double_torch(img1[i,0,:,:])
        #    img2[i,0,:,:] = self.im2double_torch(img2[i,0,:,:])
        
        #data_range = torch.max(img1, img2) - torch.min(img1, img2)
        size_average = True
        data_range = 1
        
        if len(img1.size()) == 2:
            shape_ = img1.shape[-1]
            img1 = img1.view(1,1,shape_ ,shape_ )
            img2 = img2.view(1,1,shape_ ,shape_ )
        
        window_size = 11
        window = self.create_window(window_size, 10)
        window = window.type_as(img1)

        mu1 = F.conv2d(img1.unsqueeze(1), window, padding=window_size//2)
        mu2 = F.conv2d(img2.unsqueeze(1), window, padding=window_size//2)
        mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1.unsqueeze(1)*img1.unsqueeze(1), window, padding=window_size//2) - mu1_sq
        sigma2_sq = F.conv2d(img2.unsqueeze(1)*img2.unsqueeze(1), window, padding=window_size//2) - mu2_sq
        sigma12 = F.conv2d(img1.unsqueeze(1)*img2.unsqueeze(1), window, padding=window_size//2) - mu1_mu2

        C1, C2 = (0.01*data_range)**2, (0.03*data_range)**2
        #C1, C2 = 0.01**2, 0.03**2

        ssim_map = ((2*mu1_mu2+C1)*(2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
        if size_average:
            return ssim_map.mean().item()
        else:
            return ssim_map.mean(1).mean(1).mean(1).item()
            
    def plot_cycle(self, save_name, model_A, model_B, plot_phase=True, phase='train', step=1, epoch=1):
    # def plot_cycle(self, img_idx,save_name,indexes, hypotheses, real_A, fake_B, rec_A, real_B, fake_A, rec_B, model_A,model_B,test=False, plot_phase=True, phase='train', total_num=1000, step=1, epoch=1): 
        """set layer to 1 and plot_phase to False to plot imag channel"""
        
        img_list = ['real_A', 'rec_A', 'real_B', 'rec_B', 'real_A', 'rec_A', 'real_B', 'rec_B'] #['real_A', 'fake_B', 'rec_A', 'real_B', 'fake_A', 'rec_B', 'real_A', 'fake_B', 'rec_A', 'real_B', 'fake_A', 'rec_B']
        img_list_ = ['phi', 'phi_rec', 'I1', 'I1_rec', 'A', 'A_rec', 'I2', 'I2_rec']
        
        save_path = "{}/{}/epoch{}/".format(self.save_run, phase, epoch)  #F"{self.save_run}/{phase}/epoch{epoch}/"
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            os.makedirs(save_path + 'pictures/')
               
        for img_idx in range(self.batch_size):
            #axs = axs.ravel()
            
            I_matrix = np.zeros((self.RAYS,self.RAYS,len(self.z),2))
            
            for j in range(len(self.z)):
                I_matrix[:,:,j,0] = self.real_B[img_idx,j,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)]
                I_matrix[:,:,j,1] = self.real_B_lr[img_idx,j,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)]
            
            io.savemat(save_path + "PCI_{}_{}_{}_{}.mat".format(str(self.image_type[0].decode("utf-8")), str(self.indexes[img_idx].item()), str(self.hypotheses[img_idx].decode("utf-8")), str(self.recon_alg[0].decode("utf-8"))), {'phi_proj': self.real_A[img_idx,0,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)], 'phi': self.rec_A[img_idx,0,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)], 'A_proj': self.real_A[img_idx,1,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)], 'A': self.rec_A[img_idx,1,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)], 'I': I_matrix})
            
            #axs[i].axis("off")
            #axs[i].set_title(img_list[img_idx], fontsize=36)
            
            if step != int(self.total_num[0]/self.batch_size):
                fig, axs = plt.subplots(2, int(len(img_list)/2), figsize=(20, 20), facecolor='w', edgecolor='k')
                fig.subplots_adjust(hspace=0.0001, wspace=0.0001)
            
                for i in range(len(self.z)):
                    for j in range(int(len(img_list)/2)):
                        axs[i,j].imshow(getattr(self,img_list[int(len(img_list)/2)*i+j])[img_idx, i, int(self.RAYS/2):int(self.RAYS*3/2), int(self.RAYS/2):int(self.RAYS*3/2)].detach().numpy(), cmap = plt.cm.gray)
                        axs[i,j].axis("off")
                        axs[i,j].set_title(img_list_[int(len(img_list)/2)*i+j], fontsize=36)
            
                if self.savefig == 1:
                    plt.savefig(save_path + "pictures/PCI_{}_{}_{}_{}.png".format(str(self.image_type[0].decode("utf-8")),str(self.indexes[img_idx].item()),str(self.hypotheses[img_idx].decode("utf-8")), self.recon_alg[0].decode("utf-8")))
            
                plt.cla()
                plt.close()  
              
    def logic_tensor(self, pred, is_real):
        if is_real:
            target = torch.tensor(1.0)
        else:
            target = torch.tensor(0.0)
        return target.expand_as(pred).to(torch.float32)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def backward_D(self, netD, real, fake, phase):
        pred_real = netD(real)
        #print(pred_real)
        loss_D_real = self.GANLoss(pred_real, True)
        pred_fake = netD(fake.detach())
        #print(pred_fake)
        loss_D_fake = self.GANLoss(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        
        if phase == 'train':
            loss_D.backward()
        
        return loss_D
    
    def _validate_input(self, x, batch_first=False):
        """
        Args:
            x: tensor of [batch size, num features] or [num samples, batch size, num features]

        Returns:
            tensor of [batch size, num features] or [batch size, num samples, num features]
        """
        x = tf.convert_to_tensor(x)
        if x.shape.ndims == 2:
            return x
        if x.shape.ndims == 3:
            if batch_first:
                return tf.transpose(x, [1, 0, 2])
            else:
                return x
        raise RuntimeError("Tensor must be rank 2 or 3, found {}".format(x.shape.ndims))
    
    def _r_precision_r(self, x):
        """
        Computes (x - mu) inv(Sigma) (x - mu)^T
        Args:
            x: tensor of [batch size, num samples, num features]

        Returns:
            A tensor of [num samples, batch size]
        """
        loc = self._expand_if_x_rank_3(self.loc, x, axis=1)

        # x_precision_x expects data in [batch size, num samples, num features]
        r_precision_r = self.cov_obj.x_precision_x(x - loc)

        if x.shape.ndims == 3:
            if r_precision_r.shape.ndims == 1:
                # x_precision_x removes sample dimensions if sample dim is 1, add it again
                r_precision_r = tf.expand_dims(r_precision_r, axis=1)

            # Transpose to [num samples, batch size]
            r_precision_r = tf.transpose(r_precision_r, [1, 0])
        self.__r_precision_r = r_precision_r
        return r_precision_r

    def _k_log_2_pi(self, x):
        with tf.name_scope('k_log_2_pi'):
            k = tf.cast(tf.shape(x)[-1], x.dtype)
            return k * np.log(2.0 * np.pi)
    
    def _expand_if_x_rank_3(value, x, axis):
        if x.shape.ndims == 2:
            return value
        else:
            return tf.expand_dims(value, axis=axis)
              
    def log_prob(self, x):
        """
        log p(x) = - 0.5 * [ log(det(Sigma)) + (x - mu) inv(Sigma) (x - mu)^T + k log(2 pi) ]
        Args:
            x: tensor of [batch size, num features] or [num samples, batch size, num features]

        Returns:
            log p(x) tensor of [num samples, batch size, num features]
        """
        x = self._validate_input(x, batch_first=True)

        r_precision_r = self._r_precision_r(x)

        k_log_2_pi = self._k_log_2_pi(x)

        log_det_cov = self._expand_if_x_rank_3(self.log_det_covar, x, axis=0)

        return -0.5 * (log_det_cov + r_precision_r + k_log_2_pi)
    
    def batch_squared_error_with_covariance(self, predictions, labels, inv_covariance,
                                         name='batch_squared_error_with_covariance', out_name=None):
        
        if labels is None:
            labels_m_pred = predictions
        else:
            labels_m_pred = labels - predictions

        labels_m_pred = tf.expand_dims(labels_m_pred.detach().numpy(), 1)
        
        '''
        # Matrix with the left side: (x-mu) Sigma^-1
        if len(inv_covariance.shape) == 2:
            # Shared covariance matrix
            left_side = tf.matmul(labels_m_pred, inv_covariance)
        else:
            # A covariance matrix per element in the batch
            left_side = tf.matmul(labels_m_pred, inv_covariance)
        '''
        
        left_side = np.matmul(labels_m_pred, inv_covariance)
        # Explicitly multiply each element and sum over the rows, i.e. batch-wise dot product
        batch_mse = np.multiply(left_side, labels_m_pred)
        batch_mse = np.mean(np.mean(batch_mse, axis=2))  # Error per sample
        
        #print('batch_mse:', batch_mse)
        #if batch_mse.shape[1].value == 1:
        #    batch_mse = np.squeeze(batch_mse, axis=1, name=out_name)  # Remove sample dimension

        return batch_mse
    
    def get_inv_covariance(self, inv_covariance=None, covariance=None, dtype=tf.float32, name='inv_covariance'):
        # Compute the inverse of a covariance matrix
        '''
        if inv_covariance is None:
            if covariance is not None:
                if isinstance(covariance, tf.Tensor):
                    inv_covariance = tf.matrix_inverse(covariance)
                    inv_covariance = tf.cast(inv_covariance, dtype)
                else:
                    inv_covariance = np.linalg.inv(covariance.detach().numpy())
                    inv_covariance = inv_covariance.astype(dtype.as_numpy_dtype)
            else:
                raise RuntimeError("Must provide covariance matrix if inv(Sigma) is not given")
        '''
        
        inv_covariance = np.linalg.inv(covariance.detach().numpy())
        inv_covariance = inv_covariance.astype(dtype.as_numpy_dtype)
                    
        return inv_covariance
        
    def loss_NPCC(self, y_true, y_pred):
        x = y_true
        y = y_pred
        mx = torch.mean(x)
        my = torch.mean(y)
        xm, ym = x-mx, y-my
        r_num = torch.sum(xm * ym)
        r_den = torch.sum(torch.sum(torch.pow(xm, 2)) * torch.sum(torch.pow(ym, 2)))
        r = r_num / r_den
        return 1 - r**2
    
    def get_k_log_2_pi(self, x):
        # Compute k * log(2*pi), where k is the dimensionality of x
        k_log_2_pi = np.sum(-np.log(x.detach().numpy()))
        return k_log_2_pi
    
    # Define the loss of the model as the negative log likelihood of a multivariate Gaussian distribution
    def log_prob_loss(self, chol_precision_weights, log_diag_precision, y_true, y_pred):
        
        y_predict_flat = torch.reshape(y_pred, (self.batch_size, 128, 128)) #keras.layers.Flatten(input_shape=(w, h))(y_pred)
        #log_diag_precision = torch.reshape(log_diag_precision, (self.batch_size, -1))
        
        inv_covariance = np.exp(-log_diag_precision.detach().numpy())
        #inv_covariance = np.exp(-chol_precision_weights[:,:,:,0].unsqueeze(-1).detach().numpy())
        
        #print('y_mean:', y_mean)
        #print('log_diag_precision:', log_diag_precision)
        
        #mvg = mvg_dist.MultivariateNormalDiag(loc=y_mean, log_diag_precision=log_diag_precision)
        
        #mvg = mvg_dist.MultivariateNormalPrecCholFilters(loc=y_mean, weights_precision=chol_precision_weights,
        #                                                 filters_precision=None,
        #                                                 log_diag_chol_precision=log_diag_precision,
        #                                                 sample_shape=(10,128,128,1))
        
        
        #mvg = mvg_dist.MultivariateNormalPrecCholFilters(loc=y_mean, weights_precision=chol_precision_weights,
        #                                                 filters_precision=None,
        #                                                 log_diag_chol_precision=log_diag_chol_precision,
        #                                                 sample_shape=(self.batch_size,128,128,1))
        
        '''                                           
        # The probability of the input data under the model
        y_true_flat = torch.reshape(y_true, (self.batch_size, -1)) #keras.layers.Flatten(input_shape=(w, h))(y_true)
        #log_prob = mvg.log_prob(y_true_flat.detach().numpy())
        log_prob = self.log_prob(y_true_flat.detach().numpy())
        #print('log_prob:', log_prob)
        neg_log_prob = tf.reduce_mean(log_prob)
        #print('neg_log_prob:', neg_log_prob)
        '''

        # The probability of the input data under the model
        y_true_flat = torch.reshape(y_true, (self.batch_size, 128, 128))
        squared_error = self.batch_squared_error_with_covariance(y_true_flat, y_predict_flat, inv_covariance)
        k_log_2_pi = self.get_k_log_2_pi(log_diag_precision)
        
        #with tf.Session() as sess:
        #    log_prob,neg_log_prob = sess.run([log_prob, neg_log_prob])
        #print(0.5 * (squared_error + k_log_2_pi))

        return torch.tensor(0.5 * (squared_error + k_log_2_pi), requires_grad=True)
    
    '''    
    def loss_covariance_matrix_reduction_A(self, real, fake):
        
        pred_p, chol_precision_weights_p, log_diag_chol_precision_p = self.netG_A_cov_reduction_p(torch.unsqueeze(fake[:,0,:,:], 1))
        
        pred_b, chol_precision_weights_b, log_diag_chol_precision_b = self.netG_A_cov_reduction_b(torch.unsqueeze(fake[:,2,:,:], 1))
        
        #weights_precision = np.random.normal(size=(b, w, h, nb))  # Random matrix
        #chol_precision_weights_p = chol_precision_weights_p.astype(dtype)
        chol_precision_weights_p[..., 0:2 // 2] = 0  # Equivalent of upper triangular to zero
        log_diag_chol_precision_p = chol_precision_weights_p[..., 2 // 2]  # Get the diagonal
        chol_precision_weights_p[..., 2 // 2] = torch.exp(log_diag_chol_precision_p)  # Exponentiate to remove log
        log_diag_chol_precision_p = torch.reshape(log_diag_chol_precision_p, (self.batch_size, 128 * 128))  # Flatten
        
        #weights_precision = np.random.normal(size=(b, w, h, nb))  # Random matrix
        #chol_precision_weights_b = chol_precision_weights_p.astype(dtype)
        chol_precision_weights_b[..., 0:2 // 2] = 0  # Equivalent of upper triangular to zero
        log_diag_chol_precision_b = chol_precision_weights_p[..., 2 // 2]  # Get the diagonal
        chol_precision_weights_b[..., 2 // 2] = torch.exp(log_diag_chol_precision_b)  # Exponentiate to remove log
        log_diag_chol_precision_b = torch.reshape(log_diag_chol_precision_b, (self.batch_size, 128 * 128))  # Flatten

        loss_p = self.log_prob_loss(chol_precision_weights_p, log_diag_chol_precision_p, real[:,0,:,:], pred_p)
        # loss_b = self.log_prob_loss(chol_precision_weights_b, log_diag_chol_precision_b, real[:,2,:,:], pred_b)
        
        # print(loss_p, loss_b)
        
        return loss_p #, loss_b
    '''

    def loss_covariance_matrix_reduction_A(self, real, fake):
        
        pred_p, chol_precision_weights_p, log_diag_chol_precision_p = self.netG_A_cov_reduction(torch.unsqueeze(fake[:,0,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)], 1))
        
        '''
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(fake[0,0,:,:].detach().numpy())
        ax[1].imshow(pred_p[0,0,:,:].detach().numpy())
        plt.show()
        '''
        
        #pred_b, chol_precision_weights_b, log_diag_chol_precision_b = self.netG_B_cov_reduction_b(torch.unsqueeze(fake[:,1,:,:], 1))
        
        #weights_precision = np.random.normal(size=(b, w, h, nb))  # Random matrix
        #chol_precision_weights_p = chol_precision_weights_p.astype(dtype)
        chol_precision_weights_p[..., 0:2 // 2] = 0  # Equivalent of upper triangular to zero
        log_diag_chol_precision_p = chol_precision_weights_p[..., 2 // 2]  # Get the diagonal
        chol_precision_weights_p[..., 2 // 2] = torch.exp(log_diag_chol_precision_p)  # Exponentiate to remove log
        log_diag_chol_precision_p = torch.reshape(log_diag_chol_precision_p, (self.batch_size, 128, 128))  # Flatten
        
        #weights_precision = np.random.normal(size=(b, w, h, nb))  # Random matrix
        #chol_precision_weights_b = chol_precision_weights_p.astype(dtype)
        #chol_precision_weights_b[..., 0:2 // 2] = 0  # Equivalent of upper triangular to zero
        #log_diag_chol_precision_b = chol_precision_weights_p[..., 2 // 2]  # Get the diagonal
        #chol_precision_weights_b[..., 2 // 2] = torch.exp(log_diag_chol_precision_b)  # Exponentiate to remove log
        #log_diag_chol_precision_b = torch.reshape(log_diag_chol_precision_b, (self.batch_size, 128 * 128))  # Flatten
        
        loss_p = self.log_prob_loss(chol_precision_weights_p, log_diag_chol_precision_p, real[:,0,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)], pred_p)
        #loss_b = self.log_prob_loss(chol_precision_weights_b, log_diag_chol_precision_b, real[:,1,:,:], pred_b)
        
        # print(loss_p, loss_b)
        
        return loss_p #, loss_b
    
    def loss_covariance_matrix_reduction_B(self, real, fake):
        
        pred_p, chol_precision_weights_p, log_diag_chol_precision_p = self.netG_B_cov_reduction(torch.unsqueeze(fake[:,0,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)], 1))
        
        '''
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(fake[0,0,:,:].detach().numpy())
        ax[1].imshow(pred_p[0,0,:,:].detach().numpy())
        plt.show()
        '''
        
        #pred_b, chol_precision_weights_b, log_diag_chol_precision_b = self.netG_B_cov_reduction_b(torch.unsqueeze(fake[:,1,:,:], 1))
        
        #weights_precision = np.random.normal(size=(b, w, h, nb))  # Random matrix
        #chol_precision_weights_p = chol_precision_weights_p.astype(dtype)
        chol_precision_weights_p[..., 0:2 // 2] = 0  # Equivalent of upper triangular to zero
        log_diag_chol_precision_p = chol_precision_weights_p[..., 2 // 2]  # Get the diagonal
        chol_precision_weights_p[..., 2 // 2] = torch.exp(log_diag_chol_precision_p)  # Exponentiate to remove log
        log_diag_chol_precision_p = torch.reshape(log_diag_chol_precision_p, (self.batch_size, 128, 128))  # Flatten
        
        #weights_precision = np.random.normal(size=(b, w, h, nb))  # Random matrix
        #chol_precision_weights_b = chol_precision_weights_p.astype(dtype)
        #chol_precision_weights_b[..., 0:2 // 2] = 0  # Equivalent of upper triangular to zero
        #log_diag_chol_precision_b = chol_precision_weights_p[..., 2 // 2]  # Get the diagonal
        #chol_precision_weights_b[..., 2 // 2] = torch.exp(log_diag_chol_precision_b)  # Exponentiate to remove log
        #log_diag_chol_precision_b = torch.reshape(log_diag_chol_precision_b, (self.batch_size, 128 * 128))  # Flatten
        
        loss_p = self.log_prob_loss(chol_precision_weights_p, log_diag_chol_precision_p, real[:,0,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)], pred_p)
        #loss_b = self.log_prob_loss(chol_precision_weights_b, log_diag_chol_precision_b, real[:,1,:,:], pred_b)
        
        # print(loss_p, loss_b)
        
        return loss_p #, loss_b
        
    def loss_HO(self, real, fake, w, y, mean_H0, mean_H1):
        
        #loss = torch.matmul(torch.pow(torch.matmul(w, (fake.reshape((128*128,10))-mean_H0.reshape((128*128,10)))), 2), y.reshape((10, 1))) + torch.matmul(torch.pow(torch.matmul(w, (fake.reshape((128*128,10))-mean_H1.reshape((128*128,10)))), 2), y.reshape((10, 1)))
        
        loss = np.matmul(np.matmul(w, (fake.reshape((128*128,self.batch_size)).detach().numpy()-mean_H0.reshape((128*128,self.batch_size)).detach().numpy())) ** 2, y.reshape((self.batch_size, 1)).detach().numpy()) + np.matmul(np.matmul(w, (fake.reshape((128*128,self.batch_size)).detach().numpy()-mean_H1.reshape((128*128,self.batch_size)).detach().numpy())) ** 2, y.reshape((self.batch_size, 1)).detach().numpy())
        
        return loss #torch.mean(torch.pow((real - fake), 2))
    
    def VGGLoss(self, real, fake):
        
        loss = []
        
        for i in range(self.batch_size):
            loss.append((fake[i,0,:,:].detach().numpy() - real[i,0,:,:].detach().numpy()) ** 2)
        
        return np.mean(np.sqrt(loss))
    
    def loss_HO_supervised(self, real, fake):
        
        g0_I1 = torch.zeros((128*128, 1))
        g1_I1 = torch.zeros((128*128, 1))
        g0_I1 = torch.reshape(self.batchnorm(real[:,0,:,:].unsqueeze(1)), (-1,128*128,1))
        g1_I1 = torch.reshape(self.batchnorm(fake[:,0,:,:].unsqueeze(1)), (-1,128*128,1))

        #for i in range(int(self.batch_size/2)):
        #    g0_I1 += torch.reshape(real[i,0,:,:], (128*128, 1))
        #    g1_I1 += torch.reshape(real[i+1,0,:,:], (128*128, 1))
        #    #g0_I2 += torch.reshape(real[i,2,:,:], (128*128, 1))
        #    #g1_I2 += torch.reshape(real[i+1,2,:,:], (128*128, 1))
        
        delta_g_I1 = g1_I1-g0_I1
        loss_G_I1 = 0
        
        y = self.netG_A_HO_template(fake)
        
        for i in range(self.batch_size):
            if y[i,0] > 0.5:
                loss_G_I1 += torch.matmul(torch.transpose(self.netG_A_HO_template.dense3_2_1.weight.data.reshape(128*128,1), 0, 1), (torch.reshape(fake[i,0,:,:], (128*128, 1)) - g0_I1)) ** 2
            else:    
                loss_G_I1 += torch.matmul(torch.transpose(self.netG_A_HO_template.dense3_2_1.weight.data.reshape(128*128,1), 0, 1), (torch.reshape(fake[i,0,:,:], (128*128, 1)) - g1_I1)) ** 2
        
        loss_G_I1 -= 2*torch.matmul(torch.transpose(self.netG_A_HO_template.dense3_2_1.weight.data.reshape(128*128,1), 0, 1), delta_g_I1)
        
        return torch.tensor(torch.mean(loss_G_I1), requires_grad=True) #+loss_G_I2
    
    def backward_G(self, phase):
    # def backward_G(self, phase, real_A, fake_A, rec_A, real_B, fake_B, rec_B):
        
        if phase == 'train':
          with torch.autograd.set_detect_anomaly(True):
            
            '''
            if self.choice == 1:
                self.loss_cov_A = self.loss_covariance_matrix_reduction_A(rec_A.clone().detach().requires_grad_(True), real_A.clone().detach().requires_grad_(True))
                self.loss_cov_B = self.loss_covariance_matrix_reduction_B(rec_B_intensity.clone().detach().requires_grad_(True), real_A_intensity.clone().detach().requires_grad_(True))
            else:
                self.loss_Hotelling_template = self.loss_HO_supervised(rec_A.clone().detach().requires_grad_(True), real_A.clone().detach().requires_grad_(True))
            '''
            
            self.loss_G_A = self.GANLoss(self.netD_A(self.real_B[:,0,:,:].unsqueeze(1).clone().detach().requires_grad_(True)), True) * self.lambda_GA
            #self.loss_VGG_B = self.VGGLoss(real_B_VGG, rec_B_VGG)
            self.loss_cycle_B = self.criterionCycle2(self.rec_B[:,0,:,:].clone().detach().requires_grad_(True), self.real_B[:,0,:,:].clone().detach().requires_grad_(True)) * self.lambda_B
            #self.loss_NPCC_B = self.loss_NPCC(rec_B[:,0,:,:].clone().detach().requires_grad_(True), real_B[:,0,:,:].clone().detach().requires_grad_(True))
            self.psnr_B = self.compute_PSNR(self.rec_B[:,0,:,:].clone().detach().requires_grad_(True), self.real_B[:,0,:,:].clone().detach().requires_grad_(True))
            self.ssim_B = self.compute_SSIM(self.rec_B[:,0,:,:].clone().detach().requires_grad_(True), self.real_B[:,0,:,:].clone().detach().requires_grad_(True))
            self.rmse_B = self.compute_RMSE(self.rec_B[:,0,:,:].clone().detach().requires_grad_(True), self.real_B[:,0,:,:].clone().detach().requires_grad_(True))
            self.loss_G_B = self.GANLoss(self.netD_B(self.rec_A[:,0,:,:].unsqueeze(1).clone().detach().requires_grad_(True)), False) * self.lambda_GB
            #self.loss_VGG_A = self.VGGLoss(real_A_intensity_VGG, rec_A_intensity_VGG)
            self.loss_cycle_A = self.criterionCycle2(self.rec_A[:,0,:,:].clone().detach().requires_grad_(True), self.real_A[:,0,:,:].clone().detach().requires_grad_(True)) * self.lambda_A
            #self.loss_NPCC_A = self.loss_NPCC(rec_A[:,0,:,:].clone().detach().requires_grad_(True), real_A[:,0,:,:].clone().detach().requires_grad_(True))
            self.psnr_A = self.compute_PSNR(self.rec_A[:,0,:,:].clone().detach().requires_grad_(True), self.real_A[:,0,:,:].clone().detach().requires_grad_(True))
            self.ssim_A = self.compute_SSIM(self.rec_A[:,0,:,:].clone().detach().requires_grad_(True), self.real_A[:,0,:,:].clone().detach().requires_grad_(True))
            self.rmse_A = self.compute_RMSE(self.rec_A[:,0,:,:].clone().detach().requires_grad_(True), self.real_A[:,0,:,:].clone().detach().requires_grad_(True))
            
            self.loss_cov_A = self.loss_covariance_matrix_reduction_A(self.rec_A.clone().detach().requires_grad_(True), self.real_A.clone().detach().requires_grad_(True))
            self.loss_cov_B = self.loss_covariance_matrix_reduction_B(self.rec_B.clone().detach().requires_grad_(True), self.real_B.clone().detach().requires_grad_(True))
                
            self.loss_G = self.loss_G_A + self.loss_cycle_A + self.loss_G_B + self.loss_cycle_B + self.loss_cov_A + self.loss_cov_B
            
            '''
            if self.choice == 1:
                self.loss_G = self.loss_G_A + self.loss_cycle_A + self.loss_G_B + self.loss_cycle_B # + self.loss_cov_A + self.loss_cov_B
            else:
                self.loss_G = self.loss_G_A + self.loss_cycle_A + self.loss_G_B + self.loss_cycle_B # + self.loss_Hotelling_template
            '''
            
            self.loss_G.requires_grad_(True)
            self.loss_G.backward()
            
            if self.clip_max != 0:
                nn.utils.clip_grad_norm_(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), self.clip_max)
        else:
          with torch.autograd.set_detect_anomaly(False):
            
            self.loss_G_A_test = self.GANLoss(self.netD_A(self.real_B[:,0,:,:].unsqueeze(1).clone().detach().requires_grad_(True)), True) * self.lambda_GA
            self.loss_cycle_B_test = self.criterionCycle2(self.rec_B[:,0,:,:].clone().detach().requires_grad_(True), self.real_B[:,0,:,:].clone().detach().requires_grad_(True)) * self.lambda_B
            self.psnr_B_test = self.compute_PSNR(self.rec_B[:,0,:,:].clone().detach().requires_grad_(True), self.real_B[:,0,:,:].clone().detach().requires_grad_(True))
            self.ssim_B_test = self.compute_SSIM(self.rec_B[:,0,:,:].clone().detach().requires_grad_(True), self.real_B[:,0,:,:].clone().detach().requires_grad_(True))
            self.rmse_B_test = self.compute_RMSE(self.rec_B[:,0,:,:].clone().detach().requires_grad_(True), self.real_B[:,0,:,:].clone().detach().requires_grad_(True))
            self.loss_G_B_test = self.GANLoss(self.netD_B(self.rec_A[:,0,:,:].unsqueeze(1).clone().detach().requires_grad_(True)), False) * self.lambda_GB
            self.loss_cycle_A_test = self.criterionCycle2(self.rec_A[:,0,:,:].clone().detach().requires_grad_(True), self.real_A[:,0,:,:].clone().detach().requires_grad_(True)) * self.lambda_A
            self.psnr_A_test = self.compute_PSNR(self.rec_A[:,0,:,:].clone().detach().requires_grad_(True), self.real_A[:,0,:,:].clone().detach().requires_grad_(True))
            self.ssim_A_test = self.compute_SSIM(self.rec_A[:,0,:,:].clone().detach().requires_grad_(True), self.real_A[:,0,:,:].clone().detach().requires_grad_(True))
            self.rmse_A_test = self.compute_RMSE(self.rec_A[:,0,:,:].clone().detach().requires_grad_(True), self.real_A[:,0,:,:].clone().detach().requires_grad_(True))
            
            self.loss_cov_A_test = self.loss_covariance_matrix_reduction_A(self.rec_A.clone().detach().requires_grad_(True), self.real_A.clone().detach().requires_grad_(True))
            self.loss_cov_B_test = self.loss_covariance_matrix_reduction_B(self.rec_B.clone().detach().requires_grad_(True), self.real_B.clone().detach().requires_grad_(True))
                
            self.loss_G_test = self.loss_G_A_test + self.loss_cycle_A_test + self.loss_G_B_test + self.loss_cycle_B_test + self.loss_cov_A_test + self.loss_cov_B_test

    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    '''
    def forward2(self, training_of_val_phase, index, hypotheses, real_A_lr, real_A, real_B_lr):
        
        fprop_B, fprop_B0 = self.standard_forward_prop(real_A)
        bprop_A = self.standard_backward_prop(real_B_lr, fprop_B0, real_A)
        
        return bprop_A, fprop_B
        #return fake_A, rec_A, fake_B, rec_B
    '''
    
    '''
    def forward(self, training_of_val_phase, index, hypotheses, real_A_lr, real_A, real_B_lr):
        
        fake_A = np.zeros((self.batch_size,2,self.RAYS*self.padding,self.RAYS*self.padding))
        
        for i in range(self.batch_size):
            for j in range(2):
                self.scaler1 = preprocessing.StandardScaler().fit(real_A_lr[i,j,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)])
        
                real_A_n = self.scaler1.transform(real_A_lr[i,j,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)])

                fake_A_n, fake_A_VGG_n = self.netG_B(torch.from_numpy(real_A_n.reshape([1,1,self.RAYS,self.RAYS])).type(torch.double))
        
                fake_A[i,j,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)] = self.scaler1.inverse_transform(fake_A_n[0,0,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)])
        
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(real_A_lr[0,0,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)].detach().numpy())
        ax[1].imshow(fake_A[0,0,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)].detach().numpy())
        plt.savefig('real_vs_fake_A.png')
        plt.close()
        
        print('real_A_lr:', real_A_lr)
        print('fake_A:', fake_A)
        
        fprop_B, fprop_B0 = self.standard_forward_prop(fake_A)
        # fprop_B, fprop_B0 = self.standard_forward_prop(real_A_lr)
        
        self.scaler2 = preprocessing.StandardScaler().fit(fprop_B)
        
        fprop_B = self.scaler2.transform(fprop_B)
        
        rec_B, rec_B_VGG = self.netG_A(fprop_B)
        
        rec_B = self.scaler2.inverse_transform(rec_B)
        
        self.scaler3 = preprocessing.StandardScaler().fit(real_B_lr)
        
        real_B_lr = self.scaler3.transform(real_B_lr)
        
        fake_B, fake_B_VGG = self.netG_A(real_B_lr)
        
        fake_B = self.scaler3.inverse_transform(fake_B)
        
        bprop_A = self.standard_backward_prop(fake_B, fprop_B0, fake_A)
        #bprop_A = self.standard_backward_prop(real_B_lr, fprop_B0, real_A)
        
        self.scaler4 = preprocessing.StandardScaler().fit(bprop_A)
        
        bprop_A = self.scaler4.transform(bprop_A)
        
        rec_A, rec_A_VGG = self.netG_B(bprop_A)
        
        rec_A = self.scaler4.inverse_transform(rec_A)
        
        return fake_A, rec_A, fake_B, rec_B
        #return fake_A, rec_A, fake_B, rec_B
    '''
    
    def forward(self):
        
        self.real_A_lr[:, 0, :, :] = -self.real_A_lr[:, 0, :, :]
        
        self.fake_B, self.fake_B_VGG = self.netG_B(self.real_A_lr)
        
        self.real_A_lr[:, 0, :, :] = -self.real_A_lr[:, 0, :, :]
        
        self.rec_A, self.rec_A_VGG = self.netG_A(self.fake_B)
        
        self.rec_A[:, 0, :, :] = -self.rec_A[:, 0, :, :]
        
        # self.fake_A[:, 0, :, :] = -self.fake_A[:, 0, :, :]
        
        self.fake_A, self.fake_A_VGG = self.netG_A(self.real_B_lr)
        
        self.rec_B, self.rec_B_VGG = self.netG_B(self.fake_A)
        
        self.fake_A[:, 0, :, :] = -self.fake_A[:, 0, :, :]
        
    '''
    def forward(self):
        
        self.real_A_lr[:, 0, :, :] = -self.real_A_lr[:, 0, :, :]
        
        self.fake_A, self.fake_A_VGG = self.netG_B(self.real_A_lr)
        
        self.fake_A[:, 0, :, :] = -self.fake_A[:, 0, :, :]
        
        self.fprop_B, self.fprop_B0 = self.standard_forward_prop(self.fake_A)
        
        self.rec_B, self.rec_B_VGG = self.netG_A(self.fprop_B)
        
        #print('rec_B:', rec_B)
        
        self.fake_B, self.fake_B_VGG = self.netG_A(self.real_B_lr)
        
        self.bprop_A = self.standard_backward_prop(self.fake_B.detach().numpy(), self.fprop_B0, self.fake_A.detach().numpy())
        
        self.bprop_A[:, 0, :, :] = -self.bprop_A[:, 0, :, :]
        
        self.rec_A, self.rec_A_VGG = self.netG_B(self.bprop_A)
        
        self.rec_A[:, 0, :, :] = -self.rec_A[:, 0, :, :] + self.real_A[:, 0, :, :]
        
        # return fake_A, rec_A, fake_B, rec_B
    '''
    
    def optimization(self, phase):
    # def optimization(self, epoch, index, hypotheses, real_A, real_A_lr, real_B_lr, real_B):
        
        #fake_A, rec_A, fake_B, rec_B = self.forward('train', index, hypotheses, real_A, real_A, real_B)
        self.forward()
        
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        
        self.optimizer_D.zero_grad()
        self.optimizer_G.zero_grad()
        
        if phase == 'train':
            self.loss_D_A = self.backward_D(self.netD_A, self.real_B[:,0,:,:].unsqueeze(1), self.rec_B[:,0,:,:].unsqueeze(1), phase)
            self.loss_D_B = self.backward_D(self.netD_B, self.real_A[:,0,:,:].unsqueeze(1), self.rec_A[:,0,:,:].unsqueeze(1), phase)
        else:
            self.loss_D_A_test = self.backward_D(self.netD_A, self.real_B[:,0,:,:].unsqueeze(1), self.rec_B[:,0,:,:].unsqueeze(1), phase)
            self.loss_D_B_test = self.backward_D(self.netD_B, self.real_A[:,0,:,:].unsqueeze(1), self.rec_A[:,0,:,:].unsqueeze(1), phase)
        
        self.optimizer_D.step()
        
        # self.backward_G('train', self.real_A, self.fake_A, self.rec_A, self.real_B, self.fake_B, self.rec_B)
        self.backward_G(phase)
        self.optimizer_G.step()
        
        # return real_A, fake_A, rec_A, real_B, fake_B, rec_B
    
    '''
    def optimization_test(self):
    # def optimization_test(self, epoch, index_test, hypotheses_test, real_A_test, real_A_lr_test, real_B_test, real_B_lr_test):
        
        # fake_A_test, rec_A_test, fake_B_test, rec_B_test = self.forward('test', epoch, index_test, hypotheses_test, real_A_lr_test, real_A_test, real_B_lr_test, real_B_test)
        self.forward('test')
        
        self.loss_D_A_test = self.backward_D(self.netD_A, self.real_B_test[:,0,:,:].unsqueeze(1), self.rec_B_test[:,0,:,:].unsqueeze(1), 'test')
        self.loss_D_B_test = self.backward_D(self.netD_B, self.real_A_test[:,0,:,:].unsqueeze(1), self.rec_A_test[:,0,:,:].unsqueeze(1), 'test')
        # self.backward_G('test', self.real_A_test, self.fake_A_test, self.rec_A_test, self.real_B_test, self.fake_B_test, rec_B_test)
        self.backward_G('test')
        
        #return real_A_test, fake_A_test, rec_A_test, real_B_test, fake_B_test, rec_B_test
    '''
    
    '''
    def optimization2(self, index, hypotheses, real_A, real_A_lr, real_B_lr, real_B):
        
        rec_A, rec_B = self.forward2('train', index, hypotheses, real_A, real_A, real_B)
        
        return rec_A, rec_B
    
    def optimization_test2(self, index_test, hypotheses_test, real_A_test, real_A_test_lr, real_B_test, real_B_test_lr):
        
        rec_A_test, rec_B_test = self.forward2('test', index_test, hypotheses_test, real_A_test, real_B_test, fake_B_test)
        
        return rec_A_test, rec_B_test
    '''
        
    def write_to_stat(self, epoch,iter, fake_B, rec_A, real_B, fake_A, rec_B):
        with open(self.save_stats, "a+") as f:
            f.write('\n -------------------------------------------------------\nEpoch [{}/{}], Step [{}/{}]\n'.format(
                epoch + 1, self.num_epochs, iter + 1, self.total_step))
            for i in range(len(self.img_names)):
                self.print_numpy_to_log(getattr(self,self.img_names[i]).detach().cpu().numpy(), f, self.img_names[i])

    def save_net(self, name,epoch,net,optimizer,loss):
        model_save_name = F'{self.run_name}_{self.model_A}_{self.model_B}_{name}_{epoch}ep.pt'
        path = F"{self.load_run}/save"
        if not os.path.exists(path):
            os.makedirs(path)
        print('saving trained model {} in path: {}'.format(model_save_name, path))
        torch.save({
          'epoch': epoch,
          'model_state_dict': net.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'Hotelling_template_phase': self.weights_A_phase,
          'Hotelling_template_absorption': self.weights_A_absorption,
          'Hotelling_template_I1': self.weights_B_I1,
          'Hotelling_template_I2': self.weights_B_I2,
          'loss': loss}, path+F'/{model_save_name}')

    def print_numpy_to_log(self, x, f, note):
        x = x.astype(np.float64)
        x = x.flatten()
        print('%s:  mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (note,np.mean(x), np.min(x),np.max(x),np.median(x), np.std(x)),file=f)
    
    def visual_iter(self,epoch,iter,model_A, model_B):
        # self.write_to_stat(epoch, iter, fake_B, rec_A, real_B, fake_A, rec_B, real_B_intensity, fake_B_intensity, rec_B_intensity)
        save_name = '{:03d}epoch_{:04d}step'.format(epoch + 1, iter + 1)
        self.plot_cycle(save_name, model_A, model_B, False, 'train', step=iter+1, epoch=epoch)
        # plot_cycle(self, save_name, model_A, model_B, plot_phase=True, phase='train', step=1, epoch=1)
        
    def visual_val(self,epoch,iter, model_A, model_B):
        save_name = '{:03d}epoch_{:02d}step'.format(epoch + 1, iter + 1)
        #self.plot_cycle(0,save_name,index, hypothesis, real_A, fake_B, rec_A, real_B, fake_A, rec_B, model_A, model_B,True,'test', total_num=total_num, step=idx+1, epoch=epoch)
        self.plot_cycle(save_name, model_A, model_B, False, 'test', step=iter+1, epoch=epoch)

    def save_models(self, epoch, phase):
        if phase == 'train':
            self.save_net('netG_A', epoch + 1, self.netG_A, self.optimizer_G, self.loss_G_A)
            self.save_net('netG_B', epoch + 1, self.netG_B, self.optimizer_G, self.loss_G_B)
            self.save_net('netD_A', epoch + 1, self.netD_A, self.optimizer_D, self.loss_D_A)
            self.save_net('netD_B', epoch + 1, self.netD_B, self.optimizer_D, self.loss_D_B)
        else:
            self.save_net('netG_A', epoch + 1, self.netG_A, self.optimizer_G, self.loss_G_A_test)
            self.save_net('netG_B', epoch + 1, self.netG_B, self.optimizer_G, self.loss_G_B_test)
            self.save_net('netD_A', epoch + 1, self.netD_A, self.optimizer_D, self.loss_D_A_test)
            self.save_net('netD_B', epoch + 1, self.netD_B, self.optimizer_D, self.loss_D_B_test)
    
    
    def load_models(self, model, epoch, path, model_load_name1, model_load_name2, model_load_name3, model_load_name4):
        self.netG_A.load_state_dict(torch.load(path+F'{model_load_name1}')['model_state_dict']) 
        self.optimizer_G.load_state_dict(torch.load(path+F'{model_load_name1}')['optimizer_state_dict'])
        self.netD_A.load_state_dict(torch.load(path+F'{model_load_name2}')['model_state_dict'])
        self.netG_B.load_state_dict(torch.load(path+F'{model_load_name3}')['model_state_dict'])
        self.netD_B.load_state_dict(torch.load(path+F'{model_load_name4}')['model_state_dict'])
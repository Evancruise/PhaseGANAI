# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 09:31:07 2022

@author: USER
"""

import numpy as np
import tensorflow as tf
import cv2
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from torch.autograd import Variable
import scipy.io as io

class Fresnel_propagation():
    def __init__(self, opt):
        self.batch_size = 2
        self.z = opt.z
        self.lamda = 3.0996e-11
        self.pi = 3.14
        self.k = 2.0271e+11
        self.RAYS = 128
        self.RAYS_p = 256
        self.padding = 2
        self.delta_x = 15e-6
        self.alpha = 1e-5
        self.I_det_all = torch.zeros((self.batch_size,len(self.z),self.RAYS*self.padding,self.RAYS*self.padding))
        self.I_det_all0 = torch.zeros((self.batch_size,len(self.z),self.RAYS*self.padding,self.RAYS*self.padding))
        self.U = np.zeros((self.batch_size,2,self.RAYS*self.padding,self.RAYS*self.padding))
    
    def forward_propagation_v2(self, U):
        
        #print('U.shape:', np.array(U.detach().numpy()).shape)
        
        # pi,lamda,pxs = [i for i in [self.pi,self.lamda,self.delta_x]]
        
        Kx = torch.arange(-1, 1, 2/self.RAYS_p/self.padding).to(torch.float32) * 0.5/self.delta_x
        Ky = torch.arange(-1, 1, 2/self.RAYS_p/self.padding).to(torch.float32) * 0.5/self.delta_x
        
        Kx,Ky = torch.meshgrid(Kx,Ky)
        Kx = torch.fft.fftshift(Kx)
        Ky = torch.fft.fftshift(Ky)

        fig, ax = plt.subplots(self.batch_size,len(self.z))
        
        U_pad = np.ones((self.batch_size,2,self.RAYS*self.padding,self.RAYS*self.padding))
        
        U_i = np.ones((self.batch_size,2,self.RAYS*self.padding,self.RAYS*self.padding))
        U_ia = np.ones((self.batch_size,2,self.RAYS*self.padding,self.RAYS*self.padding))
        
        U_pad2 = np.ones((self.batch_size,self.RAYS_p*self.padding,self.RAYS_p*self.padding))
        U_pad2a = np.ones((self.batch_size,self.RAYS_p*self.padding,self.RAYS_p*self.padding))
        
        I_det = np.zeros((self.batch_size,len(self.z),self.RAYS_p*self.padding,self.RAYS_p*self.padding))
        I_det0 = np.zeros((self.batch_size,len(self.z),self.RAYS_p*self.padding,self.RAYS_p*self.padding))
            
        I_contact = np.zeros((self.batch_size,1,self.RAYS_p*self.padding,self.RAYS_p*self.padding))
        
        I_det_all = torch.zeros((self.batch_size,len(self.z),self.RAYS*self.padding,self.RAYS*self.padding))
        I_det_all0 = torch.zeros((self.batch_size,len(self.z),self.RAYS*self.padding,self.RAYS*self.padding))
        
        for i in range(self.batch_size):
        
            #ax[i,0].imshow(U[i,0,:,:].detach().numpy())
            #ax[i,1].imshow(U[i,1,:,:].detach().numpy())
            
            U_pad[i,0,:,:] = np.array(U[i,0,:,:].detach().numpy())
            U_pad[i,1,:,:] = np.array(U[i,1,:,:].detach().numpy())
            
            U_i[i,0,:,:] = np.exp(-U_pad[i,1,:,:]+U_pad[i,0,:,:]*1j)
            U_ia[i,0,:,:] = np.exp(-U_pad[i,1,:,:])
            
            U_pad2[i,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)] = U_i[i,0,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)]
            U_pad2[i] = np.fft.fft2(U_pad2[i])

            U_pad2a[i,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)] = U_ia[i,0,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)]
            U_pad2a[i] = np.fft.fft2(U_pad2a[i])
            
            I_contact[i,0,:,:] = abs(U_pad2[i]) ** 2
            
            #ax[i,2].imshow(I_contact[i,0,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)])
            
            for j in range(len(self.z)):
                # Fresnel_P = np.fft.fftshift(torch.exp(-1j * 2 * pi ** 2 * self.z[j] * (Kx ** 2 + Ky ** 2)/k))
                # Filter = exp(-1i.*pi.*lambda.*DO.*(u.^2+v.^2)); 
                Fresnel_P = np.array(torch.exp(-1j * self.pi * self.lamda * self.z[j] * (Kx ** 2 + Ky ** 2)))
                I_det0[i,j] = abs(np.exp(1j * self.k * self.z[j]) * np.fft.ifft2(np.array(U_pad2a[i] * Fresnel_P))) ** 2
                
                I_det_all0[i,j,:,:] = torch.tensor(abs(np.exp(1j * self.k * self.z[j]) * np.fft.ifft2(np.array(U_pad2a[i] * Fresnel_P))) ** 2)[0:int(self.RAYS*self.padding),0:int(self.RAYS*self.padding)]
                #I_det[i,j] = abs(np.exp(1j * self.k * self.z[j]) * np.fft.ifft2(np.array(U_pad2[i] * Fresnel_P))) ** 2
                I_det_all[i,j,:,:] = torch.tensor(abs(np.exp(1j * self.k * self.z[j]) * np.fft.ifft2(np.array(U_pad2[i] * Fresnel_P))) ** 2)[0:int(self.RAYS*self.padding),0:int(self.RAYS*self.padding)]
        
        return I_det_all, I_det_all0
        
    def forward_propagation(self, real_A):
        
        self.batch_size = real_A.shape[0]
        
        Kx = torch.arange(-1, 1, 2/self.RAYS_p/self.padding).to(torch.float32) * 0.5/self.delta_x
        Ky = torch.arange(-1, 1, 2/self.RAYS_p/self.padding).to(torch.float32) * 0.5/self.delta_x

        Kx,Ky = torch.meshgrid(Kx,Ky)
        Kx = torch.fft.fftshift(Kx)
        Ky = torch.fft.fftshift(Ky)

        U_pad = np.ones((self.batch_size,2,self.RAYS*self.padding,self.RAYS*self.padding))
        U_i = np.ones((self.batch_size,2,self.RAYS*self.padding,self.RAYS*self.padding))
        U_ia = np.ones((self.batch_size,2,self.RAYS*self.padding,self.RAYS*self.padding))
        
        U_pad2 = np.ones((self.batch_size,self.RAYS_p*self.padding,self.RAYS_p*self.padding))
        U_pad2a = np.ones((self.batch_size,self.RAYS_p*self.padding,self.RAYS_p*self.padding))
        
        I_det = torch.zeros((self.batch_size,len(self.z),self.RAYS_p*self.padding,self.RAYS_p*self.padding))
        I_det0 = torch.zeros((self.batch_size,len(self.z),self.RAYS_p*self.padding,self.RAYS_p*self.padding))
        I_contact = np.zeros((self.batch_size,1,self.RAYS_p*self.padding,self.RAYS_p*self.padding))
        
        for i in range(self.batch_size):
            phi = real_A[i,0,:,:]
            A = real_A[i,1,:,:]
            U_pad[i,0,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)] = np.array(phi[int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)].detach().numpy())
            U_pad[i,1,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)] = np.array(A[int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)].detach().numpy())
        
            U_i[i,0,:,:] = np.exp(-U_pad[i,1,:,:]+U_pad[i,0,:,:]*1j)
            U_ia[i,0,:,:] = np.exp(-U_pad[i,1,:,:])
            
            U_pad2[i,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)] = U_i[i,0,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)]
            U_pad2[i] = np.fft.fft2(U_pad2[i])

            U_pad2a[i,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)] = U_ia[i,0,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)]
            U_pad2a[i] = np.fft.fft2(U_pad2a[i])
            
            I_contact[i,0,:,:] = abs(U_pad2[i]) ** 2
        
            for j in range(len(self.z)):
                Fresnel_P = np.array(torch.exp(-1j * self.pi * self.lamda * self.z[j] * (Kx ** 2 + Ky ** 2))) # exp(-1i.*pi.*lambda.*DO.*(u.^2+v.^2))

                U_det = np.exp(1j * self.k * self.z[j]) * np.fft.ifft2(np.array(U_pad2[i] * Fresnel_P))
                U_det0 = np.exp(1j * self.k * self.z[j]) * np.fft.ifft2(np.array(U_pad2a[i] * Fresnel_P))
                I_det[i,j,:,:] = torch.tensor(abs(U_det) ** 2)
                I_det0[i,j,:,:] = torch.tensor(abs(U_det0) ** 2)
                #print('I_det[i,j,:,:]:', I_det[i,j,:,:])
                #print('I_det0[i,j,:,:]:', I_det0[i,j,:,:])
        
        return I_det[:,:,0:self.RAYS*self.padding,0:self.RAYS*self.padding], I_det0[:,:,0:self.RAYS*self.padding,0:self.RAYS*self.padding]

    def backward_propagation(self, I, Io, real_A):
        
        self.batch_size = real_A.shape[0]
        
        Kx = torch.arange(-1, 1, 2/self.RAYS/self.padding).to(torch.float32) * 0.5/self.delta_x
        Ky = torch.arange(-1, 1, 2/self.RAYS/self.padding).to(torch.float32) * 0.5/self.delta_x
        
        Kx,Ky = torch.meshgrid(Kx,Ky)
        Kx = torch.fft.fftshift(Kx)
        Ky = torch.fft.fftshift(Ky)
        
        I_pad = torch.zeros((self.batch_size, len(self.z), self.RAYS*self.padding, self.RAYS*self.padding), dtype=torch.complex64)
        I_pad0 = torch.zeros((self.batch_size, len(self.z), self.RAYS*self.padding, self.RAYS*self.padding), dtype=torch.complex64)
        I_ft = torch.zeros((self.batch_size, len(self.z), self.RAYS*self.padding, self.RAYS*self.padding), dtype=torch.complex64)
        I_ft0 = torch.zeros((self.batch_size, len(self.z), self.RAYS*self.padding, self.RAYS*self.padding), dtype=torch.complex64)
        I_contact = np.zeros((self.batch_size, 1, self.RAYS*self.padding, self.RAYS*self.padding))
        
        for i in range(self.batch_size):
            phi = real_A[i,0,:,:]
            A = real_A[i,1,:,:]
            I_contact[i,0,:,:] = abs(np.exp(-A+1j*phi)) ** 2
            
            for j in range(len(self.z)):
                I_pad0[i,j,:,:] = torch.tensor(Io[i,j,0:self.RAYS*self.padding,0:self.RAYS*self.padding])
                I_pad[i,j,:,:] = torch.tensor(I[i,j,0:self.RAYS*self.padding,0:self.RAYS*self.padding])
                I_ft0[i,j,:,:] = torch.fft.fft2(I_pad0[i,j,:,:])
                I_ft[i,j,:,:] = torch.fft.fft2(I_pad[i,j,:,:])
        
        coschirp = torch.zeros((len(self.z), self.RAYS*self.padding, self.RAYS*self.padding), dtype=torch.complex64)
        sinchirp = torch.zeros((len(self.z), self.RAYS*self.padding, self.RAYS*self.padding), dtype=torch.complex64)
        
        for i in range(len(self.z)):
            coschirp[i,:,:] = torch.cos(self.pi*self.lamda*(Kx**2 + Ky**2)*self.z[i])
            sinchirp[i,:,:] = torch.sin(self.pi*self.lamda*(Kx**2 + Ky**2)*self.z[i])
        
        coschirp_dfx = torch.zeros((len(self.z), self.RAYS*self.padding, self.RAYS*self.padding), dtype=torch.complex64)
        coschirp_dfy = torch.zeros((len(self.z), self.RAYS*self.padding, self.RAYS*self.padding), dtype=torch.complex64)
        sumAD2 = torch.zeros((self.RAYS*self.padding, self.RAYS*self.padding), dtype=torch.complex64)
        
        for i in range(len(self.z)):
            sumAD2 += 4 * sinchirp[i,:,:] * sinchirp[i,:,:]
            coschirp_dfx[i,:,:] = coschirp[i,:,:]*1j*Kx*self.lamda*self.z[i]/(2*self.pi) 
            coschirp_dfy[i,:,:] = coschirp[i,:,:]*1j*Ky*self.lamda*self.z[i]/(2*self.pi)
        
        dfxI0 = torch.zeros((self.batch_size, self.RAYS*self.padding, self.RAYS*self.padding))
        dfyI0 = torch.zeros((self.batch_size, self.RAYS*self.padding, self.RAYS*self.padding))
        
        for i in range(self.batch_size):
            dfxI0[i,:,:] = torch.real(torch.fft.ifft2(2*1j*self.pi*Kx*torch.fft.fft2(torch.tensor(I_contact[i,0,0:self.RAYS*self.padding,0:self.RAYS*self.padding]))))
            dfyI0[i,:,:] = torch.real(torch.fft.ifft2(2*1j*self.pi*Ky*torch.fft.fft2(torch.tensor(I_contact[i,0,0:self.RAYS*self.padding,0:self.RAYS*self.padding]))))
        
        phase = torch.zeros((self.batch_size, 1, self.RAYS*self.padding, self.RAYS*self.padding))
        absorption = torch.zeros((self.batch_size, 1, self.RAYS*self.padding, self.RAYS*self.padding))
        phase_pad = torch.zeros((self.batch_size, self.RAYS*self.padding,self.RAYS*self.padding), dtype=torch.complex64)
        
        #fig, ax = plt.subplots(3,self.batch_size)
        
        for i in range(self.batch_size):
            for n in range(5):
                nominator_term = torch.zeros((self.RAYS*self.padding,self.RAYS*self.padding), dtype=torch.complex64)
                phase_pad[i, int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)] = phase[i,0,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)]
                phase_dfxI0 = torch.fft.fft2(phase_pad[i,:,:]*dfxI0[i,:,:])
                phase_dfyI0 = torch.fft.fft2(phase_pad[i,:,:]*dfyI0[i,:,:])
                    
                for j in range(len(self.z)):
                    nominator_term += 2*sinchirp[j,:,:]*(I_ft[i,j,:,:]-I_ft0[i,j,:,:]-coschirp_dfx[j,:,:]*phase_dfxI0-coschirp_dfy[j,:,:]*phase_dfyI0)
                    
                phase_I_n = nominator_term/(sumAD2+self.alpha)
                ph = torch.real(torch.fft.ifft2(phase_I_n))
                phase[i,0,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)] = ph[int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)]/torch.tensor(I_contact[i,0,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)])
                
            absorption[i,0,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)] = -1/2 * torch.log(torch.tensor(I_contact[i,0,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)]))
            
            #ax[0,i].imshow(np.array(I_contact[i,0,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)]))
            #ax[1,i].imshow(np.array(phase[i,0,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)].detach().numpy()))
            #ax[2,i].imshow(np.array(absorption[i,0,int(self.RAYS/2):int(self.RAYS*3/2),int(self.RAYS/2):int(self.RAYS*3/2)].detach().numpy()))
            
        #plt.savefig('reconstructed_pb.png')
        #plt.close()
        
        phase_and_absorption = torch.cat((phase, absorption), axis=1)
        
        return phase_and_absorption
    def roll_n(self,X, axis, n):
        f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
        b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
        front = X[f_idx]
        back = X[b_idx]
        return torch.cat([back, front], axis)

    def batch_fftshift2d(self,x):
        # Provided by PyTorchSteerablePyramid
        real, imag = torch.unbind(x, -1)
        for dim in range(1, len(real.size())):
            n_shift = real.size(dim)//2
            if real.size(dim) % 2 != 0:
                n_shift += 1  # for odd-sized images
            real = self.roll_n(real, axis=dim, n=n_shift)
            imag = self.roll_n(imag, axis=dim, n=n_shift)
        return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

    def batch_ifftshift2d(self, x):
        real, imag = torch.unbind(x, -1)
        for dim in range(len(real.size()) - 1, 0, -1):
            real = self.roll_n(real, axis=dim, n=real.size(dim)//2)
            imag = self.roll_n(imag, axis=dim, n=imag.size(dim)//2)
        return torch.stack((real, imag), -1)  # last dim=2 (real&imag)
'''
RAYS = 128
padding = 2

datadir = 'dataset/IMGS_phase_contrast_E_40'
lhr_p = ['phi_99_mixed_approach_SKE_BKSc_H0_128x128.mat', 'phi_99_mixed_approach_SKE_BKSc_H0_128x128.mat']
lhr_b = ['b_99_mixed_approach_SKE_BKSc_H0_128x128.mat', 'b_99_mixed_approach_SKE_BKSc_H0_128x128.mat']
lhr_i = ['I_99_mixed_approach_SKE_BKSc_H0_128x128.mat', 'I_99_mixed_approach_SKE_BKSc_H0_128x128.mat']
real_A = np.zeros((len(lhr_p), 2, RAYS*padding, RAYS* padding))
real_B = np.zeros((len(lhr_p), 2, RAYS*padding, RAYS* padding))

for i in range(len(lhr_p)):
    phi = io.loadmat(datadir + "/phi/" + lhr_p[i])['phi_proj']
    A = io.loadmat(datadir + "/b/" + lhr_b[i])['A_proj']
    I1 = io.loadmat(datadir + "/intensity/" + lhr_i[i])['I1']
    I2 = io.loadmat(datadir + "/intensity/" + lhr_i[i])['I2']
    real_A[i,0,:,:] = phi
    real_A[i,1,:,:] = A
    real_B[i,0,:,:] = I1
    real_B[i,1,:,:] = I2
    # = np.concatenate((np.reshape(phi, (len(lhr_p), 1, RAYS*padding, RAYS* padding)), np.reshape(A, (len(lhr_b), 1, RAYS*padding, RAYS* padding))), axis=1)

I, Io = Fresnel_propagation().forward_propagation(real_A)

fig, ax = plt.subplots(len(lhr_p), 4)

if len(lhr_p) == 1:
    ax[0].imshow(I[0,0,int(RAYS/2):int(RAYS*3/2),int(RAYS/2):int(RAYS*3/2)].reshape((RAYS,RAYS)))
    ax[1].imshow(Io[0,1,int(RAYS/2):int(RAYS*3/2),int(RAYS/2):int(RAYS*3/2)].reshape((RAYS,RAYS)))
    ax[2].imshow(real_B[0,0,int(RAYS/2):int(RAYS*3/2),int(RAYS/2):int(RAYS*3/2)].reshape((RAYS,RAYS)))
    ax[3].imshow(real_B[0,1,int(RAYS/2):int(RAYS*3/2),int(RAYS/2):int(RAYS*3/2)].reshape((RAYS,RAYS)))
else:
    for i in range(len(lhr_p)):
        ax[i,0].imshow(I[i,0,int(RAYS/2):int(RAYS*3/2),int(RAYS/2):int(RAYS*3/2)].reshape((RAYS,RAYS)))
        ax[i,1].imshow(Io[i,1,int(RAYS/2):int(RAYS*3/2),int(RAYS/2):int(RAYS*3/2)].reshape((RAYS,RAYS)))
        ax[i,2].imshow(real_B[i,0,int(RAYS/2):int(RAYS*3/2),int(RAYS/2):int(RAYS*3/2)].reshape((RAYS,RAYS)))
        ax[i,3].imshow(real_B[i,1,int(RAYS/2):int(RAYS*3/2),int(RAYS/2):int(RAYS*3/2)].reshape((RAYS,RAYS)))

plt.savefig('I.png')
plt.close()

rec_B = Fresnel_propagation().backward_propagation(real_B, Io, real_A)

print('rec_B:', rec_B)

fig, ax = plt.subplots(len(lhr_p), 2)

if len(lhr_p) == 1:
    ax[0].imshow(rec_B[0,0,:,:].reshape((RAYS,RAYS)))
    ax[1].imshow(rec_B[0,1,:,:].reshape((RAYS,RAYS)))
else:
    for i in range(len(lhr_p)):
        ax[i,0].imshow(rec_B[i,0,:,:].reshape((RAYS,RAYS)))
        ax[i,1].imshow(rec_B[i,1,:,:].reshape((RAYS,RAYS)))

plt.savefig('phase_absorption_rec.png')
plt.close()
'''
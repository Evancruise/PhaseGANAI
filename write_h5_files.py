# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 10:16:40 2022

@author: USER
"""

import os
import numpy as np
import scipy.io as io 
import h5py
# import matplotlib.pyplot as plt
from tqdm import tqdm

data_dir = 'case1/'
datadir_h5py = data_dir + 'IMGS_PCI/'

RAYS = 128
crop = np.linspace(int(RAYS/2)+1,int(RAYS*3/2),RAYS)
print('crop:', len(crop))

if not os.path.exists(datadir_h5py):
    os.mkdir(datadir_h5py)

parameters = io.loadmat(data_dir + 'Data0_UserDefinedParameters.mat')
RAYS = int(parameters['RAYS'][0])
d = parameters['d'][0]
data_files_dir = str(parameters['data_files_dir'][0])
recon_alg = str(parameters['recon_alg'][0])
No_realizations = int(parameters['No_realizations'][0])
sigma0 = float(parameters['sigma0'][0])
image_type = str(parameters['image_type'][0])
crop = parameters['crop'][0]
print(image_type)

if image_type == 'BKS':
    for i in tqdm(range(1,No_realizations+1)):
        profiles = io.loadmat(data_dir + data_files_dir + 'PCI_{}_{}_{}_{}.mat'.format(i, sigma0, image_type, recon_alg))
        phi_proj = profiles['phi_proj']
        A_proj = profiles['A_proj']
        phi = profiles['phi']
        A = profiles['A']
        
        I_matrix = np.zeros((RAYS,RAYS,len(d),2))
        I = profiles['I']
        #fig, ax = plt.subplots(len(d),2)
        for j in range(len(d)):
            #ax[j,0].imshow(I[j,0][int(crop[0]):int(crop[-1]), int(crop[0]):int(crop[-1])])
            #ax[j,1].imshow(I[j,1][int(crop[0]):int(crop[-1]), int(crop[0]):int(crop[-1])])
            I_matrix[:,:,j,0] = I[0,j][int(crop[0])-1:int(crop[-1]), int(crop[0])-1:int(crop[-1])]
            I_matrix[:,:,j,1] = I[1,j][int(crop[0])-1:int(crop[-1]), int(crop[0])-1:int(crop[-1])]
        
        #plt.show()
        #plt.close()
        
        hf = h5py.File(datadir_h5py + 'PCI_{}_{}_H{}_{}.h5'.format(image_type,i,0,recon_alg), 'w')
        g1 = hf.create_group('ph')
        g1.create_dataset('phi_proj', data=phi_proj)
        g1.create_dataset('A_proj', data=A_proj)
        g1.create_dataset('phi', data=phi)
        g1.create_dataset('A', data=A)      

        for j in range(len(d)):
            g1.create_dataset('I{}'.format(j), data=I_matrix[:,:,j,0])
            g1.create_dataset('I{}o'.format(j), data=I_matrix[:,:,j,1])
        
        g1.create_dataset('i', data=i)
        g1.create_dataset('hypothesis', data='H0')
        g1.create_dataset('image_type', data=image_type)
        g1.create_dataset('recon_alg', data=recon_alg)
        hf.close()
        
elif image_type == 'SKE':
    for i in tqdm(range(1,No_realizations+1)):
        profiles = io.loadmat(data_dir + data_files_dir + 'PCI_{}_{}_H1_{}_{}.mat'.format(i, sigma0, image_type, recon_alg))
        phi_proj = profiles['phi_proj']
        A_proj = profiles['A_proj']
        phi = profiles['phi']
        A = profiles['A']
        
        '''
        fig, ax = plt.subplots(2,2)
        ax[0,0].imshow(phi_proj)
        ax[0,1].imshow(A_proj)
        ax[1,0].imshow(phi)
        ax[1,1].imshow(A)
        plt.show()
        plt.close()
        '''
        
        I_matrix = np.zeros((RAYS,RAYS,len(d),2))
        I = profiles['I']
        
        #fig, ax = plt.subplots(len(d),2)
        for j in range(len(d)):
            #ax[j,0].imshow(I[j,0][int(crop[0]):int(crop[-1]), int(crop[0]):int(crop[-1])])
            #ax[j,1].imshow(I[j,1][int(crop[0]):int(crop[-1]), int(crop[0]):int(crop[-1])])
            I_matrix[:,:,j,0] = I[0,j][int(crop[0])-1:int(crop[-1]), int(crop[0])-1:int(crop[-1])]
            I_matrix[:,:,j,1] = I[1,j][int(crop[0])-1:int(crop[-1]), int(crop[0])-1:int(crop[-1])]
        
        
        #plt.show()
        #plt.close()
        
        hf = h5py.File(datadir_h5py + 'PCI_{}_{}_H1_{}.h5'.format(image_type,i,recon_alg), 'w')
        g1 = hf.create_group('ph')
        g1.create_dataset('phi_proj', data=phi_proj)
        g1.create_dataset('A_proj', data=A_proj)
        g1.create_dataset('phi', data=phi)
        g1.create_dataset('A', data=A)      

        for j in range(len(d)):
            g1.create_dataset('I{}'.format(j), data=I_matrix[:,:,j,0])
            g1.create_dataset('I{}o'.format(j), data=I_matrix[:,:,j,1])
        
        g1.create_dataset('i', data=i)
        g1.create_dataset('hypothesis', data='H1')
        g1.create_dataset('image_type', data=image_type)
        g1.create_dataset('recon_alg:', data=recon_alg)
        hf.close()
        
    
elif image_type == 'SKEs':
    No_signals = int(parameters['No_signals'])
    
    for si in tqdm(range(1,No_signals+1)):    
        for i in range(1,No_realizations+1):
            profiles = io.loadmat(data_dir + data_files_dir + 'PCI_{}_{}_H{}_{}_{}.mat'.format(i, sigma0, si, image_type, recon_alg))
            phi_proj = profiles['phi_proj']
            A_proj = profiles['A_proj']
            phi = profiles['phi']
            A = profiles['A']
            
            '''
            fig, ax = plt.subplots(2,2)
            ax[0,0].imshow(phi_proj)
            ax[0,1].imshow(A_proj)
            ax[1,0].imshow(phi)
            ax[1,1].imshow(A)
            plt.show()
            plt.close()
            '''
            
            I_matrix = np.zeros((RAYS,RAYS,len(d),2))
            I = profiles['I']
            
            #fig, ax = plt.subplots(len(d),2)
            for j in range(len(d)):
                #ax[j,0].imshow(I[j,0][int(crop[0]):int(crop[-1]), int(crop[0]):int(crop[-1])])
                #ax[j,1].imshow(I[j,1][int(crop[0]):int(crop[-1]), int(crop[0]):int(crop[-1])])
                I_matrix[:,:,j,0] = I[0,j][int(crop[0])-1:int(crop[-1]), int(crop[0])-1:int(crop[-1])]
                I_matrix[:,:,j,1] = I[1,j][int(crop[0])-1:int(crop[-1]), int(crop[0])-1:int(crop[-1])]
            
            
            #plt.show()
            #plt.close()
            
            hf = h5py.File(datadir_h5py + 'PCI_{}_{}_H{}_{}.h5'.format(image_type,i,si,recon_alg), 'w')
            g1 = hf.create_group('ph')
            g1.create_dataset('phi_proj', data=phi_proj)
            g1.create_dataset('A_proj', data=A_proj)
            g1.create_dataset('phi', data=phi)
            g1.create_dataset('A', data=A)      

            for j in range(len(d)):
                g1.create_dataset('I{}'.format(j), data=I_matrix[:,:,j,0])
                g1.create_dataset('I{}o'.format(j), data=I_matrix[:,:,j,1])
            
            g1.create_dataset('i', data=i)
            g1.create_dataset('hypothesis', data='H{}'.format(si))
            g1.create_dataset('image_type', data=image_type)
            g1.create_dataset('recon_alg', data=recon_alg)
            hf.close()


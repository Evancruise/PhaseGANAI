%% Generate noiseless H0 and H1 intensity realizations
% warning off; clear all; close all; clc; pack;

%function Step4_AcquireRealizationsFromPhaseGAN(data_dir,det,image_type)

addpath('utils/');

tic;
data_dir = 'case1/';

% Cheng-Ying Chou, 2022/05/06

% data_dir='case/';det1=1; det2=2; image_type='SKE', 'SKEs', 'SKS', or 'BKS';
load([data_dir,'Data0_UserDefinedParameters.mat']);

switch image_type
    case 'BKS'
    % H0: generate BKS: clustered lumpy background
    eval(sprintf('phi1%db = zeros(RAYS,RAYS,No_signals*No_realizations);',det2))
    eval(sprintf('b1%db = zeros(RAYS,RAYS,No_signals*No_realizations);',det2))
    I1b=zeros(RAYS,RAYS,No_signals*No_realizations);
    I2b=zeros(RAYS,RAYS,No_signals*No_realizations);
    %I1bo=zeros(RAYS,RAYS,No_realizations);
    %I2bo=zeros(RAYS,RAYS,No_realizations);
    
    for i=1: No_signals*No_realizations
        load([data_dir,data_files_dir,sprintf('PCI_%d_%04g_H0_%s_%s_phasegan.mat',i,sigma0,image_type,recon_alg)], 'phi_proj', 'phi', 'A_proj', 'A', 'I');
        eval(sprintf('phi1%db = phi;', det2));
        eval(sprintf('b1%db = b;', det2));
        for i_det = 1:length(d)
            eval(sprintf('I%db = I{1,i_det};',i_det));
        end
    end
    
    fid=fopen([data_dir,data_phi_dir,sprintf('phi1%d_b_%04g_H0_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)],'wb'); fwrite(fid,eval(sprintf('phi1%db',det2)),'double'); fclose(fid);
    fid=fopen([data_dir,data_b_dir,sprintf('b1%d_b_%04g_H0_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)],'wb'); fwrite(fid,eval(sprintf('b1%db',det2)),'double'); fclose(fid);
    for i_det = 1:length(d)
        fid=fopen([data_dir,data_I_dir,sprintf('I%d_b_%04g_H0_%s_%s_phasegan.dat',i_det,sigma0,image_type,recon_alg)],'wb'); fwrite(fid,eval(sprintf('I%db',i_det)),'double'); fclose(fid);
    end
    
    case 'SKE'
        
    % H1: generate SKE
    eval(sprintf('phi1%ds = zeros(RAYS,RAYS,No_realizations);',det2))
    eval(sprintf('b1%ds = zeros(RAYS,RAYS,No_realizations);',det2))
    I1s=zeros(RAYS,RAYS,No_realizations);
    I2s=zeros(RAYS,RAYS,No_realizations);
           
    for i=1: No_realizations
        load([data_dir,data_files_dir,sprintf('PCI_%d_%04g_H1_%s_%s_phasegan.mat',i,sigma0,image_type,recon_alg)], 'phi_proj', 'phi', 'A_proj', 'A', 'I');
        eval(sprintf('phi1%ds = phi;', det2));
        eval(sprintf('b1%ds = b;', det2));
        for i_det = 1:length(d)
            eval(sprintf('I%ds = I{1,i_det};',i_det));
        end
    end
    
    fid=fopen([data_dir,data_phi_dir,sprintf('phi1%d_s_%04g_H1_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)],'wb'); fwrite(fid,eval(sprintf('phi1%ds',det2)),'double'); fclose(fid);
    fid=fopen([data_dir,data_b_dir,sprintf('b1%d_s_%04g_H1_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)],'wb'); fwrite(fid,eval(sprintf('b1%ds',det2)),'double'); fclose(fid);
    for i_det = 1:length(d)
        fid=fopen([data_dir,data_I_dir,sprintf('I%d_s_%04g_H1_%s_%s_phasegan.dat',i_det,sigma0,image_type,recon_alg)],'wb'); fwrite(fid,eval(sprintf('I%ds',i_det)),'double'); fclose(fid);
    end
    
    case 'SKEs'
    % H1: generate SKE
    eval(sprintf('phi1%ds = zeros(RAYS,RAYS,No_signals*No_realizations);',det2))
    eval(sprintf('b1%ds = zeros(RAYS,RAYS,No_signals*No_realizations);',det2))
    I1s=zeros(RAYS,RAYS,No_signals*No_realizations);
    I2s=zeros(RAYS,RAYS,No_signals*No_realizations);
        
    for si=1: No_signals
        for i=No_realizations*(si-1)+1: No_realizations*si
            load([data_dir,data_files_dir,sprintf('PCI_%d_%04g_H1_%s_%s_phasegan.mat',i,sigma0,image_type,recon_alg)], 'phi_proj', 'phi', 'A_proj', 'A', 'I');
            eval(sprintf('phi1%ds = phi;', det2));
            eval(sprintf('b1%ds = b;', det2));
            for i_det = 1:length(d)
                eval(sprintf('I%ds = I{1,i_det};',i_det));
            end
        end
    end
    fid=fopen([data_dir,data_phi_dir,sprintf('phi1%d_s_%04g_H1_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)],'wb'); fwrite(fid,eval(sprintf('phi1%ds',det2)),'double'); fclose(fid);
    fid=fopen([data_dir,data_b_dir,sprintf('b1%d_s_%04g_H1_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)],'wb'); fwrite(fid,eval(sprintf('b1%ds',det2)),'double'); fclose(fid);
    for i_det = 1:length(d)
        fid=fopen([data_dir,data_I_dir,sprintf('I%d_s_%04g_H1_%s_%s_phasegan.dat',i_det,sigma0,image_type,recon_alg)],'wb'); fwrite(fid,eval(sprintf('I%ds',i_det)),'double'); fclose(fid);
    end
end




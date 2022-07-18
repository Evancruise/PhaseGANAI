%% Calculate covariance matrices of reconstructed wavefield (phase and absorption) and intesities
% warning off; clear all; close all; clc; pack;

%function Step5_CalculateCovofRealizationsFromPhaseGAN(data_dir,det,image_type)

addpath('utils/');

tic;
data_dir = 'case1/';

% Cheng-Ying Chou, 2022/05/06

% data_dir='case/';det1=1; det2=2; image_type='SKE', 'SKEs', 'SKS', or 'BKS';
load([data_dir,'Data0_UserDefinedParameters.mat']);

crop = RAYS/2+1:RAYS*3/2;  % Suggest not to crop for SKEs
  
switch image_type
    case 'BKS'
        
        sum_phi12xyb=zeros(RAYS^2);
        sum_b12xyb=zeros(RAYS^2);
        sum_phib12xyb=zeros(RAYS^2);
        
        sum_I1xyb=zeros(RAYS^2);
        sum_I2xyb=zeros(RAYS^2);
        sum_I12xyb=zeros(RAYS^2);
        
        % Read phi, b, I1, I2 realizations for H0
        fid=fopen([data_dir,data_phi_dir,sprintf('phi1%d_b_%04g_H0_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)],'rb'); phi12b = fread(fid,[RAYS*RAYS,No_signals*No_realizations],'double'); fclose(fid);
        fid=fopen([data_dir,data_b_dir,sprintf('b1%d_b_%04g_H0_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)],'rb'); b12b = fread(fid,[RAYS*RAYS,No_signals*No_realizations],'double'); fclose(fid);
        [sum_phi12xyb, sum_b12xyb, sum_phib12xyb] = ComputeCovariance(sum_phi12xyb, sum_b12xyb, sum_phib12xyb,phi12b,b12b,No_signals*No_realizations);
        
        phi12b = reshape(phi12b, [RAYS,RAYS,No_signals*No_realizations]);
        b12b = reshape(b12b, [RAYS,RAYS,No_signals*No_realizations]);
        sum_phi12b=sum(phi12b,3); % 3-D dimension summation
        sum_b12b=sum(b12b,3);
        cov_phi12b = (sum_phi12xyb-sum_phi12b(:)*transpose(sum_phi12b(:))/(No_realizations))/((No_realizations)-1);
        cov_b12b = (sum_b12xyb-sum_b12b(:)*transpose(sum_b12b(:))/(No_realizations))/((No_realizations)-1);
        cov_phib12b = (sum_phib12xyb-sum_phi12b(:)*transpose(sum_b12b(:))/(No_realizations))/((No_realizations)-1);
        
        fid=fopen([data_dir,data_phi_dir,sprintf('cov_phi1%db_%04g_H0_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)],'wb'); 
        fwrite(fid, eval(sprintf('cov_phi1%db',det2)),'double');
        fclose(fid); clear cov_phi12b
        fid=fopen([data_dir,data_b_dir,sprintf('cov_b1%db_%04g_H0_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)],'wb'); 
        fwrite(fid, eval(sprintf('cov_b1%db',det2)),'double');
        fclose(fid); clear cov_b12b
        fid=fopen([data_dir,data_phib_dir,sprintf('cov_phib1%db_%04g_H0_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)],'wb'); 
        fwrite(fid, eval(sprintf('cov_phib1%db',det2)),'double');
        fclose(fid); clear cov_phib12b

        fid=fopen([data_dir,data_I_dir,sprintf('I%d_b_%04g_H0_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)],'rb'); I1b = fread(fid,[RAYS*RAYS,No_signals*No_realizations],'double'); fclose(fid);
        fid=fopen([data_dir,data_I_dir,sprintf('I%d_b_%04g_H0_%s_%s_phasegan.dat',det1,sigma0,image_type,recon_alg)],'rb'); I2b = fread(fid,[RAYS*RAYS,No_signals*No_realizations],'double'); fclose(fid);
        [sum_I1xyb, sum_I2xyb, sum_I12xyb] = ComputeCovariance(sum_I1xyb, sum_I2xyb, sum_I12xyb,I1b,I2b,No_signals*No_realizations);
        
        I1b = reshape(I1b, [RAYS,RAYS,No_signals*No_realizations]);
        I2b = reshape(I2b, [RAYS,RAYS,No_signals*No_realizations]);
        sum_I1b=sum(I1b,3); % 3-D dimension summation
        sum_I2b=sum(I2b,3);
        
        cov_I1b = (sum_I1xyb-sum_I1b(:)*transpose(sum_I1b(:))/(No_realizations))/((No_realizations)-1);
        cov_I2b = (sum_I2xyb-sum_I2b(:)*transpose(sum_I2b(:))/(No_realizations))/((No_realizations)-1);
        cov_I12b = (sum_I12xyb-sum_I1b(:)*transpose(sum_I2b(:))/(No_realizations))/((No_realizations)-1);
        
        fid=fopen([data_dir,data_I_dir,sprintf('cov_I%db_%04g_H0_%s_%s_phasegan.dat',det1,sigma0,image_type,recon_alg)],'wb'); 
        fwrite(fid,eval(sprintf('cov_I%db',det1)),'double'); 
        fclose(fid); clear cov_I1b
        fid=fopen([data_dir,data_I_dir,sprintf('cov_I%db_%04g_H0_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)],'wb'); 
        fwrite(fid,eval(sprintf('cov_I%db',det2)),'double'); 
        fclose(fid); clear cov_I2b
        fid=fopen([data_dir,data_I_dir,sprintf('cov_I1%db_%04g_H0_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)],'wb'); 
        fwrite(fid,eval(sprintf('cov_I1%db',det2)),'double'); 
        fclose(fid); clear cov_I12b
        
    case 'SKE'
        
        sum_phi12xys=zeros(RAYS^2);
        sum_b12xys=zeros(RAYS^2);
        sum_phib12xys=zeros(RAYS^2);
        
        sum_I1xys=zeros(RAYS^2);
        sum_I2xys=zeros(RAYS^2);
        sum_I12xys=zeros(RAYS^2);
        
        % Read phi, b, I1, I2 realizations for H1
        fid=fopen([data_dir,data_phi_dir,sprintf('phi1%d_s_%04g_H1_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)],'rb'); phi12s = fread(fid,[RAYS*RAYS,No_realizations],'double'); fclose(fid);
        fid=fopen([data_dir,data_b_dir,sprintf('b1%d_s_%04g_H1_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)],'rb'); b12s = fread(fid,[RAYS*RAYS,No_realizations],'double'); fclose(fid);
        
        [sum_phi12xys,sum_b12xys,sum_phib12xys] = ComputeCovariance(sum_phi12xys,sum_b12xys,sum_phib12xys,phi12s,b12s,No_realizations);
        
        phi12s = reshape(phi12s, [RAYS,RAYS,No_realizations]);
        b12s = reshape(b12s, [RAYS,RAYS,No_realizations]);
        sum_phi12s=sum(phi12s,3); % 3-D dimension summation
        sum_b12s=sum(b12s,3);

        cov_phi12s = (sum_phi12xys-sum_phi12s(:)*transpose(sum_phi12s(:))/(No_realizations))/((No_realizations)-1);
        cov_b12s = (sum_b12xys-sum_b12s(:)*transpose(sum_b12s(:))/(No_realizations))/((No_realizations)-1);
        cov_phib12s = (sum_phib12xys-sum_phi12s(:)*transpose(sum_b12s(:))/(No_realizations))/((No_realizations)-1);

        fid=fopen([data_dir,data_phi_dir,sprintf('cov_phi1%ds_%04g_H1_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)],'wb'); 
        fwrite(fid, eval(sprintf('cov_phi1%ds',det2)),'double');
        fclose(fid); clear cov_phi12s
        fid=fopen([data_dir,data_b_dir,sprintf('cov_b1%ds_%04g_H1_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)],'wb'); 
        fwrite(fid, eval(sprintf('cov_b1%ds',det2)),'double');
        fclose(fid); clear cov_b12s
        fid=fopen([data_dir,data_phib_dir,sprintf('cov_phib1%ds_%04g_H1_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)],'wb'); 
        fwrite(fid, eval(sprintf('cov_phib1%ds',det2)),'double');
        fclose(fid); clear cov_phib12s
        
        fid=fopen([data_dir,data_I_dir,sprintf('I%d_s_%04g_H1_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)],'rb'); I1s = fread(fid,[RAYS*RAYS,No_realizations],'double'); fclose(fid);
        fid=fopen([data_dir,data_I_dir,sprintf('I%d_s_%04g_H1_%s_%s_phasegan.dat',det1,sigma0,image_type,recon_alg)],'rb'); I2s = fread(fid,[RAYS*RAYS,No_realizations],'double'); fclose(fid);
        
        [sum_I1xys,sum_I2xys,sum_I12xys] = ComputeCovariance(sum_I1xys,sum_I2xys,sum_I12xys,I1s,I2s,No_realizations);
        
        I1s = reshape(I1s, [RAYS,RAYS,No_realizations]);
        I2s = reshape(I2s, [RAYS,RAYS,No_realizations]);
        sum_I1s=sum(I1s,3); % 3-D dimension summation
        sum_I2s=sum(I2s,3);

        cov_I1s = (sum_I1xys-sum_I1s(:)*transpose(sum_I1s(:))/(No_realizations))/((No_realizations)-1);
        cov_I2s = (sum_I2xys-sum_I2s(:)*transpose(sum_I2s(:))/(No_realizations))/((No_realizations)-1);
        cov_I12s = (sum_I12xys-sum_I1s(:)*transpose(sum_I2s(:))/(No_realizations))/((No_realizations)-1);
        
        fid=fopen([data_dir,data_I_dir,sprintf('cov_I%ds_%04g_H1_%s_%s_phasegan.dat',det1,sigma0,image_type,recon_alg)],'wb'); 
        fwrite(fid,eval(sprintf('cov_I%ds',det1)),'double'); 
        fclose(fid); clear cov_I1s
        fid=fopen([data_dir,data_I_dir,sprintf('cov_I%ds_%04g_H1_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)],'wb'); 
        fwrite(fid,eval(sprintf('cov_I%ds',det2)),'double'); 
        fclose(fid); clear cov_I2s
        fid=fopen([data_dir,data_I_dir,sprintf('cov_I1%ds_%04g_H1_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)],'wb'); 
        fwrite(fid,eval(sprintf('cov_I1%ds',det2)),'double'); 
        fclose(fid); clear cov_I12s
        
    case 'SKEs'
        sum_phi12xys=zeros(RAYS^2);
        sum_b12xys=zeros(RAYS^2);
        sum_phib12xys=zeros(RAYS^2);
        
        fid=fopen([data_dir,data_phi_dir,sprintf('phi1%d_s_%04g_H1_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)],'rb'); phi12s = fread(fid,[RAYS*RAYS,No_signals*No_realizations],'double'); fclose(fid);
        fid=fopen([data_dir,data_b_dir,sprintf('b1%d_s_%04g_H1_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)],'rb'); b12s = fread(fid,[RAYS*RAYS,No_signals*No_realizations],'double'); fclose(fid);
            
        [sum_phi12xys, sum_b12xys, sum_phib12xys] = ComputeCovariance(sum_phi12xys,sum_b12xys,sum_phib12xys,phi12s,b12s,No_signals*No_realizations);
        
        phi12s = reshape(phi12s, [RAYS,RAYS,No_signals*No_realizations]);
        b12s = reshape(b12s, [RAYS,RAYS,No_signals*No_realizations]);
        sum_phi12s=sum(phi12s,3); % 3-D dimension summation
        sum_b12s=sum(b12s,3);

        cov_phi12s = (sum_phi12xys-sum_phi12s(:)*transpose(sum_phi12s(:))/(No_signals*No_realizations))/((No_signals*No_realizations)-1);
        cov_b12s = (sum_b12xys-sum_b12s(:)*transpose(sum_b12s(:))/(No_signals*No_realizations))/((No_signals*No_realizations)-1);
        cov_phib12s = (sum_phib12xys-sum_phi12s(:)*transpose(sum_b12s(:))/(No_signals*No_realizations))/((No_signals*No_realizations)-1);

        % Save H1 phi covariance
        fid=fopen([data_dir,data_phi_dir,sprintf('cov_phi1%ds_%04g_H1_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)],'wb'); 
		fwrite(fid, eval(sprintf('cov_phi1%ds',det2)),'double');
		fclose(fid); clear cov_phi12s;
			
        % Save H1 b covariance
		fid=fopen([data_dir,data_b_dir,sprintf('cov_b1%ds_%04g_H1_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)],'wb'); 
		fwrite(fid, eval(sprintf('cov_b1%ds',det2)),'double');
		fclose(fid); clear cov_b12s;

        % Save H1 phi_b covariance
        fid=fopen([data_dir,data_phib_dir,sprintf('cov_phib1%ds_%04g_H1_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)],'wb'); 
		fwrite(fid, eval(sprintf('cov_phib1%ds',det2)),'double');
		fclose(fid); clear cov_phib12s;
        
        sum_I1xys=zeros(RAYS^2);
        sum_I2xys=zeros(RAYS^2);
        sum_I12xys=zeros(RAYS^2);
        phi12s = zeros(RAYS^2,No_signals*No_realizations);
        b12s = zeros(RAYS^2,No_signals*No_realizations);
        phib12s = zeros(RAYS^2,No_signals*No_realizations);
        
        fid=fopen([data_dir,data_I_dir,sprintf('I%d_s_%04g_H1_%s_%s_phasegan.dat',det1,sigma0,image_type,recon_alg)],'rb'); I1s = fread(fid,[RAYS*RAYS,No_signals*No_realizations],'double'); fclose(fid);
        fid=fopen([data_dir,data_I_dir,sprintf('I%d_s_%04g_H1_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)],'rb'); I2s = fread(fid,[RAYS*RAYS,No_signals*No_realizations],'double'); fclose(fid);
            
        [sum_I1xys, sum_I2xys, sum_I12xys] = ComputeCovariance(sum_I1xys,sum_I2xys,sum_I12xys,I1s,I2s,No_signals*No_realizations);        
        
        I1s = reshape(I1s, [RAYS,RAYS,No_signals*No_realizations]);
        I2s = reshape(I2s, [RAYS,RAYS,No_signals*No_realizations]);
        sum_I1s=sum(I1s,3); % 3-D dimension summation
        sum_I2s=sum(I2s,3);

        cov_I1s = (sum_I1xys-sum_I1s(:)*transpose(sum_I1s(:))/(No_signals*No_realizations))/((No_signals*No_realizations)-1);
        cov_I2s = (sum_I2xys-sum_I2s(:)*transpose(sum_I2s(:))/(No_signals*No_realizations))/((No_signals*No_realizations)-1);
        cov_I12s = (sum_I12xys-sum_I1s(:)*transpose(sum_I2s(:))/(No_signals*No_realizations))/((No_signals*No_realizations)-1);
        
        fid=fopen([data_dir,data_I_dir,sprintf('cov_I%ds_%04g_H1_%s_%s_phasegan.dat',det1,sigma0,image_type,recon_alg)],'wb'); 
		fwrite(fid,eval(sprintf('cov_I%ds',det1)),'double'); 
		fclose(fid); clear cov_I1s
			
		fid=fopen([data_dir,data_I_dir,sprintf('cov_I%ds_%04g_H1_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)],'wb'); 
		fwrite(fid,eval(sprintf('cov_I%ds',det2)),'double'); 
		fclose(fid); clear cov_I2s
            
        fid=fopen([data_dir,data_I_dir,sprintf('cov_I1%ds_%04g_H1_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)],'wb'); 
		fwrite(fid,eval(sprintf('cov_I1%ds',det2)),'double'); 
		fclose(fid); clear cov_I12s
end



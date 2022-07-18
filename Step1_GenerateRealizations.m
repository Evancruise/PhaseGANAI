%% Generate noiseless H0 and H1 intensity realizations
% warning off; clear all; close all; clc; pack;

%function Step1_GenerateRealizations(data_dir,det,image_type)

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
        I=cell(length(d),[]);
        disp(sprintf('%d-th iteration (%s)', i, image_type));
        % if mod(j,100) ==0; disp(sprintf('%d-th iteration for the background',i)); end
        
        % Create H0 intensity realizations
        I_background=CreateLumpyBackground(x_size,y_size,x_mm,y_mm,k_bar,N_bar,L_x,L_y,al,be,sigma_psi);
        [thickness_fiber, thickness_fat] = FindBackgroundComposition(E,I_background,breast_thickness,low_percentage,high_percentage,mid_percentage);

        [phi_proj, A_proj, I_contact, I]=BKS_U(data_dir,thickness_fiber,image_type); %[U, I0]=BKS_U(data_dir,image_type); %
                % BKS_U: generate projection data, wavefield, intensity
                %        {CreateLumpyBackground
                %        FindBackgroundComposition
                %        Fresnel, Intensity}
        % phase retrieval
        %[phi, A]=phase_retrieval(recon_alg, phi_proj, A_proj, I_contact, I); %[phase, A, I]=phase_retrieval(I0, recon_alg);
        [phi, A]=phase_retrieval(recon_alg, phi_proj, A_proj, I_contact, I);
        
        eval(sprintf('phi1%db(:,:,i) = phi(crop,crop);',det2))
        eval(sprintf('b1%db(:,:,i) = A(crop,crop);',det2))
        for i_det = 1:length(d)
            eval(sprintf('I%d = I{1,i_det};',i_det));
            eval(sprintf('I%db(:,:,i)=I%d(crop,crop);',i_det,i_det));
        end
        
        phi_proj = phi_proj(crop,crop);
        A_proj = A_proj(crop,crop);
        phi = phi(crop,crop);
        A = A(crop,crop);
        
        save([data_dir,data_files_dir,sprintf('PCI_%d_%04g_H0_%s_%s.mat',i,sigma0,image_type,recon_alg)], 'phi_proj', 'phi', 'A_proj', 'A', 'I');
    end
    
    fid=fopen([data_dir,data_phi_dir,sprintf('phi1%d_b_%04g_H0_%s_%s.dat',det2,sigma0,image_type,recon_alg)],'wb'); fwrite(fid,eval(sprintf('phi1%db',det2)),'double'); fclose(fid);
    fid=fopen([data_dir,data_b_dir,sprintf('b1%d_b_%04g_H0_%s_%s.dat',det2,sigma0,image_type,recon_alg)],'wb'); fwrite(fid,eval(sprintf('b1%db',det2)),'double'); fclose(fid);
    for i_det = 1:length(d)
        fid=fopen([data_dir,data_I_dir,sprintf('I%d_b_%04g_H0_%s_%s.dat',i_det,sigma0,image_type,recon_alg)],'wb'); fwrite(fid,eval(sprintf('I%db',i_det)),'double'); fclose(fid);
    end
    
    case 'SKE'
        
    % H1: generate SKE
    eval(sprintf('phi1%ds = zeros(RAYS,RAYS,No_realizations);',det2))
    eval(sprintf('b1%ds = zeros(RAYS,RAYS,No_realizations);',det2))
    I1s=zeros(RAYS,RAYS,No_realizations);
    I2s=zeros(RAYS,RAYS,No_realizations);
           
    for i=1: No_realizations
        disp(sprintf('%d-th iteration (%s)', i, image_type));
        
        % Create H0 intensity realizations
        I_background=CreateLumpyBackground(x_size,y_size,x_mm,y_mm,k_bar,N_bar,L_x,L_y,al,be,sigma_psi);
        [thickness_fiber, thickness_fat] = FindBackgroundComposition(E,I_background,breast_thickness,low_percentage,high_percentage,mid_percentage);

        % Create H1 intensity realizations
        % BKS_U: generate projection data, wavefield, intensity
        [phi_proj, A_proj, I_contact, I]=BKS_U(data_dir,thickness_fiber,image_type); %[U, I]=BKS_U(data_dir,image_type); %
                    % BKS_U: generate projection data, wavefield, intensity
                    %        {CreateLumpyBackground
                    %        FindBackgroundComposition
                    %        FindSignalComposition
                    %        Fresnel, Intensity}
        % phase retrieval
        [phi, A]=phase_retrieval(recon_alg, phi_proj, A_proj, I_contact, I); %[phi12s, b12s]=phase_retrieval(I, recon_alg);
        
        eval(sprintf('phi1%ds(:,:,i) = phi(crop,crop);',det2))
        eval(sprintf('b1%ds(:,:,i) = A(crop,crop);',det2))
        
        for i_det = 1:length(d)
            eval(sprintf('I%d = I{1,i_det};',i_det));
            eval(sprintf('I%ds(:,:,i)=I%d(crop,crop);',i_det,i_det));
        end
        
        phi_proj = phi_proj(crop,crop);
        A_proj = A_proj(crop,crop);
        phi = phi(crop,crop);
        A = A(crop,crop);
        
        save([data_dir,data_files_dir,sprintf('PCI_%d_%04g_H1_%s_%s.mat',i,sigma0,image_type,recon_alg)], 'phi_proj', 'phi', 'A_proj', 'A', 'I');
    end
    
    fid=fopen([data_dir,data_phi_dir,sprintf('phi1%d_s_%04g_H1_%s_%s.dat',det2,sigma0,image_type,recon_alg)],'wb'); fwrite(fid,eval(sprintf('phi1%ds',det2)),'double'); fclose(fid);
    fid=fopen([data_dir,data_b_dir,sprintf('b1%d_s_%04g_H1_%s_%s.dat',det2,sigma0,image_type,recon_alg)],'wb'); fwrite(fid,eval(sprintf('b1%ds',det2)),'double'); fclose(fid);
    for i_det = 1:length(d)
        fid=fopen([data_dir,data_I_dir,sprintf('I%d_s_%04g_H1_%s_%s.dat',i_det,sigma0,image_type,recon_alg)],'wb'); fwrite(fid,eval(sprintf('I%ds',i_det)),'double'); fclose(fid);
    end
    
    case 'SKEs'
    % H1: generate SKE
    eval(sprintf('phi1%ds = zeros(RAYS,RAYS,No_signals*No_realizations);',det2))
    eval(sprintf('b1%ds = zeros(RAYS,RAYS,No_signals*No_realizations);',det2))
    I1s=zeros(RAYS,RAYS,No_signals*No_realizations);
    I2s=zeros(RAYS,RAYS,No_signals*No_realizations);
        
    for si=1: No_signals
        for i=No_realizations*(si-1)+1: No_realizations*si
            disp(sprintf('%d-th iteration (%s-H%d)',i, image_type, si));
            % Create H0 intensity realizations
            I_background=CreateLumpyBackground(x_size,y_size,x_mm,y_mm,k_bar,N_bar,L_x,L_y,al,be,sigma_psi);
            [thickness_fiber, thickness_fat] = FindBackgroundComposition(E,I_background,breast_thickness,low_percentage,high_percentage,mid_percentage);
        
            % Create H1 intensity realizations
            % BKS_U: generate projection data, wavefield, intensity
            [phi_proj, A_proj, I_contact, I]=BKS_U(data_dir,thickness_fiber,image_type,si);
            % phase retrieval
            [phi, A]=phase_retrieval(recon_alg, phi_proj, A_proj, I_contact, I);
              
            eval(sprintf('phi1%ds(:,:,i) = phi(crop,crop);',det2))
            eval(sprintf('b1%ds(:,:,i) = A(crop,crop);',det2))
            
            for i_det = 1:length(d)
                eval(sprintf('I%d = I{1,i_det};',i_det));
                eval(sprintf('I%ds(:,:,i)=I%d(crop,crop);',i_det,i_det));
            end
        
            phi_proj = phi_proj(crop,crop); % ground truth
            A_proj = A_proj(crop,crop);     % ground truth
            phi = phi(crop,crop);           % reconstructed phase
            A = A(crop,crop);               % reconstructed A
            
            save([data_dir,data_files_dir,sprintf('PCI_%d_%04g_H1_%s_%s.mat',i,sigma0,image_type,recon_alg)], 'phi_proj', 'phi', 'A_proj', 'A', 'I');
        end
    end
    fid=fopen([data_dir,data_phi_dir,sprintf('phi1%d_s_%04g_H1_%s_%s.dat',det2,sigma0,image_type,recon_alg)],'wb'); fwrite(fid,eval(sprintf('phi1%ds',det2)),'double'); fclose(fid);
    fid=fopen([data_dir,data_b_dir,sprintf('b1%d_s_%04g_H1_%s_%s.dat',det2,sigma0,image_type,recon_alg)],'wb'); fwrite(fid,eval(sprintf('b1%ds',det2)),'double'); fclose(fid);
    for i_det = 1:length(d)
        fid=fopen([data_dir,data_I_dir,sprintf('I%d_s_%04g_H1_%s_%s.dat',i_det,sigma0,image_type,recon_alg)],'wb'); fwrite(fid,eval(sprintf('I%ds',i_det)),'double'); fclose(fid);
    end
end




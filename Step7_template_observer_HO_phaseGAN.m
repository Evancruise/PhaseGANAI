%% template
% IO
% (1)Ideal (AUC upper bound):t(g) = pr(g|H1)/pr(g|H0) 
% (2)Empirical:t(g) = exp[-1/2*(g-s2)^T*Sigma_g^(-1)*(g-s2)] / exp[-1/2*(g-s1)^T*Sigma_g^(-1)*(g-s1)]
%                   = delta(s)^T*Sigma^(-1)*delta(s)
% (3)SNR^2 = delta(s)^T*Sigma_g^(-1)*delta(s)
% Note that delta(s) = s2-s1.
% HO
% t(g) = transpose(w)*g;
% where w = [1/2*(K0+K1)]^(-1)*delta(g)
% where delta(g) = g2-g1 = s2-s1
%% Empirical SNR2 
% SNR2_em = 2 * mean(Tp_s-Tp_b).^2./(var(Tp_s)+var(Tp_b));
% SNR2_I1_em = 2 * mean(TI_s1-TI_b1).^2./(var(TI_s1)+var(TI_b1));

% Ideal SNR2 for IO
% dg1 = I1_0sm-I1_0bm;
% SNR2_I1(1) = dg1(:)'/sigma^2*dg1(:);
% SNR2_I_IO%d=transpose(wI%d)*dg2(:);

load([data_dir,'Data0_UserDefinedParameters_phasegan.mat']);
invert_cov=true;
pt = 20; 
ran=1:RAYS; 
ray=size(ran,2); 

switch image_type
    case 'SKE'
               %% Load and invert covariance matrix for Phi
        disp('Load and invert covariance matrix for Phi');
        
        filenames = [data_dir,data_phi_dir,sprintf('cov_phi1%ds_%04g_H1_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)];
        filenameb = [data_dir,data_phi_dir,sprintf('cov_phi1%db_%04g_H0_BKS_%s_phasegan.dat',det2,sigma0,recon_alg)];
        filename1 = [data_dir,data_phi_dir,sprintf('KP1%dinv_%04g_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)];
        
        if invert_cov %~isfile(filename1)
            fid=fopen(filenameb,'rb'); 
            KP12b=fread(fid,[RAYS^2,RAYS^2],'double');
            fclose(fid);  % Load Cov due to background
            fid=fopen(filenames,'rb'); 
            KP12s=fread(fid,[RAYS^2,RAYS^2],'double');
            fclose(fid);  % Load Cov due to signal
            
            KP12 = 0.5*(KP12b+KP12s);
            clear KP12b
            
            if ray<RAYS  % crop the covariance matrix
                KP12=reshape(KP12,[RAYS,RAYS,RAYS,RAYS]);
                KP12_ran=KP12(ran,ran,ran,ran); 
                clear KP12;
                KP12_ran=reshape(KP12_ran,[ray^2,ray^2]); 
                KP12=KP12_ran;  clear KP12_ran
            end
            
            KPinv12=blockwise_inv3(KP12); % Invert covariance matrix
            fid=fopen(filename1,'wb'); 
            fwrite(fid,KPinv12,'double');
            fclose(fid); % Save the inverted covariance matrix
        else
            fid=fopen(filename1,'rb'); 
            KPinv12=fread(fid,[RAYS^2,RAYS^2],'double');
            fclose(fid);  
        end
        
        fid=fopen([data_dir,data_phi_dir,sprintf('phi1%d_b_%04g_H0_BKS_%s_phasegan.dat',det2,sigma0,recon_alg)],'rb'); 
        phi12b=fread(fid,[RAYS*RAYS,No_realizations],'double'); fclose(fid); 
        phi12b=reshape(phi12b,[RAYS,RAYS,No_realizations]);
        
        fid=fopen([data_dir,data_phi_dir,sprintf('phi1%d_s_%04g_H1_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)],'rb');
        phi12s = fread(fid,[RAYS*RAYS,No_realizations],'double'); fclose(fid);
        phi12s = reshape(phi12s,[RAYS,RAYS,No_realizations]);
        
        phi12s = phi12s(ran,ran,:); 
        phi12b = phi12b(ran,ran,:); 
        dg_phi=phi12s-phi12b;  
        dg_phi=mean(dg_phi,3); 
        
        eval(sprintf('wp1%d=KPinv12*dg_phi(:);',det2)); % Hotelling template of phase
        eval(sprintf('SNR2_p_HO1%d=transpose(wp1%d)*dg_phi(:);',det2,det2));% SNR2 of phase for HO
        
               %% Load and invert covariance matrix for A
        disp('Load and invert covariance matrix for A');
        
        filenameb = [data_dir,data_b_dir,sprintf('cov_b1%db_%04g_H0_BKS_%s_phasegan.dat',det2,sigma0,recon_alg)];
        filenames = [data_dir,data_b_dir,sprintf('cov_b1%ds_%04g_H1_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)];
        filename1 = [data_dir,data_b_dir,sprintf('KA1%dinv_%04g_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)];
        
        if invert_cov %~isfile(filename1)
            fid=fopen(filenameb,'rb'); 
            KA12b=fread(fid,[RAYS^2,RAYS^2],'double');
            fclose(fid);  % Load Cov due to background
            fid=fopen(filenames,'rb'); 
            KA12s=fread(fid,[RAYS^2,RAYS^2],'double');
            fclose(fid);  % Load Cov due to signal
            
            KA12 = 0.5*(KA12s+KA12b);
            clear KA12b KA12s
            
            if ray<RAYS  % crop the covariance matrix
                KA12=reshape(KA12,[RAYS,RAYS,RAYS,RAYS]);
                KA12_ran=KA12(ran,ran,ran,ran); 
                clear KA12;
                KA12_ran=reshape(KA12_ran,[ray^2,ray^2]); 
                KA12=KA12_ran;  clear KA12_ran
            end
            
            KAinv12=blockwise_inv3(KA12); % Invert covariance matrix
            fid=fopen(filename1,'wb'); 
            fwrite(fid,KAinv12,'double');
            fclose(fid); % Save the inverted covariance matrix
        else
            fid=fopen(filename1,'rb'); 
            KAinv12=fread(fid,[RAYS^2,RAYS^2],'double');
            fclose(fid);  
        end 
        
        fid=fopen([data_dir,data_b_dir,sprintf('b1%d_b_%04g_H0_BKS_%s_phasegan.dat',det2,sigma0,recon_alg)],'rb'); 
        b12b=fread(fid,[RAYS*RAYS,No_realizations],'double'); fclose(fid); 
        b12b=reshape(b12b,[RAYS,RAYS,No_realizations]);
        
        fid=fopen([data_dir,data_b_dir,sprintf('b1%d_s_%04g_H1_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)],'rb'); 
        b12s = fread(fid,[RAYS*RAYS,No_realizations],'double'); fclose(fid);
        b12s = reshape(b12s,[RAYS,RAYS,No_realizations]);
        
        b12s = b12s(ran,ran,:); 
        b12b = b12b(ran,ran,:); 
        dg_b=phi12s-b12b;  
        dg_b=mean(dg_b,3); 
        
        eval(sprintf('wb1%d=KAinv12*dg_b(:);',det2)); % Hotelling template of phase
        eval(sprintf('SNR2_b_HO1%d=transpose(wb1%d)*dg_b(:);',det2,det2));% SNR2 of phase for HO        
        
              %% Load and invert covariance matrix for phi + A
        disp('Load and invert covariance matrix for Phi+A');
        
        filenameb = [data_dir,data_phib_dir,sprintf('cov_phib1%db_%04g_H0_BKS_%s_phasegan.dat',det2,sigma0,recon_alg)];
        filenames = [data_dir,data_phib_dir,sprintf('cov_phib1%ds_%04g_H1_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)];
        filename1 = [data_dir,data_phib_dir,sprintf('KPA1%dinv_%04g_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)];
        
        if invert_cov %~isfile(filename1)
            fid=fopen(filenameb,'rb'); 
            KPA12b=fread(fid,[RAYS^2,RAYS^2],'double');
            fclose(fid);  % Load Cov due to background
            fid=fopen(filenames,'rb'); 
            KPA12s=fread(fid,[RAYS^2,RAYS^2],'double');
            fclose(fid);  % Load Cov due to signal
            
            KPA12 = 0.5*(KPA12s+KPA12b);
            clear KPA12b KPA12s
            KPAinv12=blockwise_inv4(KP12, KPA12, KPA12', KA12);
            clear KP12 KPA12 KA12
            
            fid=fopen(filename1,'wb'); 
            fwrite(fid,KPAinv12,'double');
            fclose(fid); % Save the inverted covariance matrix
        else
            fid=fopen(filename1,'rb'); 
            KPAinv12=fread(fid,[2*RAYS^2,2*RAYS^2],'double');
            fclose(fid);
        end
        
        eval(sprintf('wpb1%d=KPAinv12*[dg_phi(:);dg_b(:)];',det2)); % Hotelling template of phase+absorption
        eval(sprintf('SNR2_pb_HO1%d=transpose(wpb1%d)*[dg_phi(:); dg_b(:)];',det2,det2)) %SNR2_of phase+absorption for HO 
                                    
              %% Load and invert covariance matrix for I1
        disp('Load and invert covariance matrix for I1');
        
        filenameb=[data_dir,data_I_dir,sprintf('cov_I%db_%04g_H0_BKS_%s_phasegan.dat',det1,sigma0,recon_alg)];
        filenames=[data_dir,data_I_dir,sprintf('cov_I%ds_%04g_H1_%s_%s_phasegan.dat',det1,sigma0,image_type,recon_alg)];
        filename1=[data_dir,data_I_dir,sprintf('KI%dinv_%04g_%s_%s_phasegan.dat',det1,sigma0,image_type,recon_alg)];

        if invert_cov %~isfile(filename1)
            fid=fopen(filenameb,'rb'); 
            KI1b=fread(fid,[RAYS^2,RAYS^2],'double');
            fclose(fid);
            fid=fopen(filenames,'rb'); 
            KI1s=fread(fid,[RAYS^2,RAYS^2],'double');
            fclose(fid);
            
            KI1 = 0.5*(KI1b+KI1s);
            clear KI1b KI1s
            
            if ray<RAYS  % crop the covariance matrix
                KI1=reshape(KI1,[RAYS,RAYS,RAYS,RAYS]);
                KI1_ran=KI1(ran,ran,ran,ran); 
                clear KI1;
                KI1_ran=reshape(KI1_ran,[ray^2,ray^2]); 
                KI1=KI1_ran;
            end
            
            KIinv1=blockwise_inv3(KI1); % Invert covariance matrix
            fid=fopen(filename1,'wb'); 
            fwrite(fid,KIinv1,'double');
            fclose(fid); % Save the inverted covariance matrix
        else
            fid=fopen(filename1,'rb'); 
            KIinv1=fread(fid,[RAYS^2,RAYS^2],'double');
            fclose(fid);  
        end 
        
        fid=fopen([data_dir,data_I_dir,sprintf('I%d_b_%04g_H0_BKS_%s_phasegan.dat',det1,sigma0,recon_alg)],'rb'); 
        I1b=fread(fid,[RAYS*RAYS,No_realizations],'double'); fclose(fid); 
        I1b=reshape(I1b,[RAYS,RAYS,No_realizations]);
        
        fid=fopen([data_dir,data_I_dir,sprintf('I%d_s_%04g_H1_%s_%s_phasegan.dat',det1,sigma0,image_type,recon_alg)],'rb');
        I1s = fread(fid,[RAYS*RAYS,No_realizations],'double'); fclose(fid);
        I1s = reshape(I1s,[RAYS,RAYS,No_realizations]);
        
        I1s=I1s(ran,ran,:); 
        I1b=I1b(ran,ran,:); 
        dg1=I1s-I1b;  
        dg1=mean(dg1,3); 
        
        eval(sprintf('wI%d=KIinv1*[dg1(:)];',det1)); % Hotelling template of intensity on detector det1
        eval(sprintf('SNR2_I_HO%d=transpose(wI%d)*dg1(:);',det1,det1)); %SNR2 of intensity on detector det1 for HO
        
               %% Load and invert covariance matrix for I2
        disp('Load and invert covariance matrix for I2');
        
        filenameb=[data_dir,data_I_dir,sprintf('cov_I%db_%04g_H0_BKS_%s_phasegan.dat',det2,sigma0,recon_alg)];
        filenames=[data_dir,data_I_dir,sprintf('cov_I%ds_%04g_H1_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)];
        filename1=[data_dir,data_I_dir,sprintf('KI%dinv_%04g_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)];

        if invert_cov %~isfile(filename1)
            fid=fopen(filenameb,'rb'); 
            KI2b=fread(fid,[RAYS^2,RAYS^2],'double');
            fclose(fid);
            fid=fopen(filenames,'rb'); 
            KI2s=fread(fid,[RAYS^2,RAYS^2],'double');
            fclose(fid);
            
            KI2 = 0.5*(KI2b+KI2s);
            clear KI2b K12s
            
            if ray<RAYS  % crop the covariance matrix
                KI2=reshape(KI2,[RAYS,RAYS,RAYS,RAYS]);
                KI2_ran=KI2(ran,ran,ran,ran); 
                clear KP12;
                KI2_ran=reshape(KI2_ran,[ray^2,ray^2]); 
                KI2=KI2_ran;
            end
            
            KIinv2=blockwise_inv3(KI2); % Invert covariance matrix
            fid=fopen(filename1,'wb'); 
            fwrite(fid,KIinv2,'double');
            fclose(fid); % Save the inverted covariance matrix
        else
            fid=fopen(filename1,'rb'); 
            KIinv2=fread(fid,[RAYS^2,RAYS^2],'double');
            fclose(fid);  
        end 
        
        fid=fopen([data_dir,data_I_dir,sprintf('I%d_b_%04g_H0_BKS_%s_phasegan.dat',det2,sigma0,recon_alg)],'rb'); 
        I2b=fread(fid,[RAYS*RAYS,No_realizations],'double'); fclose(fid); 
        I2b=reshape(I2b,[RAYS,RAYS,No_realizations]);
        
        fid=fopen([data_dir,data_I_dir,sprintf('I%d_s_%04g_H1_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)],'rb');
        I2s = fread(fid,[RAYS*RAYS,No_realizations],'double'); fclose(fid);
        I2s = reshape(I2s,[RAYS,RAYS,No_realizations]);
        
        I2s=I2s(ran,ran,:); 
        I2b=I2b(ran,ran,:); 
        dg2=I2s-I2b;  
        dg2=mean(dg2,3); 
        
        eval(sprintf('wI%d=KIinv2*[dg2(:)];',det2)); % Hotelling template of intensity on detector det1
        eval(sprintf('SNR2_I_HO%d=transpose(wI%d)*dg2(:);',det2,det2)); %SNR2 of intensity on detector det1 for HO
              
              %% Load and invert covariance matrix for I1+I2
        disp('Load and invert covariance matrix for I1+I2');
        
        filenameb=[data_dir,data_I_dir,sprintf('cov_I1%db_%04g_H0_BKS_%s_phasegan.dat',det2,sigma0,recon_alg)];
        filenames=[data_dir,data_I_dir,sprintf('cov_I1%ds_%04g_H1_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)];
        filename1=[data_dir,data_I_dir,sprintf('KI1%dinv_%04g_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)];
        
        if invert_cov %~isfile(filename1)
            fid=fopen(filenameb,'rb'); 
            KI12b=fread(fid,[RAYS^2,RAYS^2],'double');
            fclose(fid);  % Load Cov due to background
            fid=fopen(filenames,'rb'); 
            KI12s=fread(fid,[RAYS^2,RAYS^2],'double');
            fclose(fid);  % Load Cov due to signal
            
            KI12 = 0.5*(KI12b+KI12s);
            clear KPA12b
            KIinv12=blockwise_inv4(KI1, KI12, KI12', KI2);
            clear KI1 KI2 KI12
            
            fid=fopen(filename1,'wb'); 
            fwrite(fid,KIinv12,'double');
            fclose(fid); % Save the inverted covariance matrix
        else
            fid=fopen(filename1,'rb'); 
            KIinv12=fread(fid,[2*RAYS^2,2*RAYS^2],'double');
            fclose(fid);
        end

        eval(sprintf('wI1%d=KIinv12*[dg1(:);dg2(:)];',det2)); % Hotelling template of phase+absorption
        eval(sprintf('SNR2_I_HO1%d=transpose(wI1%d)*[dg1(:); dg2(:)];',det2,det2)) %SNR2_of phase+absorption for HO           
        
              %% Hotelling observer study
        disp('Hotelling observer study');
        Tp_s=zeros(No_realizations,1); 
        Tp_b=Tp_s;
        Tb_s=zeros(No_realizations,1); 
        Tb_b=Tb_s;
        Tpb_s=zeros(No_realizations,1); 
        Tpb_b=Tpb_s;

        TI_s1=zeros(No_realizations,1); 
        TI_b1=TI_s1;
        TI_s2=zeros(No_realizations,1); 
        TI_b2=TI_s2;
        TI_s12=zeros(No_realizations,1); 
        TI_b12=TI_s12;

        for i=1:No_realizations   
            gp=phi12s(:,:,i);
            eval(sprintf('Tp_s(i)=transpose(wp1%d)*gp(:);',det2))
    
            gb=b12s(:,:,i); 
            eval(sprintf('Tb_s(i)=transpose(wb1%d)*gb(:);',det2))
            eval(sprintf('Tpb_s(i)=transpose(wpb1%d)*[gp(:);gb(:)];',det2))
    
            gp=phi12b(:,:,i);
            eval(sprintf('Tp_b(i)=transpose(wp1%d)*gp(:);',det2))
    
            gb=b12b(:,:,i);
            eval(sprintf('Tb_b(i)=transpose(wb1%d)*gb(:);',det2))
            eval(sprintf('Tpb_b(i)=transpose(wpb1%d)*[gp(:);gb(:)];',det2))
    
            g1=I1s(:,:,i);
            eval(sprintf('TI_s1(i)=transpose(wI%d)*g1(:);',det1))
    
            g2=I2s(:,:,i); 
            eval(sprintf('TI_s2(i)=transpose(wI%d)*g2(:);',det2))
            eval(sprintf('TI_s12(i)=transpose(wI1%d)*[g1(:);g2(:)];',det2))
    
            g1=I1b(:,:,i);
            eval(sprintf('TI_b1(i)=transpose(wI%d)*g1(:);',det1))
    
            g2=I2b(:,:,i);
            eval(sprintf('TI_b2(i)=transpose(wI%d)*g2(:);',det2))
            eval(sprintf('TI_b12(i)=transpose(wI1%d)*[g1(:);g2(:)];',det2))
        end
        
        T_s=[Tp_s, Tb_s, Tpb_s, TI_s1, TI_s2, TI_s12];
        T_b=[Tp_b, Tb_b, Tpb_b, TI_b1, TI_b2, TI_b12];
        
    case 'SKEs'
        
              %% Load and invert covariance matrix for Phi
        disp('Load and invert covariance matrix for Phi');
            
        filenameb=[data_dir,data_phi_dir,sprintf('cov_phi1%db_%04g_H0_BKS_%s_phasegan.dat',det2,sigma0,recon_alg)]; 
        filenames=[data_dir,data_phi_dir,sprintf('cov_phi1%ds_%04g_H1_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)]; 
        filename1=[data_dir,data_phi_dir,sprintf('KP1%dinv_%04g_%s_%s_H1.dat',det2,sigma0,image_type,recon_alg)];

        if invert_cov %~isfile(filename1)
            fid=fopen(filenameb,'rb'); 
            KP12b=fread(fid,[RAYS^2,RAYS^2],'double');
            fclose(fid);  % Load Cov due to background
            fid=fopen(filenames,'rb'); 
            KP12s=fread(fid,[RAYS^2,RAYS^2],'double');
            fclose(fid);  % Load Cov due to signal
    
            KP12 = 0.5*(KP12b+KP12s);
            clear KP12b
        
            if ray<RAYS  % crop the covariance matrix
               KP12=reshape(KP12,[RAYS,RAYS,RAYS,RAYS]);
               KP12_ran=KP12(ran,ran,ran,ran); 
               clear KP12;
               KP12_ran=reshape(KP12_ran,[ray^2,ray^2]); 
               KP12=KP12_ran;  clear KP12_ran
            end
    
            KPinv12=blockwise_inv3(KP12); % Invert covariance matrix
            fid=fopen(filename1,'wb'); 
            fwrite(fid,KPinv12,'double');
            fclose(fid); % Save the inverted covariance matrix
        else
            fid=fopen(filename1,'rb'); 
            KPinv12=fread(fid,[RAYS^2,RAYS^2],'double');
            fclose(fid);  
        end
	
		fid=fopen([data_dir,data_phi_dir,sprintf('phi1%d_b_%04g_H0_BKS_%s_phasegan.dat',det2,sigma0,recon_alg)],'rb'); 
		phi12b=fread(fid,[RAYS*RAYS,No_signals*No_realizations],'double'); fclose(fid); 
		phi12b=reshape(phi12b,[RAYS,RAYS,No_signals*No_realizations]);
        
        fid=fopen([data_dir,data_phi_dir,sprintf('phi1%d_s_%04g_H1_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)],'rb');
		phi12s = fread(fid,[RAYS*RAYS,No_signals*No_realizations],'double'); fclose(fid);
		phi12s = reshape(phi12s,[RAYS,RAYS,No_signals*No_realizations]);
        
		phi12s = phi12s(ran,ran,:); 
		phi12b = phi12b(ran,ran,:); 
		dg_phi=phi12s-phi12b;  
		dg_phi=mean(dg_phi,3); 
	
		eval(sprintf('wp1%d=KPinv12*dg_phi(:);',det2)); % Hotelling template of phase
		%eval(sprintf('SNR2_p_HO1%d_H%d=transpose(wp1%d)*dg_phi(:);',det2,si,det2,si));% SNR2 of phase
	
			  %% Load and invert covariance matrix for A
		disp('Load and invert covariance matrix for A');
		
		filenameb=[data_dir,data_b_dir,sprintf('cov_b1%db_%04g_H0_BKS_%s_phasegan.dat',det2,sigma0,recon_alg)]; 
		filenames=[data_dir,data_b_dir,sprintf('cov_b1%ds_%04g_H1_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)];
		filename1=[data_dir,data_b_dir,sprintf('KA1%dinv_%04g_%s_%s_H1.dat',det2,sigma0,image_type,recon_alg)];
	
		if invert_cov %~isfile(filename1)
			fid=fopen(filenameb,'rb'); 
			KA12b=fread(fid,[RAYS^2,RAYS^2],'double');
			fclose(fid);  % Load Cov due to background
			fid=fopen(filenames,'rb'); 
			KA12s=fread(fid,[RAYS^2,RAYS^2],'double');
			fclose(fid);  % Load Cov due to signal
	
			KA12 = 0.5*(KA12b+KA12s);
			clear KA12b KA12s
		
			if ray<RAYS  % crop the covariance matrix
				KA12=reshape(KA12,[RAYS,RAYS,RAYS,RAYS]);
				KA12_ran=KA12(ran,ran,ran,ran); 
				clear KA12;
				KA12_ran=reshape(KA12_ran,[ray^2,ray^2]); 
				KA12=KA12_ran;  clear KA12_ran
			end
	
			KAinv12=blockwise_inv3(KA12); % Invert covariance matrix
			fid=fopen(filename1,'wb'); 
			fwrite(fid,KAinv12,'double');
			fclose(fid); % Save the inverted covariance matrix
		else
			fid=fopen(filename1,'rb'); 
			KAinv12=fread(fid,[RAYS^2,RAYS^2],'double');
			fclose(fid);  
		end
	
		fid=fopen([data_dir,data_b_dir,sprintf('b1%d_s_%04g_H1_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)],'rb');
		b12s=fread(fid,[RAYS*RAYS,No_signals*No_realizations],'double'); fclose(fid);
		b12s=reshape(b12s,[RAYS,RAYS,No_signals*No_realizations]);
	
		fid=fopen([data_dir,data_b_dir,sprintf('b1%d_b_%04g_H0_BKS_%s_phasegan.dat',det2,sigma0,recon_alg)],'rb'); 
		b12b=fread(fid,[RAYS*RAYS,No_signals*No_realizations],'double'); fclose(fid); 
		b12b=reshape(b12b,[RAYS,RAYS,No_signals*No_realizations]);
	
		b12s=b12s(ran,ran,:); 
		b12b=b12b(ran,ran,:); 
		dg_b=b12s-b12b;  
		dg_b=mean(dg_b,3); 
	
		eval(sprintf('wb1%d=KAinv12*dg_b(:);',det2)); % Hotelling template of phase
		%eval(sprintf('SNR2_b_HO1%d_H%d=transpose(wb1%d_H%d)*dg_b(:);',det2,si,det2,si));% SNR2 of phase for HO
	
				%% Load and invert covariance matrix for phi + A
		disp('Load and invert covariance matrix for Phi+A');
		
		filenameb=[data_dir,data_phib_dir,sprintf('cov_phib1%db_%04g_H0_BKS_%s_phasegan.dat',det2,sigma0,recon_alg)]; 
		filenames=[data_dir,data_phib_dir,sprintf('cov_phib1%ds_%04g_H1_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)]; 
		filename1=[data_dir,data_phib_dir,sprintf('KPA1%dinv_%04g_%s_%s_H1.dat',det2,sigma0,image_type,recon_alg)];
	
		if invert_cov %~isfile(filename1)
			fid=fopen(filenameb,'rb'); 
			KPA12b=fread(fid,[RAYS^2,RAYS^2],'double');
			fclose(fid);  % Load Cov due to background
			fid=fopen(filenames,'rb'); 
			KPA12s=fread(fid,[RAYS^2,RAYS^2],'double');
			fclose(fid);  % Load Cov due to signal
		
			KPA12 = 0.5*(KPA12b+KPA12s);
			clear KPA12b KPA12s
			KPAinv12=blockwise_inv4(KP12, KPA12, KPA12', KA12);
			clear KP12 KPA12 KA12
		
			fid=fopen(filename1,'wb'); 
			fwrite(fid,KPAinv12,'double');
			fclose(fid); % Save the inverted covariance matrix
		else
			fid=fopen(filename1,'rb'); 
			KPAinv12=fread(fid,[2*RAYS^2,2*RAYS^2],'double');
			fclose(fid);
		end
	
		eval(sprintf('wpb1%d=KPAinv12*[dg_phi(:);dg_b(:)];',det2)); % Hotelling template of phase+absorption
		%eval(sprintf('SNR2_pb_HO1%d_H%d=transpose(wpb1%d_H%d)*[dg_phi(:); dg_b(:)];',det2,si,det2,si)) %SNR2_of phase+absorption
	
				%% Load and invert covariance matrix for I1
		disp('Load and invert covariance matrix for I1');
				
		filenameb=[data_dir,data_I_dir,sprintf('cov_I%db_%04g_H0_BKS_%s_phasegan.dat',det1,sigma0,recon_alg)]; 
		filenames=[data_dir,data_I_dir,sprintf('cov_I%ds_%04g_H1_%s_%s_phasegan.dat',det1,sigma0,image_type,recon_alg)]; 
		filename1=[data_dir,data_I_dir,sprintf('KI%dinv_%04g_%s_%s_H1.dat',det1,sigma0,image_type,recon_alg)];
	
		if invert_cov %~isfile(filename1)
			fid=fopen(filenameb,'rb'); 
			KI1b=fread(fid,[RAYS^2,RAYS^2],'double');
			fclose(fid);  % Load Cov due to background
			fid=fopen(filenames,'rb'); 
			KI1s=fread(fid,[RAYS^2,RAYS^2],'double');
			fclose(fid);  % Load Cov due to signal
		
			KI1 = 0.5*(KI1b+KI1s);
			clear KI1b KI1s
		
			if ray<RAYS  % crop the covariance matrix
				KI1=reshape(KI1,[RAYS,RAYS,RAYS,RAYS]);
				KI1_ran=KI1(ran,ran,ran,ran); 
				clear KI1;
				KI1_ran=reshape(KI1_ran,[ray^2,ray^2]); 
				KI1=KI1_ran;  clear KI1_ran
			end
	
			KIinv1=blockwise_inv3(KI1); % Invert covariance matrix
			fid=fopen(filename1,'wb'); 
			fwrite(fid,KIinv1,'double');
			fclose(fid); % Save the inverted covariance matrix
		
		else
			fid=fopen(filename1,'rb'); 
			KIinv1=fread(fid,[2*RAYS^2,2*RAYS^2],'double');
			fclose(fid);
		end
	
		fid=fopen([data_dir,data_I_dir,sprintf('I%d_b_%04g_H0_BKS_%s_phasegan.dat',det1,sigma0,recon_alg)],'rb'); 
		I1b=fread(fid,[RAYS*RAYS,No_signals*No_realizations],'double'); fclose(fid); 
		I1b=reshape(I1b,[RAYS,RAYS,No_signals*No_realizations]);
        
        fid=fopen([data_dir,data_I_dir,sprintf('I%d_s_%04g_H1_%s_%s_phasegan.dat',det1,sigma0,image_type,recon_alg)],'rb');
		I1s = fread(fid,[RAYS*RAYS,No_signals*No_realizations],'double'); fclose(fid);
		I1s = reshape(I1s,[RAYS,RAYS,No_signals*No_realizations]);
	
		I1s=I1s(ran,ran,:); 
		I1b=I1b(ran,ran,:); 
		dg1=I1s-I1b;  
		dg1=mean(dg1,3); 
	
		eval(sprintf('wI%d=KIinv1*[dg1(:)];',det1)); % Hotelling template of intensity on detector det1
		%eval(sprintf('SNR2_I_HO%d_H%d=transpose(wI%d_H%d)*dg1(:);',det1,si,det1,si)); %SNR2 of intensity on detector det1
	
				%% Load and invert covariance matrix for I2
		disp('Load and invert covariance matrix for I2');
				
		filenameb=[data_dir,data_I_dir,sprintf('cov_I%db_%04g_H0_BKS_%s_phasegan.dat',det2,sigma0,recon_alg)]; 
		filenames=[data_dir,data_I_dir,sprintf('cov_I%ds_%04g_H1_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)]; 
		filename1=[data_dir,data_I_dir,sprintf('KI%dinv_%04g_%s_%s_H1.dat',det2,sigma0,image_type,recon_alg)];
	
		if invert_cov %~isfile(filename1)
			fid=fopen(filenameb,'rb'); 
			KI2b=fread(fid,[RAYS^2,RAYS^2],'double');
			fclose(fid);  % Load Cov due to background
			fid=fopen(filenames,'rb'); 
			KI2s=fread(fid,[RAYS^2,RAYS^2],'double');
			fclose(fid);  % Load Cov due to signal
		
			KI2 = 0.5*(KI2b+KI2s);
			clear KI2b K12s
		
			if ray<RAYS  % crop the covariance matrix
				KI2=reshape(KI2,[RAYS,RAYS,RAYS,RAYS]);
				KI2_ran=KI2(ran,ran,ran,ran); 
				clear KI2;
				KI2_ran=reshape(KI2_ran,[ray^2,ray^2]); 
				KI2=KI2_ran;  clear KI2_ran
			end
	
			KIinv2=blockwise_inv3(KI2); % Invert covariance matrix
			fid=fopen(filename1,'wb'); 
			fwrite(fid,KIinv2,'double');
			fclose(fid); % Save the inverted covariance matrix
		
		else
			fid=fopen(filename1,'rb'); 
			KIinv2=fread(fid,[2*RAYS^2,2*RAYS^2],'double');
			fclose(fid);
		end
	
		fid=fopen([data_dir,data_I_dir,sprintf('I%d_b_%04g_H0_BKS_%s_phasegan.dat',det2,sigma0,recon_alg)],'rb'); 
		I2b = fread(fid,[RAYS*RAYS,No_signals*No_realizations],'double'); fclose(fid); 
		I2b = reshape(I2b,[RAYS,RAYS,No_signals*No_realizations]);
		
		fid=fopen([data_dir,data_I_dir,sprintf('I%d_s_%04g_H1_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)],'rb');
		I2s = fread(fid,[RAYS*RAYS,No_signals*No_realizations],'double'); fclose(fid);
		I2s = reshape(I2s,[RAYS,RAYS,No_signals*No_realizations]);
	
		I2s=I2s(ran,ran,:); 
		I2b=I2b(ran,ran,:); 
		dg2=I2s-I2b;  
		dg2=mean(dg2,3); 
	
		eval(sprintf('wI%d=KIinv1*[dg1(:)];',det2)); % Hotelling template of intensity on detector det1
		%eval(sprintf('SNR2_I_HO%d_H%d=transpose(wI%d_H%d)*dg1(:);',det2,si,det2,si)); %SNR2 of intensity on detector det1
	
			  %% Load and invert covariance matrix for I1+I2
		disp('Load and invert covariance matrix for I1+I2');
		
		filenameb=[data_dir,data_I_dir,sprintf('cov_I1%db_%04g_H0_BKS_%s_phasegan.dat',det2,sigma0,recon_alg)];
		filenames=[data_dir,data_I_dir,sprintf('cov_I1%ds_%04g_H1_%s_%s_phasegan.dat',det2,sigma0,image_type,recon_alg)];
		filename1=[data_dir,data_I_dir,sprintf('KI1%dinv_%04g_%s_%s_H1.dat',det2,sigma0,image_type,recon_alg)];
	
		if invert_cov %~isfile(filename1)
			fid=fopen(filenameb,'rb'); 
			KI12b=fread(fid,[RAYS^2,RAYS^2],'double');
			fclose(fid);  % Load Cov due to background
			fid=fopen(filenames,'rb'); 
			KI12s=fread(fid,[RAYS^2,RAYS^2],'double');
			fclose(fid);  % Load Cov due to signal
		
			KI12 = 0.5*(KI12b+KI12s);
			clear KI12b KI12s
			KIinv12=blockwise_inv4(KI1, KI12, KI12', KI2);
			clear KI1 KI2 KI12
		
			fid=fopen(filename1,'wb'); 
			fwrite(fid,KIinv12,'double');
			fclose(fid); % Save the inverted covariance matrix
		else
			fid=fopen(filename1,'rb'); 
			KIinv12=fread(fid,[2*RAYS^2,2*RAYS^2],'double');
			fclose(fid);
		end
	
		eval(sprintf('wI1%d=KIinv12*[dg1(:);dg2(:)];',det2)); % Hotelling template of phase+absorption
		%eval(sprintf('SNR2_I_HO1%d_H%d=transpose(wI1%d_H%d)*[dg1(:); dg2(:)];',det2,si,det2,si)) %SNR2_of phase+absorption
                    
              %% Hotelling observer study
        T_si = zeros(No_signals, No_realizations, 6);
        T_bi = zeros(No_signals, No_realizations, 6);
        T_s = zeros(No_realizations,6);
        T_b = zeros(No_realizations,6);
        
        disp(sprintf('Hotelling observer study'));
        
        for i =1:No_realizations
            max_phi12s = -Inf;
            max_phi12b = -Inf;
            max_b12s = -Inf;
            max_b12b = -Inf;
            max_phib12s = -Inf;
            max_phib12b = -Inf;
            max_I1s = -Inf;
            max_I1b = -Inf;
            max_I2s = -Inf;
            max_I2b = -Inf;
            max_I12s = -Inf;
            max_I12b = -Inf;
                
            for si=1:No_signals   
                gp=phi12s(:,:,i);
                eval(sprintf('T_si(si,i,1)=transpose(wp1%d)*gp(:);',det2))
                
                gb=b12s(:,:,i); 
                eval(sprintf('T_si(si,i,2)=transpose(wb1%d)*gp(:);',det2))
                eval(sprintf('T_si(si,i,3)=transpose(wpb1%d)*[gp(:);gb(:)];',det2))
    
                gp=phi12b(:,:,i);
                eval(sprintf('T_bi(si,i,1)=transpose(wp1%d)*gp(:);',det2))
    
                gb=b12b(:,:,i);
                eval(sprintf('T_bi(si,i,2)=transpose(wb1%d)*gb(:);',det2))
                eval(sprintf('T_bi(si,i,3)=transpose(wpb1%d)*[gp(:);gb(:)];',det2))
    
                g1=I1s(:,:,i);
                eval(sprintf('T_si(si,i,4)=transpose(wI%d)*g1(:);',det1))
    
                g2=I2s(:,:,i); 
                eval(sprintf('T_si(si,i,5)=transpose(wI%d)*g2(:);',det2))
                eval(sprintf('T_si(si,i,6)=transpose(wI1%d)*[g1(:);g2(:)];',det2))
    
                g1=I1b(:,:,i);
                eval(sprintf('T_bi(si,i,4)=transpose(wI%d)*g1(:);',det1))
    
                g2=I2b(:,:,i);
                eval(sprintf('T_bi(si,i,5)=transpose(wI%d)*g2(:);',det2))
                eval(sprintf('T_bi(si,i,6)=transpose(wI1%d)*[g1(:);g2(:)];',det2))
                
                if T_si(si,i,1) > max_phi12s
                    max_phi12s = T_si(si,i,1);
                end
                if T_bi(si,i,1) > max_phi12b
                    max_phi12b = T_bi(si,i,1);
                end
                if T_si(si,i,2) > max_b12s
                    max_b12s = T_si(si,i,2);
                end
                if T_bi(si,i,2) > max_b12b
                    max_b12b = T_bi(si,i,2);
                end
                if T_si(si,i,3) > max_phib12s
                    max_phib12s = T_si(si,i,3);
                end
                if T_bi(si,i,3) > max_phib12b
                    max_phib12b = T_bi(si,i,3);
                end
                if T_si(si,i,4) > max_I1s
                    max_I1s = T_si(si,i,4);
                end
                if T_bi(si,i,4) > max_I1b
                    max_I1b = T_bi(si,i,4);
                end
                if T_si(si,i,5) > max_I2s
                    max_I2s = T_si(si,i,5);
                end
                if T_bi(si,i,5) > max_I2s
                    max_I2s = T_bi(si,i,5);
                end
                if T_si(si,i,6) > max_I12s
                    max_I12s = T_si(si,i,6);
                end
                if T_bi(si,i,6) > max_I12s
                    max_I12s = T_bi(si,i,6);
                end
            end
            T_s(i,1) = max_phi12s;
            T_b(i,1) = max_phi12b;
            T_s(i,2) = max_b12s;
            T_b(i,2) = max_b12b;
            T_s(i,3) = max_phib12s;
            T_b(i,3) = max_phib12b;
            T_s(i,4) = max_I1s;
            T_b(i,4) = max_I1b;
            T_s(i,5) = max_I2s;
            T_b(i,5) = max_I2b;
            T_s(i,6) = max_I12s;
            T_b(i,6) = max_I12b;
        end
end

eval(sprintf('SNR2_em_1%d = 2 * mean(T_s-T_b).^2./(var(T_s)+var(T_b));',det2)) % da^2 in equation (10)
    
min_T=min(T_s, T_b); min_T=min(min_T);
max_T=max(T_s, T_b); max_T=max(max_T);

TPF=zeros(pt,6); 
FPF=TPF;
AUC = zeros(1,6);
for i=1:6
    thres=linspace(min_T(i),max_T(i),pt);
    for j=1:pt
        TPF(j,i)=size(find(T_s(:,i)>thres(j)),1)/No_realizations; 
        FPF(j,i)=size(find(T_b(:,i)>thres(j)),1)/No_realizations;     
    end
    eval(sprintf('AUC_1%d(i) = -trapz(FPF(:,i),TPF(:,i));',det2));
end

save([data_dir,sprintf('T_%d%d_%04g_%s_%s_phasegan.mat',det1,det2,sigma0,image_type,recon_alg)],'T_s','T_b'); % save T
save([data_dir,sprintf('ROC_%d%d_%04g_%s_%s_phasegan.mat',det1,det2,sigma0,image_type,recon_alg)],'TPF','FPF'); % save T
save([data_dir,sprintf('SNR2_%d%d_%04g_%s_%s_phasegan.mat',det1,det2,sigma0,image_type,recon_alg)],'SNR2*'); % save T
save([data_dir,sprintf('AUC_%d%d_%04g_%s_%s_phasegan.mat',det1,det2,sigma0,image_type,recon_alg)],'AUC*'); % save AUC

figure;
plot(FPF(:,1),TPF(:,1)); 
hold on;
plot(FPF(:,2),TPF(:,2),'r');
hold on;
plot(FPF(:,3),TPF(:,3),'g');
hold on;
plot(FPF(:,4),TPF(:,4),'m');
hold on;
plot(FPF(:,5),TPF(:,5),'c');
hold on;
plot(FPF(:,6),TPF(:,6),'k');

switch det2
        case 2
            legend(['\phi_1_',int2str(det2),' ,AUC=',num2str(sprintf('%.3f',AUC_12(1)))],['A_1_',int2str(det2),' ,AUC=',num2str(sprintf('%.3f',AUC_12(2)))],['\phi_1_',int2str(det2),'+A_1_',int2str(det2),' ,AUC=',num2str(sprintf('%.3f',AUC_12(3)))],['I_1 ,AUC=',num2str(sprintf('%.3f',AUC_12(4)))],['I_',int2str(det2),' ,AUC=',num2str(sprintf('%.3f',AUC_12(5)))],['I_1+I_',int2str(det2),' ,AUC=',num2str(sprintf('%.3f',AUC_12(6)))],'Location','southeast');
        case 3
            legend(['\phi_1_',int2str(det2),' ,AUC=',num2str(sprintf('%.3f',AUC_13(1)))],['A_1_',int2str(det2),' ,AUC=',num2str(sprintf('%.3f',AUC_13(2)))],['\phi_1_',int2str(det2),'+A_1_',int2str(det2),' ,AUC=',num2str(sprintf('%.3f',AUC_13(3)))],['I_1 ,AUC=',num2str(sprintf('%.3f',AUC_13(4)))],['I_',int2str(det2),' ,AUC=',num2str(sprintf('%.3f',AUC_13(5)))],['I_1+I_',int2str(det2),' ,AUC=',num2str(sprintf('%.3f',AUC_13(6)))],'Location','southeast');
        case 4
            legend(['\phi_1_',int2str(det2),' ,AUC=',num2str(sprintf('%.3f',AUC_14(1)))],['A_1_',int2str(det2),' ,AUC=',num2str(sprintf('%.3f',AUC_14(2)))],['\phi_1_',int2str(det2),'+A_1_',int2str(det2),' ,AUC=',num2str(sprintf('%.3f',AUC_14(3)))],['I_1 ,AUC=',num2str(sprintf('%.3f',AUC_14(4)))],['I_',int2str(det2),' ,AUC=',num2str(sprintf('%.3f',AUC_14(5)))],['I_1+I_',int2str(det2),' ,AUC=',num2str(sprintf('%.3f',AUC_14(6)))],'Location','southeast');
        case 5
            legend(['\phi_1_',int2str(det2),' ,AUC=',num2str(sprintf('%.3f',AUC_15(1)))],['A_1_',int2str(det2),' ,AUC=',num2str(sprintf('%.3f',AUC_15(2)))],['\phi_1_',int2str(det2),'+A_1_',int2str(det2),' ,AUC=',num2str(sprintf('%.3f',AUC_15(3)))],['I_1 ,AUC=',num2str(sprintf('%.3f',AUC_15(4)))],['I_',int2str(det2),' ,AUC=',num2str(sprintf('%.3f',AUC_15(5)))],['I_1+I_',int2str(det2),' ,AUC=',num2str(sprintf('%.3f',AUC_15(6)))],'Location','southeast');
        case 6
            legend(['\phi_1_',int2str(det2),' ,AUC=',num2str(sprintf('%.3f',AUC_16(1)))],['A_1_',int2str(det2),' ,AUC=',num2str(sprintf('%.3f',AUC_16(2)))],['\phi_1_',int2str(det2),'+A_1_',int2str(det2),' ,AUC=',num2str(sprintf('%.3f',AUC_16(3)))],['I_1 ,AUC=',num2str(sprintf('%.3f',AUC_16(4)))],['I_',int2str(det2),' ,AUC=',num2str(sprintf('%.3f',AUC_16(5)))],['I_1+I_',int2str(det2),' ,AUC=',num2str(sprintf('%.3f',AUC_16(6)))],'Location','southeast');
end

saveas(gcf,['ROC_curve_detector_pairs(',num2str(det1),',',num2str(det2),')_',image_type,'_phasegan.png']);
print('-depsc',['ROC_curve_detector_pairs(',num2str(det1),',',num2str(det2),')_',image_type,'_phasegan.eps']);

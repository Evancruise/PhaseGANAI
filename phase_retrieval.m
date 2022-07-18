function [phase, A]=phase_retrieval(recon_alg, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

data_dir = 'case1/';
load([data_dir,'Data0_UserDefinedParameters.mat']);

if nargin > 1
    phi_truth =  varargin{1};
    A_truth = varargin{2};
    I_contact = varargin{3};
    I = varargin{4};

    %noise = sigma0*randn(ZETAS,RAYS);
    for i_det = 1:length(d)
        eval(sprintf('I%d = I{3,i_det};', i_det));
        eval(sprintf('I%do = I{4,i_det};', i_det));
    end
end

%%

PAD=2; 
VIEWS = 1; 
SAMPLE_RATIO =1; 
RAYS = 128*SAMPLE_RATIO;
DELTA_X = 25e-6/SAMPLE_RATIO; 
d = [0.009, 0.078];%, 0.133];%, 0.012]; % m 

% 

%{
recon_alg = 'mixed';  %CTF, TIE, mixed

% Parameters for CTF and TIE
det=[1,2];        % Detector planes to be reconstructed 
grad= 'fft2';     % 'grad' or 'fft2': Specify the TIE implementation method 
threshold1=1e-9;  % threshold parameter for CTF
alpha_TIE=1;      % regularization parameter for TIE

% Parameters for mixed approach
delta_beta=1e2;  % delta/beta ratio 
alpha_mix=1e-16;
LP_cutoff = 0.5; 
LP_slope = .5e3; 
%}

%--------------------------------------------------------------------------------%
%% Load parameter and data

% load parameters
% load([data_dir,'Data_UserDefinedParameters_append.mat']);
% Load intensity data
warning off;

%% Ground truth
%phi_truth = -2*pi/lambda*proj_delta; 
%A_truth = 2*pi/lambda*proj_beta; 
%% Choose phase retrieval algorithm

switch recon_alg
    case 'CTF'
        [RAYS, ZETAS] = size(I1);
              
              %% Calculate frequency componenets
        u=(-1:2/RAYS/PAD:1-2/RAYS/PAD)*0.5/DELTA_X; 
        v=(-1:2/ZETAS/PAD:1-2/ZETAS/PAD)*0.5/DELTA_X; 
        [u,v]=meshgrid(u,v); u=fftshift(u);v=fftshift(v);
        %{        
        f2 = u.^2 + v.^2;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CTF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        fc1=f2.*d(det(1));  fc2=f2.*d(det(2)); 
        d1=d(det(1));  d2=d(det(2)); 
        I_ft1=eval(sprintf('fft2(I%d-1,PAD*ZETAS,PAD*RAYS);',det(1)));  % if pad >1, zero pad
        I_ft2=eval(sprintf('fft2(I%d-1,PAD*ZETAS,PAD*RAYS);',det(2)));  % if pad >1, zero pad
        

        Phi12=(cos(fc2).*I_ft1-cos(fc1).*I_ft2)./2./(sin(fc1-fc2)); Phi12(isinf(Phi12))=0; Phi12(isnan(Phi12))=0;  Phi12(abs(sin(fc1-fc2))<threshold1)=0; %Phi12(index12)=Phi12(index12).*filter12(index12);

        A12  =(sin(fc2).*I_ft1-sin(fc1).*I_ft2)./2./(sin(fc1-fc2)); A12(isnan(A12))=0; index2=find(abs(sin(fc1-fc2))<=threshold1);
        A12(index2)=(I_ft1(index2)*d2-I_ft2(index2)*d1)/2/(d1-d2); % Set the origin of A to a finite value

        phase=real(ifft2(Phi12)); phase=phase(1:ZETAS,1:RAYS); 

        A=real(ifft2(A12));A=A(1:ZETAS,1:RAYS);
        %}
        
        
        for j=1:length(d)
            eval(sprintf('I_ft%d = fft2(I%d,PAD*RAYS,PAD*RAYS);',j,j));
            eval(sprintf('I_ft%do = fft2(I%do,PAD*RAYS,PAD*RAYS);',j,j));
        end
            
        coschirp = zeros(length(d), RAYS*PAD, RAYS*PAD);
        sinchirp = zeros(length(d), RAYS*PAD, RAYS*PAD);
            
        for j = 1:length(d)
            coschirp(j,:,:) = cos(pi*lambda*(u.^2+v.^2).*d(j));
            sinchirp(j,:,:) = sin(pi*lambda*(u.^2+v.^2).*d(j));
        end
            
        A = zeros(RAYS*PAD, RAYS*PAD);
        B = zeros(RAYS*PAD, RAYS*PAD);
        C = zeros(RAYS*PAD, RAYS*PAD);
        I_sin = zeros(RAYS*PAD, RAYS*PAD);
        I_cos = zeros(RAYS*PAD, RAYS*PAD);
         
        for j = 1:length(d)
            eval(sprintf('A = A + reshape(sinchirp(j,:,:),[RAYS*PAD,RAYS*PAD]).*reshape(coschirp(j,:,:),[RAYS*PAD,RAYS*PAD]);')); 
            eval(sprintf('B = B + reshape(sinchirp(j,:,:),[RAYS*PAD,RAYS*PAD]).*reshape(sinchirp(j,:,:),[RAYS*PAD,RAYS*PAD]);')); 
            eval(sprintf('C = C + reshape(coschirp(j,:,:),[RAYS*PAD,RAYS*PAD]).*reshape(coschirp(j,:,:),[RAYS*PAD,RAYS*PAD]);')); 
            eval(sprintf('I_sin = I_sin + I_ft%d.*reshape(sinchirp(j,:,:),[RAYS*PAD,RAYS*PAD]);',j));
            eval(sprintf('I_cos = I_cos + I_ft%d.*reshape(coschirp(j,:,:),[RAYS*PAD,RAYS*PAD]);',j));
        end
        Delta = B.*C-A.*A;
            
        Phi12c=1./(2.*Delta+alpha_CTF).*(C.*I_sin-A.*I_cos);
        phi12c=real(ifft2(Phi12c)); 
        phase=phi12c(1:ZETAS,1:RAYS); %(RAYS/2+1:RAYS*3/2,RAYS/2+1:RAYS*3/2); 
            
        %A12c=1./(2.*Delta+al).*(A.*I_cos-B.*I_sin);
        %a12c=real(ifft2(A12c)); 
        %b=a12c(RAYS/2+1:RAYS*3/2,RAYS/2+1:RAYS*3/2);
        A = -0.5*log(I_contact);
        
        
    case 'TIE'
        
        [RAYS, ZETAS] = size(I1);
              %% Calculate frequency componenets

        u=(-1:2/RAYS/PAD:1-2/RAYS/PAD)*0.5/DELTA_X; 
        v=(-1:2/ZETAS/PAD:1-2/ZETAS/PAD)*0.5/DELTA_X; 
        [u,v]=meshgrid(u,v); u=fftshift(u);v=fftshift(v);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TIE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % TIE Ref: Langer's paper "Quantitative comparison of direct phase retrieval algorithms in in-line phase tomography"
        % Eq. (20)
        switch grad
            case 'grad'
                dIz = zeros(ZETAS,RAYS);
                dIz   = eval(sprintf('(I%d-I%d)/(d(%d)-d(%d))',det(2), det(1), det(2), det(1)));  
                F_dIz = -1./(4.*pi^2.*(u.^2+v.^2+alpha_TIE)) .*fft2(dIz, ZETAS*PAD, RAYS*PAD);
                inv_Lap_dIz = real(ifft2(F_dIz)); inv_Lap_dIz = inv_Lap_dIz(1:RAYS, 1:RAYS); 

               [fx,fy] = gradient(inv_Lap_dIz); 
                Fx = fx./I_contact; Fy = fy./I_contact; 
                div = divergence(Fx, Fy); 

                Phi = 1./(2.*pi.*lambda.* (u.^2+v.^2+alpha_TIE)).*fft2(div, ZETAS*PAD, RAYS*PAD);
                Phi = Phi ./(DELTA_X.^2);         
                phi = real(ifft2(Phi)); phase=phi(1:RAYS, 1:RAYS);
                A=-0.5*log(I_contact);
                %figure;imagesc(phase); colormap(gray)
                %figure; plot(phi_truth(64,:)); hold on;plot(phase(64,:),'r')
            case 'fft2'
                dIz = zeros(ZETAS,RAYS);
                dIz   = eval(sprintf('(I%d-I%d)/(d(%d)-d(%d))',det(2), det(1) ,det(2), det(1)));  
                F_dIz = -1./(4.*pi^2.*(u.^2+v.^2+alpha_TIE)) .*fft2(dIz, ZETAS*PAD, RAYS*PAD);
                inv_Lap_dIz = real(ifft2(F_dIz)); inv_Lap_dIz = inv_Lap_dIz(1:RAYS, 1:RAYS); 

                % Use FFT2 to compute gradient
                dfx = ifft2(1i*2*pi*u .* fft2(inv_Lap_dIz, ZETAS*PAD, RAYS*PAD)); dfx=dfx(1:ZETAS, 1:RAYS); dfx = dfx./I_contact; 
                dfy = ifft2(1i*2*pi*v .* fft2(inv_Lap_dIz, ZETAS*PAD, RAYS*PAD)); dfy=dfy(1:ZETAS, 1:RAYS); dfy = dfy./I_contact; 

                d2fx = ifft2(1i*2*pi*u .*fft2(dfx, ZETAS*PAD, RAYS*PAD)); 
                d2fy = ifft2(1i*2*pi*v .*fft2(dfy, ZETAS*PAD, RAYS*PAD)); 

                div = d2fx + d2fy; 
                Phi = 1./(2.*pi.*lambda.*(u.^2+v.^2+alpha_TIE)).*fft2(div, ZETAS*PAD, RAYS*PAD);   
                phi = real(ifft2(Phi)); phase=phi(1:RAYS, 1:RAYS);  
                A=-0.5*log(I_contact);
                %figure;imagesc(phase); colormap(gray)
                %figure; plot(phi_truth(64,:)); hold on;plot(phase(64,:),'r')
        end 
        
%         dIz   = eval(sprintf('(I%d-I%d)/(d(det(2))-d(det(1)))',det(2), det(1)));  
%         F_dIz = -2*pi/lambda * fft2(dIz,ZETAS*PAD, RAYS*PAD); 
%         
%         F_I0_dphase = F_dIz./(1i.*2.*pi.*(u+v+alpha)); 
%         I0_dphase = real(ifft2(F_I0_dphase)); I0_dphase = I0_dphase(1:RAYS, 1:RAYS); 
%         dphase = I0_dphase./I_contact; 
%         Phi=fft2(dphase, ZETAS*PAD, RAYS*PAD)./(1i.*2.*pi.*(u+v+alpha)); 
%         phi=real(ifft2(Phi)); phi=phi(1:ZETAS, 1:RAYS); 
%         
%         figure; plot(phi_truth(64,:)); hold on;plot(phi(64,:),'r')

        
    case 'mixed_approach'
        [RAYS, ZETAS] = size(I1);
              %% Calculate frequency componenets

        u=(-1:2/RAYS/PAD:1-2/RAYS/PAD)*0.5/DELTA_X; 
        v=(-1:2/ZETAS/PAD:1-2/ZETAS/PAD)*0.5/DELTA_X; 
        [u,v]=meshgrid(u,v); u=fftshift(u);v=fftshift(v);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% mixed %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        coschirp = zeros(RAYS*PAD, RAYS*PAD, length(d));
        sinchirp = zeros(RAYS*PAD, RAYS*PAD, length(d));

        coschirp_dfx = zeros(RAYS*PAD,RAYS*PAD, length(d)); %[0 for x in range(self.ND)]
        coschirp_dfy = zeros(RAYS*PAD,RAYS*PAD, length(d)); %[0 for x in range(self.ND)]

        for j = 1:length(d)
            coschirp(:,:,j) = cos(pi*lambda.*(u.^2+v.^2).*d(j));
            sinchirp(:,:,j) = sin(pi*lambda.*(u.^2+v.^2).*d(j));
        end

        sumAD2 = zeros(RAYS*PAD,RAYS*PAD);
        sumID = zeros(RAYS*PAD, RAYS*PAD);

        % equation (29)
        for j = 1:length(d)
            sumAD2 = sumAD2 + 4.*sinchirp(:,:,j).^2; % # Denominator Eq. 17

            sumID=sumID+ eval(sprintf('fft2(I%d-I%do,PAD*ZETAS,PAD*RAYS)',j,j));% Numerator Eq. 17
            coschirp_dfx(:,:,j) = coschirp(:,:,j).*(lambda.*d(j)).*1i.*u; % x direction
            coschirp_dfy(:,:,j) = coschirp(:,:,j).*(lambda.*d(j)).*1i.*v; % y direction
        end

        I_contact_pad=padarray(I_contact, [RAYS*(PAD-1), RAYS*(PAD-1)], I_contact(1),'post'); % Pad the intensity
        dfxI0 = real(ifft2(2*1i*pi.*u.*fft2(I_contact_pad))); dfxI0=dfxI0(1:ZETAS, 1:RAYS);   %./real(ifft2(fft2(I_contact,RAYS*PAD,RAYS*PAD))); % x direction: gradient(I_0)
        dfyI0 = real(ifft2(2*1i*pi.*v.*fft2(I_contact_pad))); dfyI0=dfyI0(1:ZETAS, 1:RAYS);%./real(ifft2(fft2(I_contact,RAYS*PAD,RAYS*PAD))); % y direction: gradient(I_0)

        

%         Regularization parameters
%         self.R = np.sqrt(np.square(self.fx) + np.square(self.fy))
%         self.LP_cutoff = 0.5
%         self.LP_slope = .5e3
%         self.LPfilter = 1 - 1/(1 + np.exp(-self.LP_slope *(self.R-self.LP_cutoff))) #Logistic filter
%         prior = np.fft.fft2(np.log(I0)*I0/2)
%         prior = self.LPfilter * prior       

        R =(u.^2+v.^2).*(RAYS*DELTA_X)^2; 
        LPfilter = 1 - 1./(1 + exp(-LP_slope *(R-LP_cutoff))); %Logistic filter
        A=-0.5*log(I_contact); 
        prior = fft2(A.*I_contact, ZETAS*PAD, RAYS*PAD);
        prior = LPfilter .* prior; 
        
        phase = zeros(RAYS, RAYS);        
        % Mixed approach iterations 
        for n = 1:2
            nominator_term = zeros(RAYS*PAD,RAYS*PAD);

            phase_dfxI0 = fft2(phase.*dfxI0, RAYS*PAD,RAYS*PAD); % x direction: F(gradient(phase*gradient(I_0)))
            phase_dfyI0 = fft2(phase.*dfyI0, RAYS*PAD,RAYS*PAD); % y direction: F(gradient(phase*gradient(I_0)))
            for j = 1:length(d)
                nominator_term = nominator_term +2*sinchirp(:,:,j).*(sumID-coschirp_dfx(:,:,j).*phase_dfxI0-coschirp_dfy(:,:,j).*phase_dfyI0);
            end
            
            nominator_term = nominator_term / (length(d)-1); 
  
            phase_I_n = (nominator_term+alpha_mix.*delta_beta.*prior)./(sumAD2+alpha_mix);
           % phase_I_n = (nominator_term)./(sumAD2);

            phase = real(ifft2(phase_I_n));  phase=phase(1:RAYS,1:RAYS); 
            phase = phase./I_contact;
            %figure; imagesc(phase);colormap(gray);
            
            %figure;plot(A_truth(RAYS/2,:)); hold on; plot(A(RAYS/2,:),'r')

        end
        %{
        figure(1);
        imagesc(phi_truth(crop,crop));
        figure(2);
        imagesc(A_truth(crop,crop))
        figure(3);
        imagesc(phase(crop,crop));
        figure(4);
        imagesc(A(crop,crop));
        % visualization: CTF 
        % figure; plot(phi_truth(RAYS,:), 'b'); hold on; plot(phase(RAYS,:), 'r');
        % figure; plot(A_truth(RAYS,:), 'b'); hold on; plot(A(RAYS,:), 'r');
        % visualization: TIE/mixed approach 
        figure(5); plot(phi_truth(RAYS/2,:), 'b'); hold on; plot(phase(RAYS/2,:), 'r'); hold off;
        figure(6); plot(A_truth(RAYS/2,:), 'b'); hold on; plot(A(RAYS/2,:), 'r'); hold off;
        %}
end

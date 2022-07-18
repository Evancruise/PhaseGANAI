function [phi_proj, A_proj, varargout]=BKS_U(data_dir,thickness_fiber,img_type, varargin)

if nargin > 3
    si= varargin{1};
end

% BKS_U computes wavefield for both H0 or H1.
% [U, I10, I20,... In0]=BKS_U(data_dir, "BKS")
% [U, I10, I20,... In0]=BKS_U(data_dir, "SKE")
% [U, I10, I20,... In0]=BKS_U(data_dir, "SKEs")
% [U, I10, I20,... In0]=BKS_U(data_dir, "SKS", I)  % I: Gaussian Signal
% [Input option: I] signal image
% [Ouput option: I10, I20, ... In0] Intensity on detector planes

load([data_dir,'Data0_UserDefinedParameters.mat']);
load('breast_values.mat','mu_fiber','mu_fat','mu_calc','density_fiber','density_fat','density_calc');

re = 2.82*10^-15;                                       % constant - electron radius [m]
u = 1.66*10^-24;                                        % constant - atomic mass unit [g]

No_signals=9;

mu_fat = mu_fat(E*2-1);              % fat attenuation coefficient [1/cm]
mu_fiber = mu_fiber(E*2-1);          % fiber attenuation coefficient [1/cm]
mu_calc = mu_calc(E*2-1);            % calcification attenuation coefficient [1/cm]

delta_fat = re*lambda.^2*density_fat*100^3/(4*pi*u);       % real part of the complex refractive index   [-]
delta_fiber = re*lambda.^2*density_fiber*100^3/(4*pi*u);   % real part of the complex refractive index   [-]
delta_calc = re*lambda.^2*density_calc*(100^3)/(4*pi*u);   % real part of the complex refractive index   [-]

clear re u density_fat density_fiber;

%I_background=CreateLumpyBackground(x_size,y_size,x_mm,y_mm,k_bar,N_bar,L_x,L_y,al,be,sigma_psi);
%[thickness_fiber, thickness_fat] = FindBackgroundComposition(E,I_background,breast_thickness,low_percentage,high_percentage,mid_percentage);

phi_proj=zeros(RAYS*PAD);
A_proj=zeros(RAYS*PAD); 

switch img_type
    case 'SKE'
        x0=[-RAYS/2:+1:RAYS/2-1]*delta_x;                                  %[m]
        y0=[-RAYS/2:+1:RAYS/2-1]*delta_x;
        [x,y]=meshgrid(x0,y0);                                             %[m]

        sphere_shape = 2*real(sqrt((d_sphere(lcv_d)/2)^2 - (x.^2+y.^2)));  %[m]
                clear x y;

        sphere_delta = zeros(size(sphere_shape)); sphere_mu = sphere_delta;

        sphere_delta = sphere_shape*delta_calc; %[m] *[-]
        sphere_mu = sphere_shape*100*mu_calc*0.24;  % Tune down the absorption so that it is not too much larger than the phase signal
        
        % breast_thickness [cm]
        phi_proj(RAYS/2+1:RAYS/2*3,RAYS/2+1:RAYS/2*3) = -(2*pi)/lambda*((breast_thickness-thickness_fiber-sphere_shape*100)*delta_fat/100 + sphere_delta + thickness_fiber*delta_fiber/100); phi_proj=(phi_proj*0.01); 
        A_proj(RAYS/2+1:RAYS/2*3,RAYS/2+1:RAYS/2*3)   =     ((breast_thickness-thickness_fiber-sphere_shape*100)*mu_fat + sphere_mu + thickness_fiber*mu_fiber)/2; A_proj=(A_proj);
    case 'SKEs'
        x0=[-RAYS/2:+1:RAYS/2-1]*delta_x;                                  %[m]
        y0=[-RAYS/2:+1:RAYS/2-1]*delta_x;
        [x,y]=meshgrid(x0,y0);                                             %[m]
        sz=sqrt(No_signals);
        %si=5;
        [row,col] = ind2sub([sz,sz],si);  ind=[floor(22*x_size/128), floor(65*x_size/128), floor(108*x_size/128)];
        xc=x(ind(row),ind(col)); yc=y(ind(row), ind(col));
        sphere_shape = 2*real(sqrt((d_sphere(lcv_d)/2)^2 - ((x-xc).^2+(y-yc).^2)));  %[m]
            clear x y;

        sphere_delta = zeros(size(sphere_shape)); sphere_mu = sphere_delta;

        sphere_delta = sphere_shape*delta_calc; %[m] *[-]
        sphere_mu = sphere_shape*100*mu_calc*0.24;  % Tune down the absorption so that it is not too much larger than the phase signal
    
        % breast_thickness [cm]
        phi_proj(RAYS/2+1:RAYS/2*3,RAYS/2+1:RAYS/2*3) = -(2*pi)/lambda*((breast_thickness-thickness_fiber-sphere_shape*100)*delta_fat/100 + sphere_delta + thickness_fiber*delta_fiber/100); phi_proj=(phi_proj*0.01);
        A_proj(RAYS/2+1:RAYS/2*3,RAYS/2+1:RAYS/2*3)   =     ((breast_thickness-thickness_fiber-sphere_shape*100)*mu_fat + sphere_mu + thickness_fiber*mu_fiber)/2; A_proj=(A_proj);

    case 'SKS'
        x0=[-RAYS/2:+1:RAYS/2-1]*delta_x;                                  %[m]
        y0=[-RAYS/2:+1:RAYS/2-1]*delta_x;
        [x,y]=meshgrid(x0,y0);                                             %[m]

        sphere_shape = FindSignalComposition(E, I, lcv_d);  %[cm]
        sphere_shape = sphere_shape/100; % convert [cm] to [m]
        clear x y;

        sphere_delta = zeros(size(sphere_shape)); sphere_mu = sphere_delta;

        sphere_delta = sphere_shape*delta_calc; %[m] *[-]
        sphere_mu = sphere_shape*100*mu_calc*0.24;  % Tune down the absorption so that it is not too much larger than the phase signal

        % breast_thickness [cm]
        phi_proj(RAYS/2+1:RAYS/2*3,RAYS/2+1:RAYS/2*3) = -(2*pi)/lambda*((breast_thickness-thickness_fiber-sphere_shape*100)*delta_fat/100 + sphere_delta + thickness_fiber*delta_fiber/100); phi_proj=(phi_proj*0.01);
        A_proj(RAYS/2+1:RAYS/2*3,RAYS/2+1:RAYS/2*3)   =     ((breast_thickness-thickness_fiber-sphere_shape*100)*mu_fat + sphere_mu + thickness_fiber*mu_fiber)/2; A_proj=(A_proj);


    case 'BKS'
        phi_proj(RAYS/2+1:RAYS/2*3,RAYS/2+1:RAYS/2*3) = -(2*pi)/lambda*((breast_thickness-thickness_fiber)*delta_fat/100 + thickness_fiber*delta_fiber/100); phi_proj=(phi_proj*0.01); 
        A_proj(RAYS/2+1:RAYS/2*3,RAYS/2+1:RAYS/2*3)   =     ((breast_thickness-thickness_fiber)*mu_fat + thickness_fiber*mu_fiber)/2; A_proj=(A_proj);        
end
            

    
%[proj_phase,proj_atten]=ForwardProjection(VIEWS,RAYS,delta_x, signal_type);% proj(ZETAS,RAYS,VIEWS); 
% CALCULATE PROJECTED DATA
% U_contact=exp(-k.*proj_atten + 1j.*k.*proj_phase);
% proj_atten = \int \beta dz
% k* proj_atten = k \int \beta dz = 0.5* \int \mu dz 

U = exp(-A_proj+1i*phi_proj);         % field [no units]
U_contact = exp(-A_proj);

[min_x,min_y] = find(abs(thickness_fiber-mean(thickness_fiber(:)))==min(abs(thickness_fiber(:)-mean(thickness_fiber(:)))));
 min_x = min_x(1); min_y = min_y(1);

%% APPLY FRESNEL PROPAGATOR
%  disp('Applying the Fresnel propagator...');
%  U_detector1 = Fresnel(U,U(min_x,min_y),delta_x, k, d(1));
%  I1_0 = (abs(U_detector1).^2); % intensity in the detector plane
%U_detector = Fresnel(U,U(min_x,min_y),delta_x, -(2*pi)/lambda, d(1));

nout= max(nargout,1);

if nout > 2
    I = cell(4,[]);
    noise = sigma0*randn(PAD*ZETAS,PAD*RAYS);
    for i_det=1:length(d)
         U_detector = Fresnelo(U,U(min_x,min_y),delta_x,lambda,d(i_det));
         U_detector_0 = Fresnelo(U_contact,U_contact(min_x,min_y),delta_x,lambda,d(i_det));
         I{1,i_det} = abs(U_detector).^2;
         I{2,i_det} = abs(U_detector_0).^2;
         I{3,i_det}=abs(U_detector).^2 + noise;
         I{4,i_det}=abs(U_detector_0).^2 + noise;
    end
    %varargout{1} = abs(U_contact).^2;
    %for i_det=1:length(d)
    %    U_detector = Fresnelo(U,U(min_x,min_y),delta_x,lambda,d(i_det));
    %    U_detector_0 = Fresnelo(U_contact,U_contact(min_x,min_y),delta_x,lambda,d(i_det));
    %    %U_detector = Fresnel(U,U(min_x,min_y),delta_x, k, d(i_det));
    %    eval(sprintf('I%d_0 = (abs(U_detector).^2);',i_det)) %    I2_0 = (abs(U_detector2).^2);                           % intensity in the detector plane
    %    varargout{2*(i_det-1)+2}=abs(U_detector).^2;
    %    varargout{2*(i_det-1)+3}=abs(U_detector_0).^2;
    %end
    varargout{1} = abs(U_contact).^2;
    varargout{2} = I;
end
end

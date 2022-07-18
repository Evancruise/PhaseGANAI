clear all; close all;clc;
tic; 
%% Wave propagation 
% ForwardProjection: generate projection data at any given incident frequency
% Fresnel: calculate the propagated wavefield at distance d
% main program: evaluate the weighted intensity data from a incident spectrum

%% User defined values
VIEWS = 1; 
SAMPLE_RATIO =1; 
RAYS = 128*SAMPLE_RATIO;
DELTA_X = 25e-6/SAMPLE_RATIO; 
d = [0.05, 0.12, 0.277, 0.352, 0.533];%, 0.133];%, 0.012]; % m 

%target = 'W55Al5'; spectra_length=140; % gauss16, gauss50, gauss85, spectra_length=120;
% W30Al1; W40Al4; W55Al5; W55Ti2; W70Al1; W70Zn1; spectra_length=140;
% Mo (80), W(120)
path=''; 
%path = 'ternary/'; 

%% Constants
c = 2.9979e8;                   % constant - speed of light [m/s]
h = 4.13566e-15;                % constant - Planck's constant [eV*s]

% lambda=1e-10; 
% delta1=2e-7; delta2=1.5e-7; 
% beta1=0.9e-10; beta2=1.5e-10; 

N_DO = size(d,2); 
ZETAS = RAYS; 

%% Load incident spectrum 
%fid=fopen([sprintf('Energy_%s.dat',target)],'rb'); E_target=fread(fid,[1,spectra_length],'double'); fclose(fid);
%fid=fopen([sprintf('Prob_%s.dat',target)],'rb'); S_target=fread(fid,[1,spectra_length],'double'); fclose(fid);

% Consider detector response
%load CsI_response.mat
%QDE_target = interp1(E_CsI,QDE,E_target,'spline'); 
%S_target = (S_target.*QDE_target)/sum(S_target.*QDE_target); 

E_target=60; % keV
lambda=c*h./(E_target*1e3);            % wavelength - [m]
%% Load breast tissue refractive indices
%{
load fat_tumor.mat
load calc_fiber.mat 

delta10=delta_fat; 
beta10=beta_fat; 

delta20=delta_tumor; 
beta20=beta_tumor; 

delta30=delta_fiber; 
beta30=beta_fiber; 

delta1=interp1(E,delta10,E_target,'spline'); 
beta1=interp1(E,beta10,E_target,'spline');

beta2=interp1(E,beta20,E_target,'spline'); 
delta2=interp1(E,delta20,E_target,'spline');

beta3=interp1(E,beta30,E_target,'spline'); 
delta3=interp1(E,delta30,E_target,'spline');

if strcmp(path,'binary/')==1;
    delta3(:)=0; beta3(:)=0;
    delta2=delta2-delta1; beta2=beta2-beta1;
elseif strcmp(path,'single/')==1;
    delta2(:)=0; beta2(:)=0;
    delta3(:)=0; beta3(:)=0;
else 
    delta2=delta2-delta1; beta2=beta2-beta1;
    delta3=delta3-delta1; beta3=beta3-beta1;
end
%}
%%
I = cell(1,N_DO);

[proj_delta,proj_beta] = ForwardProjection(VIEWS,RAYS,DELTA_X); 

U_contact = exp(-2*pi/lambda.*(1i.*proj_delta+proj_beta));              % field [no units]
U_contact_0=exp(-2*pi/lambda.*(proj_beta));                             % waveifled with 0 phase

for m=1: N_DO
    m
    I{m} = zeros(ZETAS,RAYS,VIEWS); 
    %for k=1:spectra_length
                
     %   delta = [delta1(k),0,delta2(k),delta2(k),delta3(k), delta3(k),0,0,0,0];
     %   beta = [beta1(k),0,beta2(k),beta2(k),beta3(k), beta3(k),0,0,0,0];

        % proj_delta(ZETAS,RAYS,VIEWS); 
        
        U_detector = Fresnel(U_contact,U_contact(1),DELTA_X,lambda,d(m));
        U_detector_0 = Fresnel(U_contact_0,U_contact_0(1),DELTA_X,lambda,d(m)); % wavefield on the detector with 0 phase
      %  I{m} = I{m} + (abs(U_detector).^2)*S_target(k);                           % intensity in the detector plane
        I{m} = I{m} + (abs(U_detector).^2);
        eval(sprintf('I%d=abs(U_detector).^2;',m))                        % intensity
        eval(sprintf('I%do=abs(U_detector_0).^2;',m))                        % zero phase intensity
     %end    
end
I_contact=abs(U_contact).^2; 
save([path,sprintf('I.mat')],'I')
%save([path,sprintf('I_%s.mat',target)],'I')
toc;
%% Test the validity of the intensity data
%{
PAD=2; 

u=(-1:2/RAYS/PAD:1-2/RAYS/PAD)*0.5/DELTA_X;
v=(-1:2/RAYS/PAD:1-2/RAYS/PAD)*0.5/DELTA_X;

[u,v]=meshgrid(u,v);
u=fftshift(u);v=fftshift(v);

x2=pi*lambda.*(u.^2+v.^2);

[phi12 b12]=phase_retrieve2(I{1},I{2},d(1),d(2),x2,RAYS,PAD,'gauss',0,1e-10,0); 
%}
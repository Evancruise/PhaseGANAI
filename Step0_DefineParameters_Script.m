warning off; clear all; close all; clc; pack;

%% Directory and simulation settings
data_dir = 'case1/';
data_phi_dir = 'phi/';
data_b_dir = 'b/';
data_phib_dir = 'phib/';
data_I_dir = 'I/';
data_files_dir = 'data/';

%% X-ray source
E = 15;                         % energy [keV]
c = 2.9979e8;                   % constant - speed of light [m/s]
h = 4.13566e-15;                % constant - Planck's constant [eV*s]
lambda=c*h./(E*1e3);            % wavelength - [m]

%% phantom
image_type = 'SKE'; % H1:'SKE', 'SKEs', or 'SKS'; H0: 'BKS'; Choose H1 or H0
No_signals = 1;                 % No of possible signal positions (if choose 'SKE' No_signals=1; if choose 'SKEs' No_signals=9)
No_realizations = 100;           % No of realizations

% d_sphere = zeros(1,100); for i=1:100; d_sphere(i) = 0.25e-6.*(1+2*i); end % the diameter of sphere for simulating signals
d_sphere= [10:10:150]*1e-6;     % [m]  the diameter of sphere for simulating signal
lcv_d = 5;                      % Signal diameter index, diameter = 30 um

breast_thickness = 1;           % Total breast thickness [cm]
low_percentage = 0.1;           % minimum percentage of epithelial content (0.1 recommended)
high_percentage = 0.9;          % maximum percentage of epithelial content (0.9 recommended)
mid_percentage = 0.5;           % mean percentage of epithelial content

% Lumpy background parameters
x_size           =    32;      %   size of image in x-dimension [pixels] (128 recommended)
y_size           =    32;      %   size of image in y-dimension [pixels] (128 recommended)
x_mm             =    38.4;     %   physical x size of the region of interest [mm] (38.4 recommended)
y_mm             =    38.4;     %   physical y size of the region of interest [mm] (38.4 recommended)
N_bar            =    20;       %   mean number of blobs in each cluster (20 recommended)
k_bar            =    150;      %   mean number of clusters (150 recommended)
L_x              =     5;       %   characteristic length of each blob along the x-axis [pixels] (5 recommended)
L_y              =     2;       %   characteristic length of each blob along the y-axis [pixels] (2 recommended)
al               =     2.1;     %   alpha blob slope parameter (2.1 recommended)
be               =     0.5;     %   beta blob shape parameter (0.5 recommended)
sigma_psi        =     12;      %   standard deviation of the blob position in each cluster [pixels] (12 recommended)

% load(['breast_values.mat']);

% mu_fat = mu_fat(E*2-1);           % fat attenuation coefficient [1/cm]
% mu_fiber = mu_fiber(E*2-1);       % fiber attenuation coefficient [1/cm]
% mu_calc = mu_calc(E*2-1);         % calcification attenuation coefficient [1/cm]
% delta_fiber = delta_fiber(E*2-1); % fiber refractive coefficient [1/cm]
% delta_fat = delta_fat(E*2-1);     % fat refractive coefficient [1/cm]
% delta_calc = delta_calc(E*2-1);   % calcification refractive coefficient [1/cm]


%% Detector
RAYS = 32;                     % the width of image
ZETAS = 32;                    % the height of image
PAD = 2;                        % number of padding
d = [0.009, 0.078, 0.132, 0.145, 0.252, 0.267]; %  the object-to-detector distance [m]
det=[1,2,3]; % The detector indices employed for reconstruction
det1 = det(1);
det2 = det(2);

delta_x = 75e-6; %10e-6; % the spatial resoltuion
sigma0 = 1e-4; % standard deviation of noise
crop = RAYS/2+1:RAYS*3/2;

%% Phase retrieval parameters for reconstruction algorithms for Step3 (i.e. CTF, TIE, mixed_approach)
recon_alg = 'mixed_approach';

% Parameters for CTF and TIE
% det=[1,2];                    % Detector planes to be reconstructed
grad= 'fft2';                   % 'grad' or 'fft2': Specify the TIE implementation method

threshold1=2e-7;                % threshold parameter for CTF
alpha_CTF=1e-16;

alpha_TIE=1;                    % regularization parameter for TIE

% Parameters for mixed approach
delta_beta=1e2;                 % delta/beta ratio
alpha_mix=1e-16;
LP_cutoff = 0.5;
LP_slope  = 0.5e3;

if ~(isfolder(data_dir))
   mkdir(data_dir);
   mkdir([data_dir,data_phi_dir]);
   mkdir([data_dir,data_b_dir]);
   mkdir([data_dir,data_I_dir]);
   mkdir([data_dir,data_files_dir]);
   mkdir([data_dir,data_phib_dir]);
end
   
save([data_dir,'Data0_UserDefinedParameters.mat']);
%clear all; close all;

filename = "Step0_DefineParameters_Script.m";
new_filename = [data_dir,'/parameters.dat'];
cmd = "cp " + '"' + filename + '" "' + new_filename + '"';
system(cmd); 




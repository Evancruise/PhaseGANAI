%% Step 1: Generate the measured data (CreateBackgroundandSignal.m)
% SKE/BKS : Signal-known-exactly/Background-known-statistically
% SKS/BKS : Signal-known-statistically/Background-known-statistically
function [I_background, K, Klocs, Nvec, Nlocx, Nlocy, Nang] = CreateBackground(x_size,y_size,all_set)
%% x_size           =       size of image in x-dimension [pixels] (128 recommended)
%% y_size           =       size of image in y-dimension [pixels] (128 recommended)
%% x_mm             =       physical x size of the region of interest [mm] (38.4 recommended)
%% y_mm             =       physical y size of the region of interest [mm] (38.4 recommended)
%% N_bar            =       mean number of blobs in each cluster (20 recommended)
%% k_bar            =       mean number of clusters (150 recommended)
%% L_x              =       characteristic length of each blob along the x-axis [pixels] (5 recommended)
%% L_y              =       characteristic length of each blob along the y-axis [pixels] (2 recommended)
%% al               =       alpha blob slope parameter (2.1 recommended)
%% be               =       beta blob shape parameter (0.5 recommended)
%% sigma_psi        =       standard deviation of the blob position in each cluster [pixels] (12 recommended)

%% Parameter settings (Table 2)
k_mean = 150; % number of clusters
N_mean = 20; % number of lumps
L_x = 5; % the "radius" of the ellipse with half-axes L_x
L_y = 2; % the "radius" of the ellipse with half-axes L_y
al = 2.1; % adjustable parameters (background)
be = 0.5; % adjustable parameters (background)
sigma_psi = 12; % the signal width
a = 100; % adjustable parameters (background)

%% Parameter settings (signal)
A = 0.2; % adjustable parameters (signal)
%h = 40; % adjustable parameters (signal)
%w = 0.5; % adjustable parameters (signal)
ws = 3; % adjustable parameters (signal)
w1 = 2; % major axis of signal
w2 = 2; % minor axis of signal
xpoints = 0; % signal position (x-axis) (0 indicattes the center of the whole image)
ypoints = 0; % signal position (y-axis) (0 indicattes the center of the whole image)
N_k_ske = 1; % the number of signal in SKE 
N_k_sks = 1; % the number of signal in SKS
range = 16; % the position interval of index for signal (i.e. range=16 indicates generating the signal position index with U(-8,8))

%% Create x y coordinate matrices
[x y] = meshgrid(-(x_size/2-0.5):(x_size/2-0.5),-(y_size/2-0.5):(y_size/2-0.5));
x = single(x); % x coordinate of rm
y = single(y); % y coordinate of rm

%% Create blank image
I_signal = single(zeros(x_size,y_size));
I_background = single(zeros(x_size,y_size));

%% Determine the number of clusters
K = single(poissrnd(k_mean)); % K ~ poiss(150)

%% Determine the cluster positions
K_pos = rand(2,K); % the coordinates of clusters (i.e. x-axis and y-axis)
K_pos(1,:) = round(K_pos(1,:)*(2*x_size-1)+1-x_size); % x-axis of clusters  
K_pos(2,:) = round(K_pos(2,:)*(2*y_size-1)+1-y_size); % y-axis of clusters

%% Find the number of "blobs" in each cluster
N_k = single(poissrnd(N_mean*ones(1,K))); % the number of blobs

%% Generate angle for "blob" rotation in each cluster
angles_b = 2*pi.*rand(1,K); % angle ~ U(0,2*pi) (uniform distribution) for BKS
%angles_ske = 2*pi.*rand(1,N_k_ske); % angle ~ U(0,2*pi) (uniform distribution) for SKE
angles_sks = 2*pi.*rand(1,N_k_sks); % angle ~ U(0,2*pi) (uniform distribution) for SKS

%% Determine the background type you want to apply
background_case = 2; 
% 1 for "manuscript_current_version.pdf"
% 2 for "Statistical texture synthesis of mammographic images with clustered lumpy backgrounds" by Bochud, Abbey, and Eckstein in Optics Express, Vol 4, No.1, pages 33-43.

switch background_case
    case 1
        %% background setting
     % -------------------------version 1---------------------------------
    for i=1:K
            %% Generate "blob" positions in each cluster
       c_position = single(round(sigma_psi*randn(2,N_k(i))./sqrt(2)));
 	        %% Add blobs for each cluster
	   for j=1:N_k(i)  
		  I_background = I_background + a*exp(-al*(sqrt(((cos(angles_b(i))*(x-K_pos(1,i)-c_position(1,j))-sin(angles_b(i))*(y-K_pos(2,i)-c_position(2,j)))/L_x).^2+((sin(angles_b(i))*(x-K_pos(1,i)-c_position(1,j))+cos(angles_b(i))*(y-K_pos(2,i)-c_position(2,j)))/L_y).^2).^be)); % equation (25)
       end
    end
    case 2
    % -------------------------version 2----------------------------------
    % assume that this means a square image
    dim = [x_size y_size];

    % integer sigma for edge adjustments
    sigint = round(sigma_psi);
    % adjust Kbar so that the average number within the image is 
    % the same but at the same time get rid of edge artifacts.
    % We do this by extending the border by 2sigma on each side.
    KbarPrime = k_mean *(1+(8*sigint)/dim(1) + (16*sigint^2)/dim(1)^2);
    % K is the number of clusters
    K = poissrnd(round(KbarPrime));

    % find each cluster location
    % Klocs = floor(rand(K,2) .* ([dim(ones(K,1),1)+10 dim(ones(K,1),2)+10])) - 4;
    % add padding around edges to get rid of edge effects
    Klocs = floor(rand(K,2) .* ([dim(ones(K,1),1)+4*sigint ...
      dim(ones(K,1),2)+4*sigint])) -2*sigint;

    % N is the number of blobs in each cluster
    Nvec  = poissrnd(N_mean * ones(K,1));
    Nlocs = zeros(K,2);
    Nang = zeros(K,1);
    
    for i=1:K,
    % blob locations
        Nlocs = sigma_psi*randn(Nvec(i),2) + Klocs(i*ones(Nvec(i),1),:);

        % blob angle -- each cluster has ONE random angle
        Nang(i)  = rand(1,1) * 2*pi;
        % You can also change the random value of this angle
        % Nang = randn(1,1)*.5 + pi/5;
  
        for j=1:Nvec(i),
            % center a grid on each cluster center
            [X,Y]=meshgrid(1:dim(2),1:dim(1));
            Nlocx(i,j) = Nlocs(j,1);
            Nlocy(i,j) = Nlocs(j,2);
            X = X - Nlocs(j,1);
            Y = Y - Nlocs(j,2);
            % calc radius..
            r = sqrt(X.^2 + Y.^2);
            % and angle.
            ang = atan2(Y,X);
            % get the characteristic length for each angle
            denom = ((L_x*ones(dim)).*(L_y*ones(dim)))./(sqrt(L_y^2*cos(ang-Nang(i)).^2 + L_x^2*sin(ang-Nang(i)).^2));
            % apply blob function.
            subimg = exp(-al*((r.^be)./(denom)));
            I_background = I_background+subimg;
        end
    end
end
end
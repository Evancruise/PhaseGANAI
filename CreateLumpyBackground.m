function I = CreateLumpyBackground(x_size,y_size,x_mm,y_mm,N_bar,k_bar,L_x,L_y,al,be,sigma_psi)

%%
%% Create a Clustered Lumpy Background
%% Adam M. Zysk
%% October-December 2009: Created
%% March 2010: Fixed bug that caused an error when generating non-square backgrounds.
%%
%% This code is modeled on the paper "Statistical texture synthesis of
%% mammographic images with clustered lumpy backgrounds" by Bochud, Optics
%% Express, 1999.
%%
%% I = CreateLumpyBackground(x_size,y_size,x_mm,y_mm,k_bar,N_bar,L_x,L_y,al,be,sigma_psi)
%%
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

%% Adjust physical dimensions for matrix size not 128x128
L_x = L_x*(38.4/x_mm)*(x_size/128);
L_y = L_y*(38.4/y_mm)*(y_size/128);
sigma_psi = sigma_psi*(76.8/(x_mm+y_mm))*((x_size+y_size)/256);

%% Create x y coordinate matrices
[x y] = meshgrid(-(x_size/2-0.5):(x_size/2-0.5),-(y_size/2-0.5):(y_size/2-0.5));
x = single(x); y = single(y);

%% Determine the number of clusters
K = single(poissrnd(4*k_bar));

%% Determine the cluster positions
K_pos = rand(2,K);
K_pos(1,:) = round(K_pos(1,:)*(2*x_size-1)+1-x_size); K_pos(2,:) = round(K_pos(2,:)*(2*y_size-1)+1-y_size);

%% Find the number of "blobs" in each cluster
N_k = single(poissrnd(N_bar*ones(1,K)));

%% Generate angle for "blob" rotation in each cluster
angles = 2*pi.*rand(1,K);

%% Create blank image
I = single(zeros(y_size,x_size));

%h = waitbar(0,'Adding lumps to the background...');

for i=1:K
    
    tic; %waitbar(i/K,h);

    %% Generate "blob" positions in each cluster
    c_position = single(round(sigma_psi*randn(2,N_k(i))./sqrt(2)));

    %% Add blobs for each cluster
    for j=1:N_k(i)  
        I = I + exp(-al*(sqrt(((cos(angles(i))*(x-K_pos(1,i)-c_position(1,j))-sin(angles(i))*(y-K_pos(2,i)-c_position(2,j)))/L_x).^2+((sin(angles(i))*(x-K_pos(1,i)-c_position(1,j))+cos(angles(i))*(y-K_pos(2,i)-c_position(2,j)))/L_y).^2).^be));
    end
    
    %clc; %disp(['Estimated Time Remaining = ' num2str((toc/60)*(K-i),4) ' minutes']);
    clear x_temp y_temp; 
end

%close(h);

%% Scale the image between zero and one
I = I-min(I(:));
I = I/max(I(:));
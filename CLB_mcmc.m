function b = CLB_mcmc(dim,Kbar,Nbar,Lx,Ly,alpha,beta,sigma)
%
% b = CLB(dim,Kbar,Nbar,Lx,Ly,alpha,beta,sigma)
%
%  Generate a clustered lumpy background (CLB) as described
%  in "Statistical texture synthesis of mammographic images with
%  clustered lumpy backgrounds," by Bochud, Abbey, and Eckstein
%  in Optics Express, Vol 4, No.1, pages 33-43.
%
%  dim   = size of resultant image
%  Kbar  = mean number of clusters
%  Nbar  = mean number of blobs in each cluster
%  Lx    = x-axis witdth of the exponential blob
%  Ly    = y-axis witdth of the exponential blob
%  alpha = first adjustable parameters of the blob --see reference
%  beta  = second adjustable parameters of the blob --see reference
%  sigma = blob extent in pixels
%
%  NOTE: To simulate example images like those obtained in the reference, 
%        use parameter values of dim=128, Kbar=150, Lbar=20, Lx=5, Ly=2, 
%        alpha=2.1, beta=0.5, and sigma=12.
%

% assume that this means a square image
if (length(dim) == 1)
  dim = [dim dim];
end

% integer sigma for edge adjustments
sigint = round(sigma);
% adjust Kbar so that the average number within the image is 
% the same but at the same time get rid of edge artifacts.
% We do this by extending the border by 2sigma on each side.
KbarPrime = Kbar *(1+(8*sigint)/dim(1) + (16*sigint^2)/dim(1)^2)
% K is the number of clusters
K = poissrnd(round(KbarPrime));

% find each cluster location
% Klocs = floor(rand(K,2) .* ([dim(ones(K,1),1)+10 dim(ones(K,1),2)+10])) - 4;
% add padding around edges to get rid of edge effects
Klocs = floor(rand(K,2) .* ([dim(ones(K,1),1)+4*sigint ...
      dim(ones(K,1),2)+4*sigint])) -2*sigint; ％ Klocs[K,2]

% N is the number of blobs in each cluster
Nvec  = poissrnd(Nbar * ones(K,1));

% zero out the image
b = zeros(dim);
Blob_locs = zeros(K,max(Nvec),3) % [# of cluster, max # blobs, (x,y,theta)]
Kang = zeros(K); % Random angle for each cluster

for i=1:K,
  % blob locations
  Nlocs = sigma*randn(Nvec(i),2) + Klocs(i*ones(Nvec(i),1),:);

  % blob angle -- each cluster has ONE random angle
  Nang  = rand(1,1) * 2*pi;
  % You can also change the random value of this angle
  % Nang = randn(1,1)*.5 + pi/5;
  for j=1:Nvec(i),
    % center a grid on each cluster center
    [X,Y]=meshgrid(1:dim(2),1:dim(1));
    X = X - Nlocs(j,1);
    Y = Y - Nlocs(j,2);
    % calc radius..
    r = sqrt(X.^2 + Y.^2);
    % and angle.
    ang = atan2(Y,X);
    % get the characteristic length for each angle
    denom = ((Lx*ones(dim)).*(Ly*ones(dim)))./(sqrt(Ly^2*cos(ang-Nang).^2 + ...
        Lx^2*sin(ang-Nang).^2));
    % apply blob function.
    subimg = exp(-alpha*((r.^beta)./(denom)));
    
    b=b+subimg;
  end
  fprintf(1,' %d / %d\r',[i K]);
  Blob_locs[i,1:Nvec(i),1:2]=Nlocs;
  Kang(i)= Nang;
end

fprintf(1,'\n');

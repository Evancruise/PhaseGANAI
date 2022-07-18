function [b,N] = LumpyBgnd(dim,Nbar,DC,lmpFcn,pars)
%
% b = LumpyBgnd(dim,Kbar,DC,lmpFcn,pars)
%
% Generate a type-I lumpy background as described in "J. P. Rolland, and 
% H. H. Barrett, 'Effect of random background inhomogeneity on observer
% detection performance', JOSA A, 9:649-658, 1992."
%
% dim    = size of resultant image
% Nbar   = mean number of lumps
% DC     = DC offset of resultant image 
% lmpFcn = Either 'GaussLmp' or 'CircLmp' for using
%          Gaussian or circ lumps, respectively
% pars   = [magnitude stddev] for 'GaussLmp'
%          [magnitude radius] for 'CircLmp'
%
% Example: b = LumpyBgnd([128 128],200,10,'GaussLmp',[1 10]);
%
% Note:  You can generate your own lump types by creating a unique
%        matlab file of the form  lmp = YourFunc(X,Y,pars), where
%        X and Y are the size of the image, and contain the x and y
%        coordinates to process your fcn.  The pars variable contains
%        any extra parameters you function needs.
%
% See Also: CLB, MVNLumpy
%

% assume square image if only one dim is given
if (length(dim)==1)
  dim = [dim dim];
end

% initialize the image
b = DC*ones(dim);

% N is the number of lumps
N = poissrnd(Nbar);

for i = 1:N,
  % random position of lump -- uniform throughout image
  pos   = (rand(1,2).*dim);
  % set up a grid of points
  [X,Y] = meshgrid((1:dim(1))-pos(1),(1:dim(2))-pos(2));
  % generate a lump centered at pos
  eval(['lmp = ',lmpFcn,'(X,Y,pars);']);
  % add it to the image
  b = b + lmp;
end



function lmp = GaussLmp(X,Y,pars)
lmp = pars(1)*exp(-.5*(X.^2+Y.^2)/pars(2)^2);

function lmp = CircLmp(X,Y,pars)
lmp = pars(1)*((X.^2+Y.^2)<=pars(2)^2);

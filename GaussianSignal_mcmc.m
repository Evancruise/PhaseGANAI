function img = GaussianSignal(dim,sigma,alpha,pos,theta)
%
%  img = GaussianSignal(dim,sigma,alpha,[pos],[theta])
%
%  -Generate an image of a Gaussian of dimensions dim,
%   with a standard deviation of sigma and magnitude of
%   alpha.
%  -If pos is not specified, then the Gaussian is centered
%   in the image.  pos is specified by (x,y) not (i,j)
%

if (length(dim) == 1)
  dim = [dim dim];
end

if (nargin == 3)
  % the flip is because we use (x,y) coordinates
  pos = fliplr((dim - [1 1]) / 2);
  theta = 0;
end

x = 0:(dim(2)-1);
y = 0:(dim(1)-1);
[X,Y] = meshgrid(x,y);

% Rotation matrix
% R=[cos(theta),sin(theta);  -sin(theta),cos(theta)];

X=X-pos(1); Y=Y-pos(2);
S=X*cos(theta)-Y*sin(theta);
T=X*sin(theta)+Y*cos(theta);

img = alpha* exp(-S.^2 /(2*sigma(1)^2) + T.^2/(2*sigma(2)^2));
%img = alpha* exp(-(1/(2*sigma^2))*((X-pos(1)).^2 + ...
                                    (Y-pos(2)).^2));

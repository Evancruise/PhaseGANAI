function img = GaussianSignal(dim,sigma,alpha,pos)
%
%  img = GaussianSignal(dim,sigma,alpha,[pos])
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
end

x = 0:(dim(2)-1);
y = 0:(dim(1)-1);
[X,Y] = meshgrid(x,y);

img = alpha* exp(-(1/(2*sigma^2))*((X-pos(1)).^2 + ...
    (Y-pos(2)).^2));
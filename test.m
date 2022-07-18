function [out1,varargout]=test(x,y,varargin)

out1=x+y;
if nargin>2
    s=varargin{1}; 
    out2=x+s; 
    varargout{1}=out2;
end
nargin
nargout
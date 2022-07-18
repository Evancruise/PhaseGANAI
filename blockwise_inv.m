% This code performs blockwise inversion of matrix H. 
% H=[A  B;
 %   C  D];

function [invH]=blockwise_inv(H) 

%rays=4*4; 
%H=randn(rays); 
 
rays=size(H,1); 

A=H(1:rays/2,1:rays/2);
B=H(1:rays/2,rays/2+1:end);
C=H(rays/2+1:end,1:rays/2);
D=H(rays/2+1:end,rays/2+1:end);

I=eye(rays/2);

invA=A\I;
tmp1=invA*B;
tmp2=(D-C*tmp1)\I;
tmp3=C*invA;
invH=[invA+tmp1*tmp2*tmp3, -tmp1*tmp2; -tmp2*tmp3, tmp2];

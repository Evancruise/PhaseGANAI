%% Analytically computing the forward projection
% The detector ranges from (-RAYS/2)*DELTA_X to (RAYS/2-1)*DELTA_X, the pixel size is DELTA_X
% (X: Pixel axis); (Y: Optical axis); (Z: Rotation axis); ~= SheppLoganPoly.cxx

function [proj_phase,proj_atten]=ForwardProjection(VIEWS,RAYS,DELTA_X)

%% INPUT Parameters
%VIEWS=90; RAYS=128; DELTA_X=1e-6;
ZETAS=RAYS; 
FACTOR = RAYS/2*DELTA_X; 
LENGTH=RAYS*DELTA_X;

%% Phantom definition  
N=10;  %the number of ellipsoids

Center = [
    0.0, 0., 0.0; % ellipsoid 1
    0.0, 0.0, 0.0; % ellipsoid 2
    -0.27, -0.20, 0.06; % ellipsoid 3
    0.14, 0.1, 0.06; % ellipsoid 4
    0.0, 0.07, -0.42; % ellipsoid 5
    -0.35, 0.05, -0.39; % ellipsoid 6
    -0.08, -0.605, -0.25; % ellipsoid 7
    0.06, -0.605, -0.25; % ellipsoid 8
    0.06, -0.45, 0.625; % ellipsoid 9
    0.08, -0.30, -0.20; % ellipsoid 10
    ]*0.40*FACTOR;

Semiaxes = [
    0.7, 0.9, 0.8; % 1
    0.62, 0.85, 0.75; % 2
    0.16, 0.22, 0.25; % 3
    0.14, 0.20, 0.20; % 4
    0.18, 0.12, 0.1; % 5
    0.08, 0.08, 0.20; % 6
    0.046, 0.023, 0.02; % 7
    0.046, 0.023, 0.02; % 8
    0.064, 0.08, 0.1; % 9
    0.22, 0.22, 0.1; % 10
    ]*0.40*FACTOR;

alpha = [0, 0, 108, 72, 0, 0, 0, 90, 90, 0]'/180*pi;

% get the refractive index from input
delta = [1.0,-0.98,0.3,0.4,0., 0,0,0,0,0]*1e-8;
beta = [1.2,-0.98,0.5,0.5,0., 0.,0.,0.,0.,0.]*1e-11;
%beta = [1.2,-0.98,0.5,0.5,0.63, 0.63,0.52,0.2,0.25,0.35]; 
  
A=Semiaxes(:,1);  % A: semiaxis of pixel axis
B=Semiaxes(:,2);  % B: semiaxis of ray axis
C=Semiaxes(:,3);  % C: semiaxis of vertical axis   

x1=Center(:,1);
y1=Center(:,2);
z1=Center(:,3);

x0=[-RAYS/2:+1:RAYS/2-1]*DELTA_X;
y0=[-RAYS/2:+1:RAYS/2-1]*DELTA_X;
z0=[RAYS/2:-1:-RAYS/2+1]*DELTA_X;

%%----Analytical foward projection
% for ellipses 

phi0=[0: VIEWS-1]*pi/VIEWS;                % VIEWS: number of tomographic view angles
s=sqrt(x1.^2+y1.^2);                       % length of the center to the origin (0,0)
gamma=atan2(y1,x1);                        % Angle between center vector to ray vector atan2(optical center/ray center)

[td,zd,phi]=meshgrid(x0,z0,phi0);			% projection data [z,x phi]; [ZETAS,RAYS1,VIEW], here x axis is like our t in 2D
[xo,yo,zo]=meshgrid(x0,y0,z0);				% create the coordinates for 3D object, f(y,x,z), (RAYS2,RAYS1,ZETAS)

f1=zeros(RAYS,RAYS,ZETAS); f2=f1;temp=f1;
proj_phase=zeros(ZETAS,RAYS,VIEWS); proj_atten=zeros(ZETAS,RAYS,VIEWS);

for j=1:N
    t=td-s(j).*cos(gamma(j)-phi);   % detector array
    phii=phi-alpha(j);   
    factor=(1-(zd-z1(j)).^2./C(j).^2); factor(factor<0)=0;
    a2=(B(j).^2.*sin(phii).^2+A(j).^2.*cos(phii).^2).*factor;
    
    proj_tmp=2.*delta(j).*A(j).*B(j).*factor./a2.*sqrt(a2-t.^2);  proj_tmp(isnan(proj_tmp))=0; proj_tmp(imag(proj_tmp)~=0)=0; proj_tmp(isinf(proj_tmp))=0;
    proj_phase=proj_phase+proj_tmp; clear proj_tmp

    proj_tmp=2.*beta(j).*A(j).*B(j).*factor./a2.*sqrt(a2-t.^2);  proj_tmp(isnan(proj_tmp))=0; proj_tmp(imag(proj_tmp)~=0)=0; proj_tmp(isinf(proj_tmp))=0;
    proj_atten=proj_atten+proj_tmp; clear proj_tmp 

    %  create true object function
    %{
    tr=(xo*cos(alpha(j))+yo*sin(alpha(j))); 
    sr=(xo*-sin(alpha(j))+yo*cos(alpha(j))); 
    tr1=(x1(j)*cos(alpha(j))+y1(j)*sin(alpha(j))); 
    sr1=(x1(j)*-sin(alpha(j))+y1(j)*cos(alpha(j)));
    tr=tr-tr1;
    sr=sr-sr1; 
    factor2=(1-(zo-z1(j)).^2./C(j).^2); factor2(find(factor2<0))=0;
%    if j==1        
%        bound=find((tr.^2./A(j).^2+sr.^2./B(j).^2)>factor2);  
% %     bound1=find((trr.^2./A(j).^2+srr.^2./B(j).^2)>factor2+0.1); 
%     clear trr srr trr1 srr1
%    end
    temp(find((tr.^2./A(j).^2+sr.^2./B(j).^2)<=factor2))=delta(j);
    temp(find((tr.^2./A(j).^2+sr.^2./B(j).^2)>factor2))=0;
    f1=f1+temp; 

    temp(find((tr.^2./A(j).^2+sr.^2./B(j).^2)<=factor2))=beta(j);
    temp(find((tr.^2./A(j).^2+sr.^2./B(j).^2)>factor2))=0;
    f2=f2+temp; 
    %}
end

% Testing the validity of the forward code
%{
a=proj_phase(RAYS/2,:,:);
a=shiftdim(a,1); 
a=circshift(a,[-1,0]);

img=iradon(a,phi0*180/pi,'linear','Hann',1,RAYS);img=flipud(img);
figure;imagesc(img);colormap(gray)
figure;imagesc(f1(:,:,RAYS/2));colormap(gray)

figure;plot(f1(:,58,RAYS/2));hold on;plot(img(:,58)/DELTA_X,'r')
%}
%% Numerical construct forward projection
%{
t=x0;
s=y0;
[t,s]=meshgrid(t,s);   %[s,t]

proj1=zeros(ZETAS,RAYS,VIEWS); proj2=zeros(ZETAS,RAYS,VIEWS); 

for i=1:VIEWS
    for j=1:ZETAS
	xi=t*cos(phi0(i))-s*sin(phi0(i));  % phi0=0, x=t; y=s; 
	yi=t*sin(phi0(i))+s*cos(phi0(i));  % 
	tmp=interp2(x0,y0,f1(:,:,j),xi,yi,'bilinear'); % interp2(row vector,column vector) [y,x]
    tmp(find(isnan(tmp)))=0;
	proj1(j,:,i) = sum(tmp,1)'*DELTA_X;
    end
end
%}
% --------------------------Backup codes -----------------------------
%%----Analytical foward projection for spheres
%{
% x: optical axis, y: detector array, z: optical axis
s=sqrt(y1.^2+z1.^2);   
gamma=atan2(z1,y1);   %atan2(optical center,detector center)

[xd,td]=meshgrid(x0,y0); %projection data [y,x]
proj1=zeros(rays,rays);proj2=zeros(rays,rays);
for j=1:N
  Rj=sqrt(R(j).^2-(x0-x1(j)).^2);
  Rj(find(imag(Rj)~=0))=0;
  Rj=repmat(Rj,rays,1);
  sj=repmat(s(j),rays,rays);
  gammaj=repmat(gamma(j),rays,rays);
  t=td-sj.*cos(gammaj);
  proj1_tmp=2.*delta(j).*sqrt(Rj.^2-t.^2);
  proj2_tmp=2.*beta(j).*sqrt(Rj.^2-t.^2); 
  proj1_tmp(find(isnan(proj1_tmp)))=0;  proj2_tmp(find(isnan(proj2_tmp)))=0;
  proj1_tmp(find(imag(proj2_tmp)~=0))=0;  proj2_tmp(find(imag(proj2_tmp)~=0))=0;
  proj1=proj1+proj1_tmp; proj2=proj2+proj2_tmp;
end
%----End of analytical forward projection for spheres

%% Test 3D analytical projection for parallel beams--- doesn't seem right
phi0=[0: VIEWS-1]*pi/VIEWS;                % VIEWS: number of tomographic view angles
s=sqrt(x1.^2+y1.^2);                       % length of the center to the origin (0,0)
gamma=atan2(y1,x1);                        % Angle between center vector to ray vector atan2(optical center/ray center)

%sample points
[td,zd,phi]=meshgrid(x0,z0,phi0);          % projection data [z,x phi]; [ZETAS,RAYS1,VIEW], here x axis is like our t in 2D

proj_phase2=zeros(ZETAS,RAYS,VIEWS); 

for j=1:N
    t=td-s(j).*cos(gamma(j)-phi);   % detector array
    phii=phi-alpha(j);   
    factor=(1-(zd-z1(j)).^2./C(j).^2); factor(factor<0)=0;
    a2=(B(j).^2.*sin(phii).^2+A(j).^2.*cos(phii).^2).*C(j).^2;
    
    proj_tmp=2.*delta(j).*A(j).*B(j).*C(j)./a2.*sqrt(a2-t.^2.*C(j).^2-(zd-z1(j)).^2.*a2./C(j).^2*7/8);  proj_tmp(find(isnan(proj_tmp)))=0; proj_tmp(find(imag(proj_tmp)~=0))=0; proj_tmp(find(isinf(proj_tmp)))=0;
    proj_phase2=proj_phase2+proj_tmp; clear proj_tmp

end
%----End of test 3D analytical projection
%}
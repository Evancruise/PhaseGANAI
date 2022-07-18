function U_z=Fresnel(T0,I0,Pixel_Size,lambda,DO)

%%  Fresnel Propagator Function
%%  Cheng-Ying Chou
%%  January 2009
%%
%%  U_z=Fresnel(T0,I0,Pixel_Size,lambda,DO)
%%
%%  T0              =   Input Wavefield (Must be a square matrix)
%%  I0              =   Input Intensity
%%  Pixel_Size      =   Pixel Size [m]
%%  lambda          =   Wavelength [m]
%%  DO           =   Propagation Distance [m]
%%  U_z             =   Output Wavefield

ZETAS = size(T0,1); RAYS=size(T0,2); VIEWS=size(T0,3); 
k = 2*pi/lambda; 
PAD = 2; 

u=(-1:2/RAYS/PAD:1-2/RAYS/PAD)*0.5/Pixel_Size;
v=(-1:2/RAYS/PAD:1-2/RAYS/PAD)*0.5/Pixel_Size;

[u,v]=meshgrid(u,v);
u=fftshift(u);v=fftshift(v);

if mod(RAYS,2) == 1
    T0 = [T0, I0*ones(ZETAS,1)];
    T0 = [T0; I0*ones(1,RAYS+1)];
    RAYS = size(T0,2);
end

T0 = padarray(T0, [RAYS*(PAD-1), RAYS*(PAD-1)], I0,'post');           % Pad the input wavefield matrix
T0 = fft2(T0);                                                        % 2-D FFT of the padded input wavefield

Filter = exp(-1i.*pi.*lambda.*DO.*(u.^2+v.^2)); 
Filter = repmat(Filter,[1,1,VIEWS]);

U_z = exp(1i*k*DO)*ifft2(T0.*Filter);                                  % Compute the output wavefield
U_z = U_z(1:RAYS,1:RAYS,:);
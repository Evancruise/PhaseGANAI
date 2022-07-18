function [thickness_calc] = FindSignalComposition(E, I, lcv_d)

%
% Extract Tissue Composition from a Clustered Lumpy Background
% Adam M. Zysk
% October/November 2009
%
% [thickness_fiber thickness_fat] = FindBackgroundComposition(I,breast_thickness,low_percentage,high_percentage,mid_percentage)
% 
% thickness_calc    =   thickness of calcification [cm]
% d_sphere(lcv_d)   =   maximum thickness of signal (calcification) [m]
%
% NOTE:  'breast_values.mat' must be in the appropriate directory for use
%        by this function.

%% Load attenuation coefficients

load('breast_values.mat','mu_fat','mu_fiber','mu_calc');
mu_fat = mu_fat(E*2-1);     % attenuation coefficient of adipose tissue [1/cm]
mu_fiber = mu_fiber(E*2-1); % attenuation coefficient of glandular tissue [1/cm]
mu_calc = mu_calc(E*2-1); % attenuation coefficient of calcification [1/cm]

%% Apply Beer's Law to set minimum and maximum intensity values
I_max = 1;
I_min = exp(-(mu_calc*d_sphere(lcv_d)));

%% Use Beer's Law and known total thickness to calculate calcification thicknesses
% log(I)     = -mu_calc * thickness_calc;
% log(I_min) = -mu_calc * d_sphere(lcv_d);         I= exp(-(mu_calc* thickness_calc))

thickness_calc = log(I)./log(I_min)*d_sphere(lcv_d)*100; % calcification thickness in cm


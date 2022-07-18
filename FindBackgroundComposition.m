function [thickness_fiber thickness_fat] = FindBackgroundComposition(E, I,breast_thickness,low_percentage,high_percentage,mid_percentage)

%
% Extract Tissue Composition from a Clustered Lumpy Background
% Adam M. Zysk
% October/November 2009
%
% [thickness_fiber thickness_fat] = FindBackgroundComposition(I,breast_thickness,low_percentage,high_percentage,mid_percentage)
% 
% breast_thickness  =   total breast thickness [cm]
% low_percentage    =   minimum percentage of epithelial content (0.1 recommended)
% high_percentage   =   maximum percentage of epithelial content (0.9 recommended)
% mid_percentage    =   mean percentage of epithelial content
% thickness_fiber   =   thickness of fibrous tissue [cm]
% thickness_fat     =   thickness of fat tissue [cm]
%
% NOTE:  'breast_values.mat' must be in the appropriate directory for use
%        by this function.

%% Load attenuation coefficients

load('breast_values.mat','mu_fat','mu_fiber');
mu_fat = mu_fat(E*2-1);     % attenuation coefficient of adipose tissue [1/cm]
mu_fiber = mu_fiber(E*2-1); % attenuation coefficient of glandular tissue [1/cm]

%% Apply Beer's Law to set minimum and maximum intensity values
I_max = exp(-(mu_fat*breast_thickness*(1-low_percentage)+mu_fiber*breast_thickness*low_percentage));
I_min = exp(-(mu_fat*breast_thickness*(1-high_percentage)+mu_fiber*breast_thickness*high_percentage));

%% Adjust the mean intensity value and apply min/max from above
while (abs(((mid_percentage-low_percentage)/(high_percentage-low_percentage))-mean(I(:)))>0.0001)
    I = I.^(mean(I(:))/((mid_percentage-low_percentage)/(high_percentage-low_percentage)));
end    
I = I*(I_max-I_min)+I_min;

%% Use Beer's Law and known total thickness to calculate fat and epithelial tissue thicknesses
thickness_fiber = -(log(I)+(mu_fat*breast_thickness))/(mu_fiber-mu_fat);
thickness_fat = breast_thickness-thickness_fiber;
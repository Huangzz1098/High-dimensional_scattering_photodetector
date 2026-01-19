%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Polarization datasets
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Citation for this code and algorithm:
% Zhengzhong Huang, Xiaogang Liu
% "On-chip Single-shot High-Dimensional Light Decoding",
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The code is written by Zhengzhong Huang, 2025
% The version of Matlab for this code is R2024a
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear all
close all

%% Load scattering layer

load('polarization_scatter','scattering_mask_abs','scattering_mask_thickness','J_matrix_1','J_matrix_2','J_matrix_3','J_matrix_4')
m = 800;
n = 800;
pitch = 4.*10.^(-6); % Pixel size
wavelength = 633.*10.^(-9);
z1 = 0.01; % Distance between the scattering plate and detector (m)
prop1 = Propagator(m, wavelength, pitch.*m, z1);

%%

RI_medium = 1.49; % Rackground refractive index
no = 1.658; % o-wave refractive index
ne = 1.486; % e-wave refractive index
RI_o = no - RI_medium;
RI_e = ne - RI_medium;

num_image = 10000; % number of picture

% The dichroic film
polar_angle = 45;
E_polarization = [(cosd(polar_angle)).^2, cosd(polar_angle) * sind(polar_angle);...
    cosd(polar_angle) * sind(polar_angle), (sind(polar_angle)).^2]; % The dichroic film

% The second layer
[polar_mask1, polar_mask2, polar_mask3, polar_mask4] = spatial_polarization(m, n); 

label_stoke_train5 = zeros(num_image, 20); % Label of stokes parameter

light1 = ones(m, n); % Incident light
s0=1; % Fully polarized light
for kk = 1:num_image

    fprintf('kk: %d\n', kk)
    number2 = num2str(kk);

    xyz = randn(3, 1); 
    xyz = xyz ./ vecnorm(xyz); % Arbitrary point on the Poincaré sphere

    % Stokes parameter
    s1 = xyz(1, :);
    s2 = xyz(2, :);
    s3 = xyz(3, :);

    % Label
    label_stoke = zeros(1, 20);
    label_stoke(length(label_stoke) / 5) = s1+1;
    label_stoke(2 * length(label_stoke) / 5) = s2+1;
    label_stoke(3 * length(label_stoke) / 5) = s3+1;
    label_stoke(4 * length(label_stoke) / 5) = s0+1;
    label_stoke = label_stoke ./ max(label_stoke);

    % Stokes to Jones matrix
    delta = atan2(s3, s2);
    theta = 0.5 * acos(s1);
    Ex = cos(theta);
    Ey = sin(theta).* exp(1i.*delta);
    E_incident = [Ex; Ey];

    [light1_1, light1_2] = polarization_propagation(m, n, E_polarization, polar_mask1,...
    polar_mask2, polar_mask3, polar_mask4, J_matrix_1, J_matrix_2,...
    J_matrix_3, J_matrix_4, E_incident, light1); % The light field output in the dichroic film 

    % The light field in the photodetector
    light2_1 = IFT(FT(light1_1) .* prop1);
    light2_2 = IFT(FT(light1_2) .* prop1);
    light3 = abs(light2_1) .^ 2+abs(light2_2) .^ 2;
    light4 = abs(light3(273:528, 273:528)) ./ max(max(abs(light3(273:528, 273:528))));

    label_stoke_train5(kk, :) = label_stoke;

    % Save intensity distribution
    path2 = ['datasets_full_stoke'];
    file_name = pad(string(number2), 12, "left","0");
    file_name = char(file_name);
    % imwrite(mat2gray(abs(light4)), [path2,'\',file_name,'.tif']);

end

% save('label_stoke_train5','label_stoke_train5')



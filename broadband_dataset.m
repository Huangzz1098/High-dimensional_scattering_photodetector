%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Spectrum datasets
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

load('scatter_sphere_20241212','scattering_mask_abs','scattering_mask_thickness')
scattering_mask_abs = scattering_mask_abs(101:900, 101:900); % Amplitude
scattering_mask_thickness = scattering_mask_thickness(101:900, 101:900); % Thickness
m = 800;
n = 800;
wave_range = 400:1:700; 
step_train = 8;
ip_wave_range_train = 1:step_train:size(wave_range, 2);
multi_wavelength = wave_range(1, ip_wave_range_train) .* 10.^(-9); % Wavelength range

pitch = 4 .* 10.^(-6); % Pixel size
z = 0.01; % Distance between the scattering plate and detector (m)

%%

num_image = 20000;
RI = 0.1;
n_pad = 5;
left_weight = 0.5 * (1 + cos(linspace(pi, 0, n_pad)));
right_weight = fliplr(left_weight);

for kk = 1:num_image
    fprintf('kk: %d\n', kk)
    number2 = num2str(kk);
    
    label_wavelength = zeros(1, size(multi_wavelength, 2));
    
    x = rand(1, 10);
    x=imresize(x, [1 38], 'bicubic');
    
    label_wavelength = x;
    
    left_pad = label_wavelength(1) * left_weight;
    right_pad = label_wavelength(end) * right_weight;
    
    label_wavelength2 = [left_pad, label_wavelength, right_pad];
    label_wavelength2 = imresize(label_wavelength2, [1 100], 'bicubic');
    
    light1 = ones(m, n);

    for jj = 1 : length(multi_wavelength)
        order_number = multi_wavelength(jj);
        scattering_mask_phase = (2 .* pi ./ multi_wavelength(jj)) .* RI .* scattering_mask_thickness .* 10.^(-6);
        scatter_mask = scattering_mask_abs .* exp(1i .* scattering_mask_phase);
        light2 = ones(size(scatter_mask));
        light2 = light2 .* scatter_mask;
        prop = Propagator(m, multi_wavelength(jj), pitch.*m, z);
        light2 = IFT(FT(light2) .* prop);
        light1 = light1 + label_wavelength(1, jj) .* abs(light2);
    end
    
    light1 = light1(floor(m/2)-128+1 : floor(m/2)+128 , floor(n/2)-128+1 : floor(n/2)+128);
    light3 = abs(light1) ./ max(max(abs(light1)));
    light3 = abs(light3).^2;
    
    path2 = ['datasets_broadband'];
    file_name = pad(string(number2), 12, "left", "0");
    file_name = char(file_name);
    % imwrite(mat2gray(abs(light3)), [path2,'\',file_name,'.tif']);
    label_broadband(kk, :) = label_wavelength2;

end

% save('label_broadband','label_broadband')
























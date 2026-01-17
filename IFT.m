%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2D IFFT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Citation for this code and algorithm:
% Zhengzhong Huang, Xiaogang Liu
% "On-chip Single-shot High-Dimensional Light Decoding",
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The code is written by Zhengzhong Huang, 2025
% The version of Matlab for this code is R2024a
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [out] = IFT(in)

[m, n] = size(in);
image1 = zeros(m, n);

for ii = 1 : m
    for jj = 1 : n
        image1(ii, jj) = exp(- 1i * pi * (ii + jj));
    end
end

FT = ifft2(image1 .* in);
out = image1 .* FT;

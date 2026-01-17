%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Angular spectrum propagation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Citation for this code and algorithm:
% Zhengzhong Huang, Xiaogang Liu
% "On-chip Single-shot High-Dimensional Light Decoding",
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The code is written by Zhengzhong Huang, 2025
% The version of Matlab for this code is R2024a
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [out] = Propagator(m, lambda, area, z)

out = zeros(m, m);

for ii = 1:m 
    for jj = 1:m
        alpha = lambda * (ii - m/2 -1) / area;
        beta = lambda * (jj - m/2 -1) / area;
        if ((alpha^2 + beta^2) <= 1)
            out(ii, jj) = exp(- 2 * pi * 1i * z * sqrt(1 - alpha^2 - beta^2) / lambda);
        end % if
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Polarization propagation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Citation for this code and algorithm:
% Zhengzhong Huang, Xiaogang Liu
% "On-chip Single-shot High-Dimensional Light Decoding",
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The code is written by Zhengzhong Huang, 2025
% The version of Matlab for this code is R2024a
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [out1, out2] = polarization_propagation(m, n, E_polarization, polar_mask1,...
    polar_mask2, polar_mask3, polar_mask4, J_matrix_1, J_matrix_2,...
    J_matrix_3, J_matrix_4, E_incident, E_field)

out1 = zeros(m, n);
out2 = zeros(m, n);

for ii = 1:m
    for jj = 1:n
        E_out = E_polarization * [polar_mask1(ii, jj), polar_mask2(ii, jj);...
        polar_mask3(ii, jj), polar_mask4(ii, jj)]*...
        [J_matrix_1(ii, jj), J_matrix_2(ii, jj); J_matrix_3(ii, jj), J_matrix_4(ii, jj)]*...
        [E_field(ii, jj), 0; 0, E_field(ii, jj)] * E_incident;
        out1(ii, jj) = E_out(1);
        out2(ii, jj) = E_out(2);
    end
end












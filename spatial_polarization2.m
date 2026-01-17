%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The dichroic film 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Citation for this code and algorithm:
% Zhengzhong Huang, Xiaogang Liu
% "On-chip Single-shot High-Dimensional Light Decoding",
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The code is written by Zhengzhong Huang, 2025
% The version of Matlab for this code is R2024a
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [out1, out2, out3, out4] = spatial_polarization2(m, n)

out1 = ones(m, n);
out2 = zeros(m, n);
out3 = zeros(m, n);
out4 = ones(m, n);

center_m = floor(m / 2);
center_n = floor(n / 2);

% for ii=1:m
%     for jj=1:n
%         if jj>n/2
%             out4(ii,jj)=exp(1i.*0.5.*pi./2);
%         end
%     end
% end

% out4(:,center_n:n)=exp(1i.*0.5.*pi./2);
out4(: , 1 : center_n) = 1i;









function [V] = PostVar(sigY,Vbar,tau)
%Calculates the posterior variance in Kalman filtering when learning about
%a constant
V = sigY^2*Vbar./(sigY^2*ones(size(tau))+Vbar*tau);
end


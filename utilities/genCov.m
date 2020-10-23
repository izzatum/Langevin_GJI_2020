function [C,s] = genCov(k,lambda0,alpha,tau,U)


    s = lambda0./(1:k).^(alpha) + tau;
    C = U*diag(s)*U';

end
function [X,G,tt] = samplerPrecondLipULA(funh,N,x0,tauk,P,coeff)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   ULA with adaptive Lipschitz step-length
%   
%   Reference: Izzatullah et al. (2020) Langevin dynamics MCMC solutions
%              for seismic inversion
%
%     
%   Implemented by  : Muhammad Izzatullah, KAUST
%   Version         : May 8, 2020
%
%   Input:
%   N         - Number of samples
%   x0        - Initial point, vector of dimension-by-one
%   tauk      - Set of initial step-length
%   nthin     - Number of thinning window
%
%   Output:
%   X         - Samples matrix, dimension-by-number of samples
%   G         - Samples gradient matrix, dimension-by-number of samples
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Size of random vector
[xi, xi2] = size(x0);        
% If x0 is a row vector, turn it to column vector
if xi2 > xi
    x0 = x0';
end

%% Initialisation
Ntau = length(tauk);     % Number of element of tauk
X = zeros(xi,N,Ntau);    % Samples
G = zeros(xi,N,Ntau);    % Gradients of log-normal distribution

for s = 1:Ntau
        
    k               = 1;           % Step
    xk              = x0;          % Initial step
    tau             = tauk(s);     % Step-length
    tt              = zeros(N,1); 
    theta           = Inf;         % Lipschitz-tau coefficient
    Pc              = chol(P);     % Cholesky factor of P
    [~,gk]          = funh(xk);
    pg              = P\gk;
    
    while k<=N

        disp(['Step: ', num2str(k)]);
        
        % ULA
        xk1     = xk - tau*pg + sqrt(2*tau)*(Pc\randn(xi,1));
        
        [fk1,gk1]   = funh(xk1);
        pg1 = P\gk1;

        if isnan(fk1) || isinf(fk1)
            X(:,k,s)  = 0;
            break;
        end
        
        % Compute Lipschitz-tau
        t1    = coeff*norm(xk1 - xk)/norm(pg1 - pg);
        t2    = sqrt(1 + theta)*tau;

        tk    = min(t1,t2);

        theta = tk/tau;
        tau   = tk;
        tt(k) = tau;
        
        xk    = xk1;
        pg    = pg1;
        
        X(:,k,s)  = xk1;
        G(:,k,s)  = -pg;

        k       = k + 1;  
    end
    
end

end
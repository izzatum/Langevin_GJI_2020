function [X,G,tt] = samplerLipULA(funh,N,x0,tauk,tmax,coeff)
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
xi = length(x0);  

%% Initialisation
Ntau = length(tauk);     % Number of element of tauk
X = zeros(xi,N,Ntau);    % Samples
G = zeros(xi,N,Ntau);    % Gradients of log-normal distribution

for s = 1:Ntau
        
    k               = 1;           % Step
    xk              = x0;          % Initial step
    tau             = tauk(s);     % Step-length
    %tmax            = tauk(s);
    tt = zeros(N,1);
    theta           = Inf;         % Lipschitz-tau coefficient
    [~,gk] = funh(xk);
    
    while k<=N

        disp(['Step: ', num2str(k)]);
        
        % ULA
        xk1     = xk - tau*gk + sqrt(2*tau)*randn(xi,1);
        
        [fk1,gk1]   = funh(xk1);

        if isnan(fk1) || isinf(fk1)
            X(:,k,s)  = 0;
            break;
        end
        
        % Compute Lipschitz-tau
        t1    = coeff*norm(xk1 - xk)/norm(gk1 - gk);
        t2    = sqrt(1 + theta)*tau;

        tk    = min(t1,t2);

        theta = tk/tau;
        tau   = min(tmax,tk);
        tt(k) = tau;
        
        X(:,k,s)  = xk1;
        G(:,k,s)  = -gk1;

        xk        = xk1;
        gk        = gk1;

        k       = k + 1;  
    end
    
end

end
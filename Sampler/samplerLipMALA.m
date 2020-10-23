function [X,G,acc,tt] = samplerLipMALA(funh,N,x0,tauk,tmax,coeff)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   MALA with adaptive Lipschitz step-length
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
%   acc       - Set of acceptance rate
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Size of random vector
xi = length(x0);

%% Initialisation
Ntau = length(tauk);     % Number of element of tauk
X = zeros(xi,N,Ntau);    % Samples
G = zeros(xi,N,Ntau);    % Gradients of log-normal distribution
acc = zeros(Ntau,1);     % Collections of acceptance rate

for s = 1:Ntau        
        
    k               = 1;           % Step
    xk              = x0;          % Initial step
    tau             = tauk(s);     % Step-length
    %tmax            = tauk(s);
    tt              = zeros(N,1); 
    accept          = 0;           % Acceptance
    theta           = Inf;         % Lipschitz-tau coefficient
    [f1,gk]         = funh(xk);
    fk = f1;
    
    tic;
    while k<=N

        disp(['Step: ', num2str(k)]);
        
         % MALA
         xk1     = xk - tau*gk + sqrt(2*tau)*randn(xi,1);
        
        [fk1,gk1]   = funh(xk1);

        if isnan(fk1) || isinf(fk1)
            X(:,k,s)  = 0;
            break;
        end
        
        pk1         = -(fk1); 
        pk          = -(fk);

        xqk1        = xk1 - tau*gk1;
        xqk         = xk - tau*gk;

        qk1         = -(0.25/tau)*norm(xk1 - xqk)^2; 
        qk          = -(0.25/tau)*norm(xk - xqk1)^2;

        %alpha       = min(1, ( pk1*qk )/( pk*qk1 ));
        %alpha       = log(pk1*qk) - log(pk*qk1);
        alpha       = (pk1 + qk) - (pk + qk1);

        % Metropolis-Hastings acceptance step
        if log(rand) < alpha
            % Compute Lipschitz-tau
            t1    = coeff*norm(xk1 - xk)/norm(gk1 - gk);
            t2    = sqrt(1 + theta)*tau;

            tk    = min(t1,t2);

            theta = tk/tau;
            tau   = min(tmax,tk);
            tt(k) = tau;
            
            fk    = fk1;
            xk    = xk1;
            gk    = gk1;

            accept  = accept + 1;
        end
        X(:,k,s)  = xk;
        G(:,k,s)  = -gk;
        k       = k + 1;
        
    end
    toc;
    
    % Calculate acceptance rate
    accept = accept/N;
    acc(s) = accept*100;
    disp(['Acceptance rate: ', num2str(acc(s))]);
    
end

end
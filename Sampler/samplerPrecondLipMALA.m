function [X,G,acc,tt] = samplerPrecondLipMALA(funh,N,x0,tauk,P,coeff)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   MALA
%   
%   Reference: Brooks et al (2002) Markov chain Monte Carlo method 
%              and its application
%
%     
%   Implemented by  : Muhammad Izzatullah, KAUST
%   Version         : May 8, 2020
%
%   Input:
%   N         - Number of samples
%   x0        - Initial point, vector of dimension-by-one
%   tauk      - Set of initial step-length
%   P         - Preconditioner
%   nthin     - Number of thinning window
%
%   Output:
%   X         - Samples matrix, dimension-by-number of samples
%   G         - Samples gradient matrix, dimension-by-number of samples
%   acc       - Set of acceptance rate
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
acc = zeros(Ntau,1);     % Collections of acceptance rate

for s = 1:Ntau        
        
    k               = 1;           % Step
    xk              = x0;          % Initial step
    tau             = tauk(s);     % Step-length
    tt              = zeros(N,1);
    accept          = 0;           % Acceptance
    theta           = Inf;         % Lipschitz-tau coefficient
    Pc              = chol(P);     % Cholesky factor of P
    
    [fk,gk] = funh(xk);
    pg = P\gk;
        
    tic;
    while k<=N

        disp(['Step: ', num2str(k)]);
        
        % MALA
        xk1     = xk - tau*pg + sqrt(2*tau)*(Pc\randn(xi,1));
        
        [fk1,gk1]   = funh(xk1);
        pg1 = P\gk1;

        if isnan(fk1) || isinf(fk1)
            X(:,k,s)  = 0;
            break;
        end
        
        pk1         = -fk1; 
        pk          = -fk;

        xqk1        = xk - tau*pg;
        xqk         = xk1 - tau*pg1;

        qk1         = -(0.25/tau)*norm(Pc*(xk1 - xqk1))^2; 
        qk          = -(0.25/tau)*norm(Pc*(xk - xqk))^2;

        %alpha       = min(1, ( pk1*qk )/( pk*qk1 ));
        alpha       = (pk1 + qk) - (pk + qk1);

        % Metropolis-Hastings acceptance step
        if log(rand) < alpha
            % Compute Lipschitz-tau
            t1    = coeff*norm(xk1 - xk)/norm(pg1 - pg);
            t2    = sqrt(1 + theta)*tau;

            tk    = min(t1,t2);

            theta = tk/tau;
            tau   = tk;
            tt(k) = tau;

            xk      = xk1;
            pg      = pg1;
            fk      = fk1; 
            
            accept  = accept + 1;
        end
            X(:,k,s)  = xk;
            G(:,k,s)  = -pg;
            k       = k + 1;
 
    end
    toc;
    
    % Calculate acceptance rate
    accept = accept/N;
    acc(s) = accept*100;
    disp(['Acceptance rate: ', num2str(acc(s))]);
    
end

end
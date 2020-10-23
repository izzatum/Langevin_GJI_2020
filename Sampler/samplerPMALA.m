function [X,G,acc] = samplerPMALA(funh,N,x0,tauk,P,lmin,lmax)
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

% Checking the input arguments
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
    accept          = 0;           % Acceptance
    Pc              = chol(P);     % Cholesky factor of P
    
    tic;
    while k<=N

        disp(['Step: ', num2str(k)]);

        [fk,gk] = funh(xk);
        
        pg = P\gk;
        pr = Pc\randn(xi,1);
        
        % MALA
        xk1     = xk - 0.5*tau*pg + sqrt(tau)*pr;
        
        [fk1,gk1]   = funh(xk1);
        pg1 = P\gk1;

        if isnan(fk1) || isinf(fk1)
            X(:,k,s)  = 0;
            break;
        end
        
        pk1         = exp(-fk1); 
        pk          = exp(-fk);

        xqk1        = xk - tau*pg;
        xqk         = xk1 - tau*pg1;

        qk1         = exp(-(0.25/tau)*norm(Pc*(xk1 - xqk1))^2); 
        qk          = exp(-(0.25/tau)*norm(Pc*(xk - xqk))^2);

        alpha       = min(1, ( pk1*qk )/( pk*qk1 ));

        % Metropolis-Hastings acceptance step
        if rand < alpha
            xk1 = thresholding(xk1,lmin,lmax);
            X(:,k,s)  = xk1;
            G(:,k,s)  = -pg1;

            xk      = xk1;

            k       = k + 1;
            accept  = accept + 1;
        else
            xk = thresholding(xk,lmin,lmax);
            X(:,k,s)  = xk;
            G(:,k,s)  = -pg1;
            k       = k + 1;
        end
 
    end
    toc;
    
    % Calculate acceptance rate
    accept = accept/N;
    acc(s) = accept*100;
    disp(['Acceptance rate: ', num2str(acc(s))]);
    
end

end
function [dksd,weight,dwksd] = compute_ksd(X, G, m, Lstr,Sstr, C, wk)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Kernel Discrepancy Kernel with Inverse Multiquadric (IMQ) Kernel
%   
%   k(x,y) = (c^{2} + ||x-y||^{2}_{2})^{-beta}
%
%   Reference: 1. Gorham and Mackey (2017) 
%                   Measuring Sample Quality with Kernels
%              2. Chwialkowski et al. (2016) 
%                   A Kernel Test of Goodness of Fit
%              3. Liu et al. (2016) 
%                   A Kernelized Stein Discrepancy for Goodness-of-fit Tests
%              4. Liu and Lee (2017) 
%                   Black-box importance sampling
%              5. Hodgkinson et al. (2020) 
%                   The reproducing Stein kernel approach for post-hoc 
%                   corrected sampling
%
%     
%   Implemented by  : Muhammad Izzatullah, KAUST
%   Version         : June 29, 2020
%
%   Input:
%   X      - n x d array, each row a sample from MCMC.
%   G      - n x d array, each row the gradient of the log target.
%   m      - desired number of sample points.
%   Lstr   - string for input (preconditioner), either 
%            'vanilla', 'med', 'sclmed', 'smpcov', 'bayesian', 'avehess' or
%            'choice'.
%   Sstr   - string for operator (preconditioner), either 
%            'vanilla', 'med', 'sclmed', 'smpcov', 'bayesian', 'avehess' or
%            'choice'.
%   C      - Preconditioner of your choice.
%
%   Output:
%   dksd   - discrepancy value at every number of sample points.
%   weight - n x 1 vector, containing the optimal weights for 
%               the sample points.
%   wksd   - discrepancy value of the optimally weighted 
%               empirical distribution based on X.  
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Dimensions of samples
[n,d] = size(X);

% Default preconditioner settings
if nargin == 3
    Sstr = "vanilla";
    Lstr = "vanilla";
end

if nargin < 6
    C = 1;
end

if nargin <= 6
    wk = 0;
end

%% Compute preconditioner
S = make_preconditioner(n,d,X,G,Sstr,C);
L = make_preconditioner(n,d,X,G,Lstr,C);

%% Compute KSD
% Stein kernel K(X,X) (i.e., Storing entries for m desired sample points)
%dksd = zeros(m,1);
K = zeros(n,m);


% Main loop
G    = G*S;
tmp0 = X/L;
StS   = S'*S;
tr   = trace(StS/L);


% Remove 'par' from parfor in case parallel toolbox is unavailable
%for k = 1:m
    
%    K = zeros(k);
    
    parfor j = 1:m%1:k
        xg = X(j,:)/L;
        
        % (X - Xj)*L^{-1}
        %tmp1 = tmp0(1:k,:) - xg;
        tmp1 = tmp0 - xg;
        
        % (X - Xj)^{T}*L^{-T}*S^{T}*S*L^{-1}*(X - Xj)
        A = sum((tmp1*S).^2,2); 

        % (G - Gj)^{T}*S^{T}*S*L^{-1}*(X - Xj)
        %B = sum((G(1:k,:) - G(j,:)).*(tmp1*S),2);
        B = sum((G - G(j,:)).*(tmp1*S),2); 

        % (X - Xj)^{T}*L^{-1}*(X - Xj)
        %C = sum((X(1:k,:) - X(j,:)).*tmp1, 2);
        C = sum((X - X(j,:)).*tmp1, 2);
        
        % G^{T}*G
        %D = G(1:k,:)*G(j,:)';
        D = G*G(j,:)';

        K(:,j) = D ./ ((1 + C).^(0.5)) ...
                 + (tr + B) ./ ((1 + C).^(1.5)) ...
                 - 3 * A ./ ((1 + C).^(2.5));
        
    end
    dksd = (sqrt(sum(triu(cumsum(K,2)),1))./(1:m))';
    % KSD values
%    dksd(k) = sqrt(mean(K,'all'));
%end


%% Compute weightage KSD

if wk
    tmp2 = K\ones(n,1);
    % Weights
    weight = tmp2./tmp2;
    % Weighted-KSD values
    dwksd = 1./sqrt(tmp2);
end


end
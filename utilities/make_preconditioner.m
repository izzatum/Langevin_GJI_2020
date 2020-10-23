function Precond = make_preconditioner(n,d,X,G,ptype,C)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Set up preconditioner matrix
%   
%   Reference: Riabiz et al (2020) Optimal Thinning of MCMC Output
%
%     
%   Implemented by  : Muhammad Izzatullah, KAUST
%   Version         : June 29, 2020
%
%   Input:
%   n      - number of sample points.
%   d      - number of dimensions.
%   X      - n x d array, each row a sample from MCMC.
%   G      - n x d array, each row the gradient of the log target.
%   ptype  - string for preconditioner, either 
%            'vanilla', 'med', 'sclmed', 'smpcov', 'bayesian', 'avehess' or
%            'choice'.
%   C      - Preconditioner of your choice.
%
%   Output:
%   Precond- d x d, symmetric positive definite preconditioner matrix  
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin == 5
    C = 0;
end

% Identity matrix
if strcmp(ptype,'vanilla')
    Precond = speye(d);

% Median heuristic
elseif strcmp(ptype,'med')
    % use a random subset of X of size n0
    n0 = 1000;
    ix = unique(round(linspace(1, n, n0)));
    ell = median(pdist(X(ix,:)));
    Precond = ell^2 * eye(d);

% Scaled median heuristic
elseif strcmp(ptype,'sclmed')
    % use a random subset of X of size n0
    n0 = 1000;
    ix = unique(round(linspace(1, n, n0)));
    ell = median(pdist(X(ix,:)));
    Precond = ell^2 * eye(d) / log(min(n, n0));
 
% Sample covarianc
elseif strcmp(ptype,'smpcov')
    Precond = cov(X);
    
% Bayesian learn
elseif strcmp(ptype,'bayesian')
    Precond = ((n-1)/(n-d-1)) * cov(X) + (1/(n-d-1)) * eye(d);
    
% Average Hessian
elseif strcmp(ptype,'avehess')
    Precond = inv((1/n) * (G') * G);
    
% Preconditioner of your choice
elseif strcmp(ptype,'choice')
    Precond = C;
    
% Throw error
else
    error('Incorrect preconditioner type.')
end

end
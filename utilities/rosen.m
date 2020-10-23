%% Rosenbrock Function 

function [f,g,H]=rosen(x)
% the 2D Rosenbrock function and its gradient
[m,n]=size(x);
% if (m ~= 2 || n ~= 1)
% 	error('Bad data sent to 2D Rosenbrock function')
% end

f = zeros(1,m);
g = zeros(n,m);

z    = [x(1,:).^2 - x(2,:); x(1,:)-0.25];
f    = 10*z(1,:).^2 + z(2,:).^4;
g   = [40.*x(1,:).*z(1,:) + 4*z(2,:).^3; -20.*z(1,:)];

if (m ~= 2 || n ~= 1)
    H  = 1;
else
    H = rosen_hess(x);
end

    
end

function H = rosen_hess(x)

    D = zeros(length(x),1);
    
    H = diag(-40*x(1,:),1) - diag(40*x(1,:),-1);
    D(1) = 120*x(1,:) - 40*x(2,:) + 12*(x(1,:)-0.25).^2;
    D(end) = 20;
    H = H + diag(D);
end
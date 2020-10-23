function [f,g] = sinedist(X)
    
    tmp = X(2,:) + sin(X(1,:));
    
    f = 0.5.*tmp.^2./0.005 + 0.5*(X(1,:).^2 + X(2,:).^2);
    
    g = [(1/0.005)*tmp.*cos(X(1,:)) + X(1,:);...
        (1/0.005)*tmp + X(2,:)];

end
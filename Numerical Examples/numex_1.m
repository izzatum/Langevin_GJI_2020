%% Setup Density
rng('default');

%% Gaussian distribution setup
A               = [2, 0.5; 0.5, 2];
L               = 1e-3*[0.5, 0; 2, 0];
b               = [1;1];


%% Function handler
% log-posterior
f               = @(x) (0.5*vecnorm(A*x - b).^2 + 0.5*vecnorm(L*x).^2);
% 1st derivative log posterior
g               = @(x) (A'*(A*x - b) + (L'*L)*x);
% 2nd derivative log posterior
H               = @(x) ((A'*A) + (L'*L));
% Function handler
funh            = @(x)funEval(x,f,g,H);

n               = 100;
x               = linspace(-1,1,n);
y               = linspace(-1,1,n);
[xx,yy]         = ndgrid(x,y);

F               = reshape(funh([xx(:), yy(:)]'),n,n);

density         = exp(-F);

figure;
subplot(2,2,1)
imagesc(x,y,density);axis square;
xlabel('m_{2}');
ylabel('m_{1}');

%% True mean and variance
c               = integral2(@(x1,x2) exp(-funh([x1;x2])),-Inf,Inf,-Inf,Inf);

mu              = [0;0];

mu(1)           = integral2(@(x1,x2) ...
                    x1.*exp(-funh([x1;x2])),-Inf,Inf,-Inf,Inf)/c;
mu(2)           = integral2(@(x1,x2) ...
                    x2.*exp(-funh([x1;x2])),-Inf,Inf,-Inf,Inf)/c;

sigma           = [0;0];

sigma(1)        = integral2(@(x1,x2) ...
                    (x1 - mu(1)).^2.*exp(-funh([x1;x2])),-Inf,Inf,-Inf,Inf)/c;
              
sigma(2)        = integral2(@(x1,x2)...
                    (x2 - mu(2)).^2.*exp(-funh([x1;x2])),-Inf,Inf,-Inf,Inf)/c;

%% Initiate MCMC samples
N = 30000;             % Number of samples
ns = 3;               % Number of MCMC methods
x0 = zeros(2,1);      % Initial point
tau = 2.6e-1;         % Step-size
X = zeros(2,N,ns);    % Samples
G = zeros(2,N,ns);    % Gradients of log-normal distribution
P= (A'*A) + (L'*L);

% Lipschitz - ULA
[X(:,:,1),G(:,:,1),tt] = samplerLipULA(funh,N,x0,tau,1,2^(-1/3));

% Lipschitz - MALA
[X(:,:,2),G(:,:,2),acclip,tt2] = samplerLipMALA(funh,N,x0,tau,1,2^(-1/3));

% MALA
[X(:,:,3),G(:,:,3),accmala] = samplerMALA(funh,N,x0,tau);

% Burnn-in
burnin = 15001;
X = X(:,burnin:end,:);
G = G(:,burnin:end,:);

% Mean and Std
Xmean = reshape(mean(X,2),2,ns);
Xstd = reshape(std(X,1,2),2,ns);
Xvar = reshape(var(X,1,2),2,ns);

% Compute ACF
acf1 = zeros(1000,ns);
acf1(:,1) = autocorr(X(1,:,1),'NumLags',999);
acf1(:,2) = autocorr(X(1,:,2),'NumLags',999);
acf1(:,3) = autocorr(X(1,:,3),'NumLags',999);

acf2 = zeros(1000,ns);
acf2(:,1) = autocorr(X(2,:,1),'NumLags',999);
acf2(:,2) = autocorr(X(2,:,2),'NumLags',999);
acf2(:,3) = autocorr(X(2,:,3),'NumLags',999);

%% Compute vanilla KSD for all samplers

[~,num_samples,~] = size(X);
nx = num_samples;
dksd = zeros(nx,ns+1);
C = inv(P);

for i = 1:ns
    tic
    dksd(:,i) = compute_ksd(X(:,:,i)', G(:,:,i)', nx, "sclmed", "vanilla");
    toc;
end

% draw i.i.d samples as benchmark
Xi = mvnrnd(mu,C,nx);
Gi = -(Xi - mu')*P;

tic
dksd(:,4) = compute_ksd(Xi, Gi, nx, "sclmed", "vanilla");
toc;

%% Plot KSD and ACF
figure;
subplot(2,2,1)
plot(acf1,'-','LineWidth',2); hold on;
plot(1:1000,zeros(1000,1),'k--','LineWidth',2); hold off;
axis tight;
legend('Lip-ULA','Lip-MALA','MALA','FontSize',11);
title('ACF: m_{1}','FontSize',14)
xlabel('Lag','FontSize',14)
ylabel('Sample ACF','FontSize',14)

subplot(2,2,2)
plot(acf2,'-','LineWidth',2); hold on;
plot(1:1000,zeros(1000,1),'k--','LineWidth',2); hold off;
axis tight;
legend('Lip-ULA','Lip-MALA','MALA','FontSize',11);
title('ACF: m_{2}','FontSize',14)
xlabel('Lag','FontSize',14)
ylabel('Sample ACF','FontSize',14)

subplot(2,2,[3,4])
loglog(1:nx,dksd,'-','LineWidth',2); hold on;
loglog(1:nx,10./sqrt(1:nx),'k-','LineWidth',2); hold off;
axis tight;
legend('Lip-ULA','Lip-MALA','MALA','i.i.d','Ref: 1/\surd{N}','FontSize',11);
title('KSD','FontSize',14)
xlabel('Number of samples, N','FontSize',14)
ylabel('Discrepancy','FontSize',14)

%% Plotting
rg = 1:10:num_samples;
% Density plot
figure;
subplot(1,3,1)
imagesc(x,y,density);hold on;axis square;
plot(X(2,rg,1),X(1,rg,1),'wo','MarkerFaceColor','white'); hold on;
plot(Xmean(2,1),Xmean(1,1),'rx','MarkerSize',7,'LineWidth',1.5); hold on;
plot(mu(2),mu(1),'k^','MarkerSize',7,'LineWidth',1.5);hold off;
xlabel('m_{2}');
ylabel('m_{1}');
title('Lipschitz ULA');

subplot(1,3,2)
imagesc(x,y,density);hold on;axis square;
plot(X(2,rg,2),X(1,rg,2),'wo','MarkerFaceColor','white'); hold on;
plot(Xmean(2,2),Xmean(1,2),'rx','MarkerSize',7,'LineWidth',1.5); hold on;
plot(mu(2),mu(1),'k^','MarkerSize',7,'LineWidth',1.5);hold off;
xlabel('m_{2}');
ylabel('m_{1}');
title('Lipschitz MALA');

subplot(1,3,3)
imagesc(x,y,density);hold on;axis square;
plot(X(2,rg,3),X(1,rg,3),'wo','MarkerFaceColor','white'); hold on;
plot(Xmean(2,3),Xmean(1,3),'rx','MarkerSize',7,'LineWidth',1.5); hold on;
plot(mu(2),mu(1),'k^','MarkerSize',7,'LineWidth',1.5);hold off;
xlabel('m_{2}');
ylabel('m_{1}');
title('MALA');

% Trace plot
figure;
subplot(3,2,1)
plot(1:num_samples,X(1,:,1));
xlabel('Number of iterations');
ylabel('m_{1}');
%title('ULA');
axis tight;

subplot(3,2,2)
plot(1:num_samples,X(2,:,1));
xlabel('Number of iterations');
ylabel('m_{2}');
%title('ULA');
axis tight;

subplot(3,2,3)
plot(1:num_samples,X(1,:,2));
xlabel('Number of iterations');
ylabel('m_{1}');
%title('Local Lipschitz ULA');
axis tight;

subplot(3,2,4)
plot(1:num_samples,X(2,:,2));
xlabel('Number of iterations');
ylabel('m_{2}');
%title('Local Lipschitz ULA');
axis tight;

subplot(3,2,5)
plot(1:num_samples,X(1,:,3));
xlabel('Number of iterations');
ylabel('m_{1}');
%title('MALA');
axis tight;


subplot(3,2,6)
plot(1:num_samples,X(2,:,3));
xlabel('Number of iterations');
ylabel('m_{2}');
%title('MALA');
axis tight;
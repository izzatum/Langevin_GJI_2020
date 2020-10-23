%% Seismic Wave Tomography
%clear; clc; close all;
rng('default');
%% Configurations
% Grid size (n-by-n)
n = 50; 
% Number of sources
s = 30;
% Number of receivers
p = 50;
% Frequency
omega = 10;
% Regularization value
alpha = 1;
% Prior model covariance factor
L = LaplacianMatrix2D(n^2);
% Setup modeling configurations
[A,b,x] = seismicwavetomo(n,s,p,omega);
% Adding noise to data (default noise level = 0.01)
[bn, NoiseInfo] = PRnoise(b);
% Data covariance Cd*eye(n)
Cd1 = (NoiseInfo.snr*NoiseInfo.level);
Cd = Cd1^2;

%% Function handler
% log-posterior
f               = @(x) 0.5*norm(A*x - bn)^2/Cd + 0.5*alpha*norm(L*x)^2;
% 1st derivative log posterior
g               = @(x) A'*(A*x - bn)/Cd + alpha*(L'*L)*x;
% Function handler
funh            = @(x)funEval(x,f,g,@(x) 1);

%% Hessian and Cholesky factor computation
% Hessian 
H    = (A'*A)/Cd + alpha*(L'*L);
% Cholesky factor of Hessian
Hc   = chol(H,'upper');
% Inverse Hessian 
Hi   = inv(H);
% SVD
S    = svd(full(H));

%% Initiate SGLD samples
N = 5e4;                       % Number of samples
ns = 3;                         % Number of samplers
x0 = zeros(size(x));            % Initial point
tau = 1/S(1);                        % Step-length
xi = length(x);                 % Size of random vector
X = zeros(xi,N,ns);             % Samples
G = zeros(xi,N,ns);             % Gradients of log-normal distribution

%% Compute MAP
map_opts = struct;
map_opts.x0 = x0;
map_opts.x_true = x;
map_opts.RegParam = alpha;
map_opts.RegMatrix = 'Laplacian2D';

[xmap,info] = IRcgls(A,bn,map_opts);

%% Plotting 2
y1  = ones(p/2,1);
y2  = ones(s,1);
xr = round(linspace(2,n,p/2));
zr = round(linspace(2,n,p/2));
xs = round(linspace(1,n,s));

figure;
subplot(221)
imagesc(reshape(x,n,n),[0, 1]);
colorbar;
axis equal tight;
xlim([1,n])
ylim([1,n])
xlabel('Lateral','FontSize', 16);
ylabel('Depth','FontSize', 16);
 
subplot(222)
im = imagesc(x0,[0, 1]); hold on;
plot(xr,y1,'kv','LineWidth',2,'markersize',12); hold on;
plot(y1,zr,'kv','LineWidth',2,'markersize',12); hold on;
plot(n*y2,xs,'r*','LineWidth',2,'markersize',12); hold off;
im.AlphaData = 0;
axis equal tight;
xlim([1,n])
ylim([1,n])
xlabel('Lateral','FontSize', 16);
ylabel('Depth','FontSize', 16);
grid on;
 
subplot(223)
imagesc(reshape(xmap,n,n),[0, 1]);
colorbar;
axis equal tight;
xlim([1,n])
ylim([1,n])
xlabel('Lateral','FontSize', 16);
ylabel('Depth','FontSize', 16);
 
subplot(224)
imagesc(reshape(sqrt(diag(Hi)),n,n));
colorbar;
axis equal tight;
xlim([1,n])
ylim([1,n])
xlabel('Lateral','FontSize', 16);
ylabel('Depth','FontSize', 16);

%% Start sampling

% Lipschitz - ULA
%[X(:,:,1),G(:,:,1)] = samplerULA(funh,N,xmap,tau);
%[X(:,:,1),G(:,:,1)] = samplerLipULA(funh,N,xmap,tau); %length(xmap)^(-1/3)
[X(:,:,1),G(:,:,1),tt] = samplerPrecondLipULA(funh,N,xmap,tau,H,length(xmap)^(-1/3));

% Lipschitz - MALA
%[X(:,:,2),G(:,:,2)] = samplerSGLD(funh,N,xmap,tau);
%[X(:,:,2),G(:,:,2),acclip] = samplerLipMALA(funh,N,xmap,tau);
[X(:,:,2),G(:,:,2),acclip,tt2] = samplerPrecondLipMALA(funh,N,xmap,tau,H,length(xmap)^(-1/3));

% MALA
%[X(:,:,3),G(:,:,3),accmala] = samplerMALA(funh,N,xmap,tau);
[X(:,:,3),G(:,:,3),accmala] = samplerPrecondMALA(funh,N,xmap,tau,H);

% Mean and Std
Xmean = reshape(mean(X,2),xi,ns);
Xvar = reshape(var(X,1,2),xi,ns);

%% Plotting 2
n = 50;
figure;
subplot(321)
% Lip-ULA
imagesc(reshape(Xmean(:,1),n,n),[0, 1]);
colorbar;
axis equal tight;
xlim([1,n])
ylim([1,n])
title('Lip-ULA: Sample mean','FontSize', 16);
xlabel('Lateral','FontSize', 16);
ylabel('Depth','FontSize', 16);
 
subplot(322)
% Lip-ULA
imagesc(reshape(Xvar(:,1)./max(Xvar(:,1)),n,n),[0, 1]);
colorbar;
axis equal tight;
xlim([1,n])
ylim([1,n])
title('Lip-ULA: Sample Variance','FontSize', 16);
xlabel('Lateral','FontSize', 16);
ylabel('Depth','FontSize', 16);
 
subplot(323)
% Lip-MALA
imagesc(reshape(Xmean(:,2),n,n),[0, 1]);
colorbar;
axis equal tight;
xlim([1,n])
ylim([1,n])
title('Lip-MALA: Sample mean','FontSize', 16);
xlabel('Lateral','FontSize', 16);
ylabel('Depth','FontSize', 16);
 
subplot(324)
% Lip-MALA
imagesc(reshape(Xvar(:,2)./max(Xvar(:,2)),n,n),[0, 1]);
colorbar;
axis equal tight;
xlim([1,n])
ylim([1,n])
title('Lip-MALA: Sample Variance','FontSize', 16);
xlabel('Lateral','FontSize', 16);
ylabel('Depth','FontSize', 16);
 
subplot(325)
% MALA
imagesc(reshape(Xmean(:,3),n,n),[0, 1]);
colorbar;
axis equal tight;
xlim([1,n])
ylim([1,n])
title('MALA: Sample mean','FontSize', 16);
xlabel('Lateral','FontSize', 16);
ylabel('Depth','FontSize', 16);
 
subplot(326)
% MALA
imagesc(reshape(Xvar(:,3)./max(Xvar(:,3)),n,n),[0, 1]);
colorbar;
axis equal tight;
xlim([1,n])
ylim([1,n])
title('MALA: Sample Variance','FontSize', 16);
xlabel('Lateral','FontSize', 16);
ylabel('Depth','FontSize', 16);

%% Compute KSD and ACF for all samplers
% Compute ACF
acf1 = zeros(5000,ns);
acf1(:,1) = autocorr(X(1215,:,1),'NumLags',4999);
acf1(:,2) = autocorr(X(1215,:,2),'NumLags',4999);
acf1(:,3) = autocorr(X(1215,:,3),'NumLags',4999);

acf2 = zeros(5000,ns);
acf2(:,1) = autocorr(X(1225,:,1),'NumLags',4999);
acf2(:,2) = autocorr(X(1225,:,2),'NumLags',4999);
acf2(:,3) = autocorr(X(1225,:,3),'NumLags',4999);

X1 = X(:,1:50:end,:);
G1 = G(:,1:50:end,:);

[~,num_samples,~] = size(X1);
nx = num_samples;
dksd = zeros(nx,ns+1);

Xiid = mvnrnd(xmap,Hi,num_samples); 
Giid = -(Xiid - xmap')*H;

dksd(:,4) = compute_ksd(Xiid, Giid, nx, "sclmed","vanilla", Hi);

for i = 1:ns
    tic
    %dksd1(:,i) = ksd(X(:,:,i)', (H*G(:,:,i))', nx,'sclmed');
    dksd(:,i) = compute_ksd(X1(:,:,i)', (H*G1(:,:,i))', nx, "sclmed","vanilla", Hi);
    toc;
end

%% Plotting

% Plot KSD and ACF
figure;
subplot(1,2,1)
plot(acf1,'-','LineWidth',1.5); hold on;
axis tight;
legend('Lip-ULA','Lip-MALA','MALA');
title('ACF: m_{1215}','FontSize', 16)
xlabel('Lag','FontSize', 16)
ylabel('Sample ACF','FontSize', 16)

subplot(1,2,2)
plot(acf2,'-','LineWidth',1.5); hold on;
axis tight;
legend('Lip-ULA','Lip-MALA','MALA');
title('ACF: m_{1225}','FontSize', 16)
xlabel('Lag','FontSize', 16)
ylabel('Sample ACF','FontSize', 16)

figure;
loglog(1:nx,dksd,'-','LineWidth',1.5); hold on;
loglog(1:nx,44./sqrt(1:nx),'k-','LineWidth',2); hold off;
axis equal;
legend('Lip-ULA','Lip-MALA','MALA','iid','Ref: 1/\surd{N}');
title('KSD','FontSize', 16)

%% Trace plots
num_samples = N;
figure;
subplot(3,2,1)
plot(1:num_samples,X(1215,:,1));
xlabel('Number of iterations','FontSize', 16);
ylabel('m_{1215}','FontSize', 16);
%title('ULA');
axis tight;

subplot(3,2,2)
plot(1:num_samples,X(1225,:,1));
xlabel('Number of iterations','FontSize', 16);
ylabel('m_{1225}','FontSize', 16);
%title('ULA');
axis tight;

subplot(3,2,3)
plot(1:num_samples,X(1215,:,2));
xlabel('Number of iterations','FontSize', 16);
ylabel('m_{1215}','FontSize', 16);
%title('Local Lipschitz ULA');
axis tight;

subplot(3,2,4)
plot(1:num_samples,X(1225,:,2));
xlabel('Number of iterations','FontSize', 16);
ylabel('m_{1225}','FontSize', 16);
%title('Local Lipschitz ULA');
axis tight;

subplot(3,2,5)
plot(1:num_samples,X(1215,:,3));
xlabel('Number of iterations','FontSize', 16);
ylabel('m_{1215}','FontSize', 16);
%title('MALA');
axis tight;


subplot(3,2,6)
plot(1:num_samples,X(1225,:,3));
xlabel('Number of iterations','FontSize', 16);
ylabel('m_{1225}','FontSize', 16);
%title('MALA');
axis tight;

%% SAVE
%save('numex_3_new.mat');
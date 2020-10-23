%% Seismic Wave Tomography
%clear; clc; close all;
rng('default');
%% Configurations

% Load inversion results
load('fwi_setting.mat');

% misfit
funh = @(m) misfitMCMC(m,mk,Dn,Cd,Pc,P,model);
 
%% Initiate SGLD samples
N = 1e5;                        % Number of samples
ns = 3;                         % Number of samplers
x0 = mk;                        % Initial point
tau = 1;                        % Step-length
xi = length(x0);                 % Size of random vector
X = zeros(xi,N,ns);             % Samples
G = zeros(xi,N,ns);             % Gradients of log-normal distribution
 
%% Plotting 2
figure;
subplot(221)
imagesc(model.x,model.z,reshape(m,model.n),[min(m), max(m)]);
colorbar;
axis equal tight;
xlabel('Lateral','FontSize', 16);
ylabel('Depth','FontSize', 16);
 
subplot(222)
im = imagesc(model.x,model.z,ones(model.n)); hold on;
plot(model.xr,model.zr,'kv','LineWidth',2,'markersize',12); hold on;
plot(model.xs,model.zs,'r*','LineWidth',2,'markersize',12); hold off;
im.AlphaData = 0;
axis equal tight;
xlabel('Lateral','FontSize', 16);
ylabel('Depth','FontSize', 16);
grid on;
 
subplot(223)
imagesc(model.x,model.z,reshape(mk,model.n),[min(m), max(m)]);
colorbar;
axis equal tight;
xlabel('Lateral','FontSize', 16);
ylabel('Depth','FontSize', 16);
 
ss = sqrt(sqrt(diag(inv(P))));
 
subplot(224)
imagesc(model.x,model.z,reshape(ss./max(ss),model.n),[0,1]);
colorbar;
axis equal tight;
xlabel('Lateral','FontSize', 16);
ylabel('Depth','FontSize', 16);
 
%% Start sampling
 
% Lipschitz - ULA
%[X(:,:,1),G(:,:,1)] = samplerLipULA(funh,N,xmap,tau);
[X(:,:,1),G(:,:,1)] = samplerPULA(funh,N,x0,tau,P,lmin,lmax);
 
% Lipschitz - MALA
%[X(:,:,2),G(:,:,2),acclip] = samplerLipMALA(funh,N,xmap,tau);
[X(:,:,2),G(:,:,2),acclip] = samplerPLMALA(funh,N,x0,tau,P,lmin,lmax);
 
% MALA
%[X(:,:,3),G(:,:,3),accmala] = samplerMALA(funh,N,xmap,tau);
[X(:,:,3),G(:,:,3),accmala] = samplerPMALA(funh,N,x0,tau,P,lmin,lmax);
 
% Burn-in
burnin = 50001;
X = X(:,burnin:end,:);
G = G(:,burnin:end,:);
 
% Mean and Std
Xmean = reshape(mean(X,2),xi,ns);
Xstd = reshape(std(X,1,2),xi,ns);
 
%% Plotting 2
figure;
subplot(231)
% Lip-ULA
imagesc(model.x,model.z,reshape(Xmean(:,1),model.n),[min(m), max(m)]);
colorbar;
axis equal tight;
%title('Sample mean','FontSize', 16);
xlabel('Lateral','FontSize', 16);
ylabel('Depth','FontSize', 16);
title("Lip-ULA",'FontSize',16);
 
subplot(234)
% Lip-ULA
imagesc(model.x,model.z,reshape(Xstd(:,1)./max(Xstd(:,1)),model.n),[0, 1]);
colorbar;
axis equal tight;
%title('Sample Standard Deviation','FontSize', 16);
xlabel('Lateral','FontSize', 16);
ylabel('Depth','FontSize', 16);
 
subplot(232)
% Lip-MALA
imagesc(model.x,model.z,reshape(Xmean(:,2),model.n),[min(m), max(m)]);
colorbar;
axis equal tight;
xlabel('Lateral','FontSize', 16);
ylabel('Depth','FontSize', 16);
title("Lip-MALA",'FontSize',16);
 
subplot(235)
% Lip-MALA
imagesc(model.x,model.z,reshape(Xstd(:,2)./max(Xstd(:,2)),model.n),[0, 1]);
colorbar;
axis equal tight;
xlabel('Lateral','FontSize', 16);
ylabel('Depth','FontSize', 16);
 
subplot(233)
% MALA
imagesc(model.x,model.z,reshape(Xmean(:,3),model.n),[min(m), max(m)]);
colorbar;
axis equal tight;
xlabel('Lateral','FontSize', 16);
ylabel('Depth','FontSize', 16);
title("MALA",'FontSize',16);
 
subplot(236)
% MALA
imagesc(model.x,model.z,reshape(Xstd(:,3)./max(Xstd(:,3)),model.n),[0, 1]);
colorbar;
axis equal tight;
xlabel('Lateral','FontSize', 16);
ylabel('Depth','FontSize', 16);
 
%% Compute KSD and ACF for all samplers
% Compute ACF
acf1 = zeros(5000,ns);
acf1(:,1) = autocorr(X(1300,:,1),'NumLags',4999);
acf1(:,2) = autocorr(X(1300,:,2),'NumLags',4999);
acf1(:,3) = autocorr(X(1300,:,3),'NumLags',4999);
 
acf2 = zeros(5000,ns);
acf2(:,1) = autocorr(X(1310,:,1),'NumLags',4999);
acf2(:,2) = autocorr(X(1310,:,2),'NumLags',4999);
acf2(:,3) = autocorr(X(1310,:,3),'NumLags',4999);
 
X1 = X(:,1:5:end,:);
G1 = G(:,1:5:end,:);
 
[~,num_samples,~] = size(X1);
nx = num_samples;
dksd1 = zeros(nx,ns);
 
for i = 1:ns
    tic
    dksd1(:,i) = compute_ksd(X(:,:,i)', G(:,:,i)', nx, "sclmed","vanilla");
    toc;
end
 
% Plot KSD and ACF
figure;
subplot(2,2,1)
plot(acf1,'-','LineWidth',1.5); hold on;
axis tight;
legend('Lip-ULA','Lip-MALA','MALA','FontSize', 16);
title('ACF: m_{1300}','FontSize', 16)
xlabel('Lag','FontSize', 16)
ylabel('Sample ACF','FontSize', 16)
 
subplot(2,2,2)
plot(acf2,'-','LineWidth',1.5); hold on;
axis tight;
legend('Lip-ULA','Lip-MALA','MALA','FontSize', 16);
title('ACF: m_{1310}','FontSize', 16)
xlabel('Lag','FontSize', 16)
ylabel('Sample ACF','FontSize', 16)
 
subplot(2,2,[3,4])
loglog(1:nx,dksd1,'-','LineWidth',2); hold on;
loglog(1:nx,31.5./sqrt(1:nx),'k-','LineWidth',2); hold off;
axis tight;
legend('Lip-ULA','Lip-MALA','MALA','Ref: 1/\surd{N}','FontSize', 16);
title('KSD','FontSize', 16)
 
%% Trace plot
figure;
subplot(3,2,1)
plot(1:num_samples,X(1300,:,1));
xlabel('Number of iterations','FontSize', 16);
ylabel('m_{1215}','FontSize', 16);
%title('ULA');
axis tight;
 
subplot(3,2,2)
plot(1:num_samples,X(1310,:,1));
xlabel('Number of iterations','FontSize', 16);
ylabel('m_{1225}','FontSize', 16);
%title('ULA');
axis tight;
 
subplot(3,2,3)
plot(1:num_samples,X(1300,:,2));
xlabel('Number of iterations','FontSize', 16);
ylabel('m_{1215}','FontSize', 16);
%title('Local Lipschitz ULA');
axis tight;
 
subplot(3,2,4)
plot(1:num_samples,X(1310,:,2));
xlabel('Number of iterations','FontSize', 16);
ylabel('m_{1225}','FontSize', 16);
%title('Local Lipschitz ULA');
axis tight;
 
subplot(3,2,5)
plot(1:num_samples,X(1300,:,3));
xlabel('Number of iterations','FontSize', 16);
ylabel('m_{1215}','FontSize', 16);
%title('MALA');
axis tight;
 
 
subplot(3,2,6)
plot(1:num_samples,X(1310,:,3));
xlabel('Number of iterations','FontSize', 16);
ylabel('m_{1225}','FontSize', 16);
%title('MALA');
axis tight;
 
%% SAVE
save('numex_4.mat');

%% Setup Density
rng('default');
load('numexAppC_mala.mat','Xmala','dksdmala')

%% Function handler

funh = @(x) rosen(x);

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
N = 30000;            % Number of samples
ns = 3;               % Number of MCMC methods
x0 = zeros(2,1);              % Initial point
tau = [3.61e-3, 3.61e-2, 3.61e-1];         % Step-size
X = zeros(2,N,length(tau),ns);    % Samples
G = zeros(2,N,length(tau),ns);    % Gradients of log-normal distribution
P = diag(1./sigma);

% Lipschitz - ULA
[X(:,:,:,1),G(:,:,:,1)] = samplerULA(funh,N,x0,tau);

% MALA
[X(:,:,:,2),G(:,:,:,2),accmala] = samplerMALA(funh,N,x0,tau);

% Burnn-in
burnin = 15001;
X = X(:,burnin:end,:,:);
G = G(:,burnin:end,:,:);

% Swap in MALA
X(:,:,2,2) = Xmala;

% Mean and Std
Xmean = reshape(mean(X,2),2,length(tau),ns);
Xstd = reshape(std(X,1,2),2,length(tau),ns);
Xvar = reshape(var(X,1,2),2,length(tau),ns);

% Compute ULA ACF
ac = 1000;
acf1_ula = zeros(ac,length(tau));
acf1_ula(:,1) = autocorr(X(1,:,1,1),'NumLags',ac-1);
acf1_ula(:,2) = autocorr(X(1,:,2,1),'NumLags',ac-1);
acf1_ula(:,3) = autocorr(X(1,:,3,1),'NumLags',ac-1);

acf2_ula = zeros(ac,ns);
acf2_ula(:,1) = autocorr(X(2,:,1,1),'NumLags',ac-1);
acf2_ula(:,2) = autocorr(X(2,:,2,1),'NumLags',ac-1);
acf2_ula(:,3) = autocorr(X(2,:,3,1),'NumLags',ac-1);

% Compute MALA ACF
acf1_mala = zeros(ac,length(tau));
acf1_mala(:,1) = autocorr(X(1,:,1,1),'NumLags',ac-1);
acf1_mala(:,2) = autocorr(X(1,:,2,1),'NumLags',ac-1);
acf1_mala(:,3) = autocorr(X(1,:,3,1),'NumLags',ac-1);

acf2_mala = zeros(ac,ns);
acf2_mala(:,1) = autocorr(X(2,:,1,1),'NumLags',ac-1);
acf2_mala(:,2) = autocorr(X(2,:,2,1),'NumLags',ac-1);
acf2_mala(:,3) = autocorr(X(2,:,3,1),'NumLags',ac-1);

%% Compute vanilla KSD for all samplers

[~,num_samples,~] = size(X);
nx = num_samples;
dksd_ula = zeros(nx,length(tau));
dksd_mala = zeros(nx,length(tau));
C = inv(P);

for i = 1:length(tau)
    tic
    dksd_ula(:,i) = compute_ksd(X(:,:,i,1)', G(:,:,i,1)', nx, "sclmed", "vanilla");
    
    dksd_mala(:,i) = compute_ksd(X(:,:,i,2)', G(:,:,i,2)', nx, "sclmed", "vanilla");
    toc;
end

% Swap in MALA
dksd_mala(:,2) = dksdmala;

%% Plot KSD and ACF
figure;
subplot(2,2,1)
plot(acf1_ula,'-','LineWidth',2); hold on;
plot(1:ac,zeros(ac,1),'k--','LineWidth',2); hold off;
axis tight;
legend('\tau_{1} = 3.61 \times 10^{-3}','\tau_{2} = 3.61 \times 10^{-2}',...
       'FontSize',9);
title('ULA - ACF: m_{1}','FontSize',14)
xlabel('Lag','FontSize',14)
ylabel('Sample ACF','FontSize',14)

subplot(2,2,2)
plot(acf2_ula,'-','LineWidth',2); hold on;
plot(1:ac,zeros(ac,1),'k--','LineWidth',2); hold off;
axis tight;
legend('\tau_{1} = 3.61 \times 10^{-3}','\tau_{2} = 3.61 \times 10^{-2}',...
       'FontSize',9);
title('ULA - ACF: m_{2}','FontSize',14)
xlabel('Lag','FontSize',14)
ylabel('Sample ACF','FontSize',14)

subplot(2,2,3)
plot(acf1_mala,'-','LineWidth',2); hold on;
plot(1:ac,zeros(ac,1),'k--','LineWidth',2); hold off;
axis tight;
legend('\tau_{1} = 3.61 \times 10^{-3}','\tau_{2} = 3.61 \times 10^{-2}',...
       'FontSize',9);
title('MALA - ACF: m_{1}','FontSize',14)
xlabel('Lag','FontSize',14)
ylabel('Sample ACF','FontSize',14)

subplot(2,2,4)
plot(acf2_mala,'-','LineWidth',2); hold on;
plot(1:ac,zeros(ac,1),'k--','LineWidth',2); hold off;
axis tight;
legend('\tau_{1} = 3.61 \times 10^{-3}','\tau_{2} = 3.61 \times 10^{-2}',...
       'FontSize',9);
title('MALA - ACF: m_{2}','FontSize',14)
xlabel('Lag','FontSize',14)
ylabel('Sample ACF','FontSize',14)


%% KSD
figure;
subplot(1,2,1)
loglog(1:nx,dksd_ula,'-','LineWidth',2); hold on;
loglog(1:nx,10./sqrt(1:nx),'k-','LineWidth',2); hold off;
axis square tight;
%legend('\tau_{1} = 3.61 \times 10^{-3}','\tau_{2} = 3.61 \times 10^{-2}',...
%        '\tau_{3} = 3.61 \times 10^{-1}', 'Ref: 1/\surd{N}','FontSize',11);
legend('\tau_{1}','\tau_{2}','\tau_{3}',...
       'Ref: 1/\surd{N}','FontSize',11,'Location','southwest');
legend('boxoff');
title('ULA','FontSize',14)
xlabel('Number of samples, N','FontSize',14)
ylabel('KSD','FontSize',14)

subplot(1,2,2)
loglog(1:nx,dksd_mala,'-','LineWidth',2); hold on;
loglog(1:nx,10./sqrt(1:nx),'k-','LineWidth',2); hold off;
axis square tight;
%legend('\tau_{1} = 3.61 \times 10^{-3}','\tau_{2} = 3.61 \times 10^{-2}',...
%        '\tau_{3} = 3.61 \times 10^{-1}', 'Ref: 1/\surd{N}','FontSize',11);
title('MALA','FontSize',14)
legend('\tau_{1}','\tau_{2}','\tau_{3}',...
       'Ref: 1/\surd{N}','FontSize',11,'Location','southwest');
legend('boxoff');
xlabel('Number of samples, N','FontSize',14)
ylabel('KSD','FontSize',14)

%% Plotting
rg = 1:10:num_samples;
% Density plot
figure;
subplot(2,3,1)
imagesc(x,y,density);hold on;axis square;
plot(X(2,rg,1,1),X(1,rg,1,1),'wo','MarkerFaceColor','white'); hold on;
plot(Xmean(2,1,1),Xmean(1,1,1),'rx','MarkerSize',7,'LineWidth',1.5); hold on;
plot(mu(2),mu(1),'k^','MarkerSize',7,'LineWidth',1.5);hold off;
xlabel('m_{2}');
ylabel('m_{1}');
title('ULA - \tau_{1} = 3.61 \times 10^{-3}');

subplot(2,3,2)
imagesc(x,y,density);hold on;axis square;
plot(X(2,rg,2,1),X(1,rg,2,1),'wo','MarkerFaceColor','white'); hold on;
plot(Xmean(2,2,1),Xmean(1,2,1),'rx','MarkerSize',7,'LineWidth',1.5); hold on;
plot(mu(2),mu(1),'k^','MarkerSize',7,'LineWidth',1.5);hold off;
xlabel('m_{2}');
ylabel('m_{1}');
title('ULA - \tau_{2} = 3.61 \times 10^{-2}');

subplot(2,3,3)
imagesc(x,y,density);hold on;axis square;
plot(X(2,rg,3,1),X(1,rg,3,1),'wo','MarkerFaceColor','white'); hold on;
plot(Xmean(2,3,1),Xmean(1,3,1),'rx','MarkerSize',7,'LineWidth',1.5); hold on;
plot(mu(2),mu(1),'k^','MarkerSize',7,'LineWidth',1.5);hold off;
xlabel('m_{2}');
ylabel('m_{1}');
title('ULA - \tau_{3} = 3.61 \times 10^{-1}');


subplot(2,3,4)
imagesc(x,y,density);hold on;axis square;
plot(X(2,rg,1,2),X(1,rg,1,2),'wo','MarkerFaceColor','white'); hold on;
plot(Xmean(2,1,2),Xmean(1,1,2),'rx','MarkerSize',7,'LineWidth',1.5); hold on;
plot(mu(2),mu(1),'k^','MarkerSize',7,'LineWidth',1.5);hold off;
xlabel('m_{2}');
ylabel('m_{1}');
title('MALA - \tau_{1} = 3.61 \times 10^{-3}');

subplot(2,3,5)
imagesc(x,y,density);hold on;axis square;
plot(X(2,rg,2,2),X(1,rg,2,2),'wo','MarkerFaceColor','white'); hold on;
plot(Xmean(2,2,2),Xmean(1,2,2),'rx','MarkerSize',7,'LineWidth',1.5); hold on;
plot(mu(2),mu(1),'k^','MarkerSize',7,'LineWidth',1.5);hold off;
xlabel('m_{2}');
ylabel('m_{1}');
title('MALA - \tau_{2} = 3.61 \times 10^{-2}');

subplot(2,3,6)
imagesc(x,y,density);hold on;axis square;
plot(X(2,rg,3,2),X(1,rg,3,2),'wo','MarkerFaceColor','white'); hold on;
plot(Xmean(2,3,2),Xmean(1,3,2),'rx','MarkerSize',7,'LineWidth',1.5); hold on;
plot(mu(2),mu(1),'k^','MarkerSize',7,'LineWidth',1.5);hold off;
xlabel('m_{2}');
ylabel('m_{1}');
title('MALA - \tau_{3} = 3.61 \times 10^{-1}');

%Trace plot
%ULA
figure;
suptitle('ULA');
subplot(3,2,1)
plot(1:num_samples,X(1,:,1,1));
xlabel('Number of iterations');
ylabel('m_{1}');
%title('ULA - 2.59 \times 10^{-2}');
axis tight;

subplot(3,2,2)
plot(1:num_samples,X(2,:,1,1));
xlabel('Number of iterations');
ylabel('m_{2}');
%title('ULA');
axis tight;

subplot(3,2,3)
plot(1:num_samples,X(1,:,2,1));
xlabel('Number of iterations');
ylabel('m_{1}');
%title('Local Lipschitz ULA');
axis tight;

subplot(3,2,4)
plot(1:num_samples,X(2,:,2,1));
xlabel('Number of iterations');
ylabel('m_{2}');
%title('Local Lipschitz ULA');
axis tight;

subplot(3,2,5)
plot(1:num_samples,X(1,:,3,1));
xlabel('Number of iterations');
ylabel('m_{1}');
%title('MALA');
axis tight;


subplot(3,2,6)
plot(1:num_samples,X(2,:,3,1));
xlabel('Number of iterations');
ylabel('m_{2}');
%title('MALA');
axis tight;

%Trace plot
% MALA
figure;
suptitle('MALA');
subplot(3,2,1)
plot(1:num_samples,X(1,:,1,2));
xlabel('Number of iterations');
ylabel('m_{1}');
%title('ULA - 2.59 \times 10^{-2}');
axis tight;

subplot(3,2,2)
plot(1:num_samples,X(2,:,1,2));
xlabel('Number of iterations');
ylabel('m_{2}');
%title('ULA');
axis tight;

subplot(3,2,3)
plot(1:num_samples,X(1,:,2,2));
xlabel('Number of iterations');
ylabel('m_{1}');
%title('Local Lipschitz ULA');
axis tight;

subplot(3,2,4)
plot(1:num_samples,X(2,:,2,2));
xlabel('Number of iterations');
ylabel('m_{2}');
%title('Local Lipschitz ULA');
axis tight;

subplot(3,2,5)
plot(1:num_samples,X(1,:,3,2));
xlabel('Number of iterations');
ylabel('m_{1}');
%title('MALA');
axis tight;


subplot(3,2,6)
plot(1:num_samples,X(2,:,3,2));
xlabel('Number of iterations');
ylabel('m_{2}');
%title('MALA');
axis tight;
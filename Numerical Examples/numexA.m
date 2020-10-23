%% Appendix: Numerical examples
rng('default');

% Target Gaussian density N(0,I)
d = 20;
n = 10000;
Sigma = speye(d);
mu = zeros(d,1);

X = zeros(n,d,4);

% Example 1: Samples from target density
X(:,:,1) = mvnrnd(mu,Sigma,n);

% Example 2: Gaussian density N([1,0,...,0],I)
mu1 = mu; mu1(1) = 1;
X(:,:,2) = mvnrnd(mu1,Sigma,n);

% Example 3: Gaussian density N(0,[0.001,0,0,...]0'.*I)
Sigma2 = Sigma; Sigma2(1) = 0.001;
X(:,:,3) = mvnrnd(mu,Sigma2,n);

% Example 4: Far from target
X(:,:,4) = gamrnd(7.5,1,n,d);

%% Compute KSD
dksd = zeros(n,4);
dksdp = zeros(n,4);

% Langevin and Preconditioned Langevin KSD
for i = 1:4
    
    tic
    dksd(:,i) = compute_ksd(X(:,:,i), -X(:,:,i), n, "sclmed","vanilla");
    dksdp(:,i) = compute_ksd(X(:,:,i), -X(:,:,i), n, "sclmed","smpcov");
    toc;
    
end

% Figure Langevin KSD
figure;
subplot(221)
loglog(1:n,dksd(:,1),'-','LineWidth',2); hold on;
loglog(1:n,5.5./sqrt(1:n),'k-','LineWidth',2); hold off;
axis square tight;
legend('Example 1','Ref: 1/\surd{N}','Location','southwest');
legend('boxoff');
ylabel('KSD','FontSize',16);
xlabel('Number of samples, N','FontSize',16);

subplot(222)
loglog(1:n,dksd(:,2),'-','LineWidth',2); hold on;
loglog(1:n,5.5./sqrt(1:n),'k-','LineWidth',2); hold off;
axis square tight;
legend('Example 2','Ref: 1/\surd{N}','Location','southwest');
legend('boxoff');
ylabel('KSD','FontSize',16);
xlabel('Number of samples, N','FontSize',16);

subplot(223)
loglog(1:n,dksd(:,3),'-','LineWidth',2); hold on;
loglog(1:n,5.5./sqrt(1:n),'k-','LineWidth',2); hold off;
axis square tight;
legend('Example 3','Ref: 1/\surd{N}','Location','southwest');
legend('boxoff');
ylabel('KSD','FontSize',16);
xlabel('Number of samples, N','FontSize',16);

subplot(224)
loglog(1:n,dksd(:,4),'-','LineWidth',2); hold on;
loglog(1:n,37./sqrt(1:n),'k-','LineWidth',2); hold off;
axis square tight;
legend('Example 4','Ref: 1/\surd{N}','Location','southwest');
legend('boxoff');
ylabel('KSD','FontSize',16);
xlabel('Number of samples, N','FontSize',16);

% Figure Preconditioned Langevin KSD
figure;
subplot(221)
loglog(1:n,dksdp(:,1),'-','LineWidth',2); hold on;
loglog(1:n,5.5./sqrt(1:n),'k-','LineWidth',2); hold off;
axis square tight;
legend('Example 1','Ref: 1/\surd{N}','Location','southwest');
legend('boxoff');
ylabel('KSD','FontSize',16);
xlabel('Number of samples, N','FontSize',16);

subplot(222)
loglog(1:n,dksdp(:,2),'-','LineWidth',2); hold on;
loglog(1:n,5.5./sqrt(1:n),'k-','LineWidth',2); hold off;
axis square tight;
legend('Example 2','Ref: 1/\surd{N}','Location','southwest');
legend('boxoff');
ylabel('KSD','FontSize',16);
xlabel('Number of samples, N','FontSize',16);

subplot(223)
loglog(1:n,dksdp(:,3),'-','LineWidth',2); hold on;
loglog(1:n,5.5./sqrt(1:n),'k-','LineWidth',2); hold off;
axis square tight;
legend('Example 3','Ref: 1/\surd{N}','Location','southwest');
legend('boxoff');
ylabel('KSD','FontSize',16);
xlabel('Number of samples, N','FontSize',16);

subplot(224)
loglog(1:n,dksdp(:,4),'-','LineWidth',2); hold on;
loglog(1:n,100./sqrt(1:n),'k-','LineWidth',2); hold off;
axis square tight;
legend('Example 4','Ref: 1/\surd{N}','Location','southwest');
legend('boxoff');
ylabel('KSD','FontSize',16);
xlabel('Number of samples, N','FontSize',16);

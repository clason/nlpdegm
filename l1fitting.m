% L1FITTING
% Test script for nonlinear L^1 data fitting using using primal-dual
% extragradient methods. Solves
%   min_u 1/alpha |S(u)-yd|_L1 + 1/2 |u|_L2^2,
% where S(u)=:y satisfies
%   <\nabla y, \nabla v > + <u y,v> = <f,v> for all v in H^1
% using the approach described in the paper
%  "Primal-dual extragradient methods for nonlinear nonsmooth 
%   PDE-constrained optimization"
% by Christian Clason and Tuomo Valkonen, http://arxiv.org/abs/1606.06219.
%
% June 20, 2016              Christian Clason <christian.clason@uni-due.de>

%% parameters
nel   = 1000;       % problem size: number of elements
alpha = 1e-2;       % regularization parameter
gamma = 1e-12;      % Moreau-Yosida regularization parameter
bg    = 1-1e-16;    % acceleration parameter \bar\gamma
maxit = 1000;       % maximal number of iterations

%% Data
fem = setupFEM(nel);         % FEM setup: mesh and matrices
x   = fem.x; xc = fem.xc;    % nodes, cell centers
M   = fem.M;      % mass matrix for piecewise constant right hand side
P   = fem.P;      % projection onto piecewise constants for linear functions
Y   = fem.Y;      % Y(v)*du assembles weighted mass matrix for piecewise constant du and linear v

Mf  = M*(1+0*x);             % right hand side (pre-multiply mass matrix)
ue  = P(2-abs(x));           % true solution: piecewise constant
ye  = fem.A(ue)\(Mf);        % exact data

% noisy data (additive random-valued impulsive noise)
d_per   = 0.3;                % percentage of corrupted data points
d_mag   = 0.1;                % magnitude of corruption
n       = nel+1;              % number of nodes
yd      = ye;                 % noisy data
ind     = find(rand(n,1) < d_per);
yd(ind) = ye(ind) + d_mag*max(abs(ye))*randn(length(ind),1);
yd      = yd + 0.0*max(yd(:))*randn(n,1);
delta   = norm(yd-ye,1)/n;    % compute noise level

% show noisy data
figure(1), plot(x,ye,x,yd),  title('Data'),  legend('exact','noisy');

% pointwise projection for prox-mapping
proj = @(x) (1/alpha*x)./max(1/alpha,abs(x));

% regularized objective functional (pointwise)
f_gamma = @(y) (abs(y) <= gamma/alpha).*(y).^2/(2*gamma) + ...
               (abs(y) >  gamma/alpha).*(abs(y)/alpha-gamma/(2*alpha^2));

%% Primal-dual extragradient method
u = ones(nel,1);        % primal variable
S = @(f) fem.A(u)\f;    % forward operator
y = S(Mf);              % state
p = zeros(nel+1,1);     % dual variable
J = zeros(maxit,1);     % (regularized) functional value
tau0   = 0.99;          % step size for prox_G
sigma0 = 1.0;           % step size for prox_F*

% estimate Lipschitz constant, scale step sizes
Lest = max(1,norm(y)/norm(u));
tau  = tau0/Lest; sigma = sigma0/Lest;

for it = 1:maxit
    uold = u;   yold = y;
    
    % primal update u = prox_{\tau G}(u-\tau K'(u)y)
    u = 1/(1+tau)*(u-tau*P(y.*S(-M*p)));
    
    % (accelerated) extragradient
    om  = 1/sqrt(1+2*bg*tau);
    tau = tau*om;  sigma = sigma/om;
    ub  = u + om*(u - uold);
    
    % dual update p = prox_{\sigma F*}(p-\sigma K(\bar u)
    yb = fem.A(ub)\Mf;
    q  = p + sigma*(yb-yd);
    p  = proj(1/(1+sigma*gamma)*q);
    
    % update forward operator
    A = fem.A(u);                 % differential operator
    try
        R = chol(A); Rt=R';       % precompute Cholesky factors; faster in 2D
        S = @(f) R\(Rt\f);        % (linearized) solution operator
    catch notspd
        S = @(f) A\f;             % fallback if numerically semidefinite
    end
    y = S(Mf);
    
    J(it) = fem.dx*sum(f_gamma(y-yd)) + fem.dx/2*norm(u)^2;
end

%% Plot results
figure(2),plot(xc,u,xc,ue); title('Reconstruction');
legend(['L^1 (\alpha=' num2str(alpha,'%0.2e') ')'],'exact');
figure(3)
hold on
set(gca, 'XScale', 'log', 'YScale', 'log')
loglog(J);xlabel('k');ylabel('J(u_k)');
hold off

% STATECONSTRAINTS
% Test script for optimal control with state constraints using primal-dual
% extragradient methods. Solves
%   min_u 1/(2\alpha)|S(u)-y^d|_L2^2 + 1/2 |u|_L2^2 
%         s.t. S(u) <= c a.e. in \Omega
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
alpha = 1e-12;      % control cost parameter 
gamma = 1e-12;      % Moreau-Yosida regularization parameter
bg    = 1-1e-16;    % acceleration parameter \bar\gamma
maxit = 10000;      % maximal number of iterations

%% Data
fem = setupFEM(nel);         % FEM setup: mesh and matrices
x   = fem.x; xc = fem.xc;    % nodes, cell centers
M   = fem.M;      % mass matrix for piecewise constant right hand side
P   = fem.P;      % projection onto piecewise constants for linear functions
Y   = fem.Y;      % Y(v)*du assembles weighted mass matrix for piecewise constant du and linear v

Mf  = M*(1+0*x);             % right hand side (pre-multiply mass matrix)
ue  = P(2-abs(x));           % true solution: piecewise constant
yd  = fem.A(ue)\(Mf);        % exact data
c   = 0.68;                  % constraint

% show target
figure(1), plot(x,yd,x,c+0*x,'--'),  title('Data'), legend('desired state','constraint');

% regularized objective functional (pointwise)
f_gamma = @(y) (y >= ((alpha+gamma)/alpha*c - gamma/alpha*yd)).*((yd-c).^2/alpha + (y-c).^2/gamma)/2 + ...
               (y <  ((alpha+gamma)/alpha*c - gamma/alpha*yd)).*((y-yd).^2/(alpha+gamma))/2;
 
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
    q  = p + sigma*yb;
    as = q>((1+sigma*gamma)/alpha*(c-yd)+sigma*c);
    p  = as.*(q-sigma*c)/(1+sigma*gamma) + ...
         (1-as).*(q-sigma*yd)/(1+sigma*(alpha+gamma));
    
    % update forward operator
    A = fem.A(u);                 % differential operator
    try
        R = chol(A); Rt=R';       % precompute Cholesky factors
        S = @(f) R\(Rt\f);        % (linearized) solution operator
    catch notspd
        S = @(f) A\f;             % fallback if numerically semidefinite
    end
    y = S(Mf);
    
    J(it) = fem.dx*sum(f_gamma(y)) + fem.dx/2*norm(u)^2;
end

%% Plot results
figure(2),plot(xc,u); title('control');
figure(3),plot(x,yd,x,y,x,c+0*x); legend('y^d','y','c'); title('state');
figure(4)
hold on
set(gca, 'XScale', 'log', 'YScale', 'log')
loglog(J);xlabel('k');ylabel('J_\gamma(u_k)');
hold off

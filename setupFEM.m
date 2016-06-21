function fem = setupFEM(nel)
% FEM = SETUPFEM(NEL)
% setup 1D FEM structure
% Input:  nel -- number of elements
% Output: fem -- structure containing
%          nel -- number of elements
%          x   -- nodes
%          xc  -- cell centers (used for plotting)
%          dx  -- mesh size
%          A   -- function A(u) returning stiffness matrix (-\Delta + u)
%          M   -- mass matrix for right hand side in V_h
%          Y   -- function Y(y) returning weighted mass matrix
%          P   -- projection V_h to U_h: trapezoidal rule
%
% Christian Clason (christian.clason@uni-graz.at)
% Bangti Jin       (btjin@math.tamu.edu)
% February 27, 2011

%% set up grid
n  = nel+1;              % number of grid points
x  = linspace(-1,1,n)';  % spatial grid points
dx = x(2)-x(1);          % spatial grid step size
xc = x(2:end)-dx/2;      % cell centers

%% set up matrices
e = ones(n,1);
% Laplace with homogeneous Neumann conditions
K = spdiags([-e 2*e -e]/dx,-1:1,n,n); K(1,1:2) = [1 -1]/dx; K(end,end-1:end) = [-1 1]/dx;

% mass matrix for piecewise linear right hand side
M = spdiags([e 4*e e]*dx/6,-1:1,n,n); M(1,1) = dx/3; M(end,end) = dx/3;

%% FEM structure: grid, stiffness & mass matrices, projections etc.
fem.nel = nel;
fem.x   = x;                              % grid
fem.xc  = xc;                             % cell centers
fem.dx  = dx;                             % mesh size

fem.A   = @(u) K+assembleU(u,dx);         % differential operator (-\Delta + u)
fem.M   = M;                              % mass matrix for right hand side in V_h
fem.Y   = @(y) assembleY(y,dx);           % weighted mass matrix: Y(y)*du = <y du,v> for du in U_h

fem.P   = @(y) 1/2*(y(1:end-1)+y(2:end)); % projection V_h to U_h: trapezoidal rule

end % function assembleFEM

%% Assemble potential mass matrix <u y,v> for u in U_h
function U = assembleU(u,dx)
nel = length(u); % number of elements

Me = dx/6 *[2 1 1 2]';  % elemental mass matrix <phi_i,phi_j>

ent = 4*nel;
row = zeros(ent,1);
col = zeros(ent,1);
val = zeros(ent,1);
ind=1;
for el=1:nel
    gl      = ind:(ind+3);
    row(gl) = el + [0 1 0 1];
    col(gl) = el + [0 0 1 1];
    val(gl) = u(el)*Me;
    ind     = ind+4;
end
U = sparse(row,col,val);
end % function assembleU

%% Assemble weighted mass matrix: Y(y)*du = <y du,v> for du in U_h
function Y = assembleY(y,dx)
n   = length(y);                                 % number of points
yp  = [y(2:end);0];                              % y_i+1
Y   = spdiags([y+2*yp 2*y+yp]*dx/6,-1:0,n,n-1);  % Simpson quadrature: midpoint by interpolation
end % function assembleY

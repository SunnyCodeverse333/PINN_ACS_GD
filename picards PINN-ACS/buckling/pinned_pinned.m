% Differential equation: d⁴w/dx⁴ + λ² * d²w/dx² = 0, where λ² = P / (EI)
% Assuming rod of length 1
clc;
clear;
close all;
tic;
rng(42);
n = 500;             % Number of neurons
L = 1;               % Length of the rod

% Bandwidth [-1, 1]
w1 = (2 * rand(n, 1) - 1);
b1 = (2 * rand(n, 1) - 1);

% sample some reference points to avoid trivial solution
xd = L *rand;
Xref = cos(w1 * xd + b1)';
Yref = ones(size(Xref,1), 1);

% Generating collocation points for PDE constraint
m2 = 100;
delta = 0.01;
x_colloc = linspace(delta, L - delta, m2);
x_vec = linspace(0 , L , 200);
YPDE = zeros(m2, 1);

d4w_dx4 = (cos(w1*x_colloc + b1).*(w1(:,1).^4))';
d2w_dx2 = -(cos(w1*x_colloc + b1) .*(w1(:,1).^2))' ;

% Boundary conditions for pinned-pinned beam
x_bc = zeros(4, n);
y_bc = zeros(4, 1);

x_bc(1, :) = cos(w1 * 0 + b1)'; % w(0)
x_bc(2, :) = cos(w1 * L + b1)'; %w(L))
x_bc(3, :) = -(cos(w1 * 0 + b1) .* (w1(:, 1).^2))'; % w''(0)
x_bc(4, :) = -(cos(w1 * L + b1) .* (w1(:, 1).^2))'; % w''(L)

A = d4w_dx4;
B = -d2w_dx2;
eigenVal =0.1; % starting Guess

MaxIter =50;
tol =1e-8;
%% picards iteration

X = [A - eigenVal*B; 1e4*x_bc; Xref];         
Y = [zeros(m2,1);1e4*y_bc; Yref]; 

w2_g = pinv(X)*Y;
u_g = (cos(w1*x_vec + b1)')* w2_g ; % intital guess
% 

w2Opt=inf;
for iter = 1:MaxIter  
    res = A - (eigenVal) * B;
    coeff = (1-0)/200; % monte carlo approximation coefficient
    X_p = cos(w1*x_vec + b1 )';
    X_p = coeff.*((u_g') * X_p);
    %  combining all the residuals (pde bc , normalisaiton)
    X = [A - eigenVal*B; 1e4*x_bc; X_p];         
    Y = [zeros(m2,1);1e4*y_bc; 1]; 

    w2 = pinv(X)*Y;

    num = (((A*w2)') * (B*w2))  + (((B*w2)') *(A*w2)) ;
    den = 2* (((B*w2)')*(B*w2)) ;
    newApproxEigen = (num)/(den) ;

    if abs(newApproxEigen - eigenVal) < tol
        break;
    end
    eigenVal =newApproxEigen;
    w2Opt = w2;
end
elapsed_time =toc;

fprintf("crtical buckling load( pinned-pinned) %.6f\n " ,eigenVal);
fprintf("time elapsed %.4f\n" , elapsed_time);
printEigenMode(w2Opt , w1 , b1 , L , eigenVal);

function printEigenMode(w2 , w1 ,b1,L , pcric)
    x_vec = linspace(0,L ,100);
    pred = w2' * cos(w1*x_vec+ b1);
    pred = pred/norm(pred); % making the L2 norm =1
    figure;
    plot(x_vec ,pred , 'b');
    title(['eigen Function at critical Load (pinned-pinned) ',num2str(pcric)]);
    xlabel('x');
    ylabel('displacement y(x)');
end
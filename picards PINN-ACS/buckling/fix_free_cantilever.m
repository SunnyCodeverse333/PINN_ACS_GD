% Differential equation: d⁴w/dx⁴ + λ² * d²w/dx² = 0, where (eigen value) -> λ² = P / (EI)
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
xd = L*rand;

Xref = cos(w1 * xd + b1)';
Yref = ones(size(Xref,1), 1);

% Generating collocation points for PDE constraint
m2 = 100;
delta = 0.01;
x_colloc = linspace(delta, L - delta, m2);
x_vec = linspace(0 , L , 200);
d4w_dx4 = (cos(w1*x_colloc + b1).*(w1(:,1).^4))';
d2w_dx2 = -(cos(w1*x_colloc + b1) .*(w1(:,1).^2))' ;

% === Boundary Conditions for Fixed-free(both rotational and translational free) (cantilever) Beam ===
x_bc = zeros(3,n);
x_bc(1,:) = cos(w1*0 + b1)' ;                           % w(0)
x_bc(2,:)=  (-cos(w1*L + b1).*(w1(:,1).^2))';           % w''(L)  since rotation is free 
x_bc(3,:) = (-sin(w1*0 + b1).*(w1(:,1)))' ;             % w'(0)

  
y_bc = zeros(3,1);
A = d4w_dx4;
B = -d2w_dx2;
%  adding an eigen value depended boundary condition 
% w'''(L) + lambda^2 *  w'(L)= 0 --> (A- eigenval *B ) --> ( eigenVal = lambda^2)
A(end+1 ,:) =(sin(w1 * L + b1).* (w1.^3) )';
B(end+1 ,:) = -(-sin(w1*L+ b1).*w1(:,1))';

MaxIter =1200;
tol =1e-8;
eigenVal = 0.1;
%% picards iteration

X = [A - eigenVal*B; 1e4*x_bc; Xref];         
Y = [zeros(m2+1,1);1e4*y_bc; Yref]; 

w2_g = pinv(X)*Y;
u_g = (cos(w1*x_vec + b1)')* w2_g ; % intital guess


w2Opt = inf;
%% rayleigh iteration (for finding critical load -first eigen value)
for iter = 1:MaxIter  

    res = A - (eigenVal) * B;
    coeff = (1-0)/200; % monte carlo approximation coefficient
    X_p = cos(w1*x_vec + b1 )';
    X_p = coeff.*((u_g') * X_p);
    %  combining all the residuals (pde bc , normalisaiton)
    
    X = [res; 1e4*x_bc; X_p];         
    Y = [zeros(m2+1,1);1e4*y_bc; 1]; 

    w2 = pinv(X)*Y;
    
    num = (((A*w2)') * (B*w2))  + (((B*w2)') *(A*w2)) ;
    den = 2* (((B*w2)')*(B*w2)) ;
    newApproxEigen = ((num)/(den)) ;

    if abs(newApproxEigen - eigenVal) < tol
        break;
    end
    eigenVal = (newApproxEigen);
    w2Opt = w2;
end
elapsed_time =toc;
fprintf("crtical buckling load (fix-free ends) %.6f\n " ,eigenVal);
fprintf("time elapsed %.4f\n" , elapsed_time);
printEigenMode(w2Opt , w1 , b1 , L , eigenVal);

function printEigenMode(w2 , w1 ,b1,L , pcric)
    x_vec = linspace(0,L ,100);
    pred = w2' * cos(w1*x_vec+ b1);
    pred = pred/norm(pred); % making the L2 norm =1
    figure;
    plot(x_vec ,pred , 'b');
    title(['eigen Function at critical Load (fix-free) ',num2str(pcric)]);
    xlabel('x');
    ylabel('displacement y(x)');
end
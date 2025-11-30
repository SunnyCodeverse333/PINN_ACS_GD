clc; clear; close all;
warning('off', 'all'); % this helps to ignore all warnings
rng(39);
tic;

% Random feature setup
n = 500;               
r = pi;   % bandwidth               
w1 = r*(2*rand(n,2)-1);  
b1 = r*(2*rand(n,1)-1);  

% Collocation points generation
m2 = 1500; 
a = 10; b = 5;
%  collocation points 
[x_colloc(:,1), x_colloc(:,2), xd] = generate_Rect_shape_points(m2, a, b, 1); % 8
%  domain for monte carlo integral 
x_vec = zeros(m2,2);

[x_vec(:,1) ,x_vec(:,2),~]=generate_Rect_shape_points(m2, a ,b ,0);

%  amplitute constraint
x_amp = cos(w1 * xd' + b1)';
y_amp = ones(size(x_amp, 1), 1);

% Residual construction
d4w_dx4 =  (cos(w1 * x_colloc' + b1) .* (w1(:,1).^4))'; 
d4w_dy4 =  (cos(w1 * x_colloc' + b1) .* (w1(:,2).^4))'; 
d4w_dx2dy2 = (cos(w1 * x_colloc' + b1) .*(w1(:,1).^2).* (w1(:,2).^2))' ;
w_x     =  cos(w1 * x_colloc' + b1)'; 

YPDE = zeros(m2, 1);

bc1 = [linspace(0, a, 120)',  zeros(120,1)]; %x,0
bc2 = [a * ones(60,1), linspace(0, b, 60)'];  % a,y
bc3 = [linspace(0,a, 120)', b * ones(120,1)]; % x,b
bc4 = [zeros(60,1), linspace(0, b, 60)']; % 0,y
boundary_points = [bc1; bc2; bc3; bc4];
%  simply suported 
%  dirchlet BC deflection is zero 
x_bc1 = cos(w1 * boundary_points' + b1)';
% moment is zero 
w_xx = (-cos(w1 * boundary_points' + b1) .* (w1(:,1).^2))'; % ∂²w/∂y² or ∂²w/∂x²
w_yy = (-cos(w1 * boundary_points' + b1) .* (w1(:,2).^2))';
% Moment Mn ≈ (∂²w/∂x² + ν ∂²w/∂y²) =0
nu = 0.3; % poisson ratio  
x_bc2 = w_xx + nu * w_yy;

x_bc = [x_bc1 ;x_bc2];
y_bc = zeros(size(x_bc,1),1);

%  governing equation (LHs and rhs )
A = (d4w_dy4 +2*d4w_dx2dy2+ d4w_dx4);
B = w_x;


% Parameters
MaxIter = 30;
% generations = 4;% 6
% guesses = 4;                   % population size per generation
generations = 3;% 6
guesses = 6;                   % population size per generation
scaling_fac =20;
init_guess_eigen = scaling_fac * rand(1, guesses);  % Initial guesses array

final_lambdas = [];
final_weights = [];
prevPop = [];

for gen = 1:generations
    fprintf("Generation %d\n", gen);

    [lambda_arr, weight_arr] = getEigenVal_Xc_incorporated(init_guess_eigen, x_vec, prevPop, ...
         A, B, w1, b1, x_bc, y_bc, x_amp, y_amp, MaxIter, m2 , n,a,b);

    for i = 1:length(lambda_arr)
        lambda = lambda_arr(i);
        weight = weight_arr(:, i);

        % Check uniqueness using eigenvalue proximity
        is_unique = true;
        for k = 1:length(final_lambdas)
            if abs(final_lambdas(k) - lambda) < 0.1
                is_unique = false;
                break;
            end
        end

        if is_unique
            final_lambdas(end+1) = lambda;
            final_weights(:, end+1) = weight;
            prevPop = final_weights;
            fprintf("  Added eigenvalue: %.6f\n", lambda);
        else
            fprintf("  Skipped near-duplicate eigenvalue: %.6f\n", lambda);
        end
    end

    init_guess_eigen = scaling_fac * rand(1, guesses);  % Initial guesses array

end

% Sort eigenvalues and weights
[final_lambdas, sortIdx] = sort(final_lambdas);
final_weights = final_weights(:, sortIdx);

fprintf("\nFinal eigenvalues (sorted):\n");
disp(final_lambdas);
fprintf("Elapsed time: %.4f seconds\n", toc);

printEigenModes(final_lambdas, a, b, final_weights, w1, b1);

function [X, Y, xd] = generate_Rect_shape_points(N, l, b, maxi)
    X = zeros(N,1); Y = zeros(N,1);
    for i = 1:N
        X(i) = (l) * rand;
        Y(i) = (b) * rand;
    end

    xd = zeros(maxi, 2);
    for i = 1:maxi
        xd(i,1) = (l ) * rand;
        xd(i,2) = (b ) * rand;
    end
    
    % plotting the domain (sampled points)
    % figure;
    % scatter(X, Y, 10, 'filled');
    % title('Collocation Points');
    % xlabel('x'); ylabel('y'); axis equal;

    % figure;
    % scatter(xd(:,1), xd(:,2), 10, 'filled');
    % title('Anchor Points');
    % xlabel('x'); ylabel('y'); axis equal;

end
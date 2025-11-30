
clc; clear; close all;
rng(39);
tic;
warning('off', 'all'); % this helps to ignore all warnings 
n = 500;
r = 8 * pi;
w1 = r * (2 * rand(n, 2) - 1);
b1 = r * (2 * rand(n, 1) - 1);

% Collocation points in L-shape domain
m2 = 900;
[x_colloc(:,1), x_colloc(:,2), xd] = generate_L_shape_points(m2, 1);
x_vec = zeros(m2 ,2);
[x_vec(:,1) , x_vec(:,2) , ~] = generate_L_shape_points(m2 , 0);

% Feature evaluation and Laplacian components
WdotXplusB = w1 * x_colloc' + b1;       % [n x m2]
phi = cos(WdotXplusB);                  % [n x m2]

d2y_dx2 = (-phi .* (w1(:,1).^2))';      % [m2 x n]
d2y_dy2 = (-phi .* (w1(:,2).^2))';      % [m2 x n]
w_x     = phi';                         % [m2 x n] n

% Final operator matrices
A = -(d2y_dx2 + d2y_dy2);  % Laplacian
B = w_x;

% Boundary points (180 points on L-shape)
bc1 = [linspace(-1, 1, 60)', -1 * ones(60,1)];
bc2 = [1 * ones(60,1), linspace(-1, 1, 60)'];
bc3 = [linspace(1, 0, 30)', 1 * ones(30,1)];
bc4 = [0 * ones(30,1), linspace(1, 0, 30)'];
bc5 = [linspace(0, -1, 30)', 0 * ones(30,1)];
bc6 = [-1 * ones(30,1), linspace(0, -1, 30)'];
boundary_points = [bc1; bc2; bc3; bc4; bc5; bc6];

x_bc = cos(w1 * boundary_points' + b1)';    % [180 x n]
y_bc = zeros(size(x_bc,1), 1);

x_amp = cos(w1 * xd' + b1)';
y_amp = ones(size(x_amp,1), 1);

%  4 4 150 


% Parameters
MaxIter = 50;
generations = 3;
guesses = 4;                   % population size per generation
scaling_fac = 50;
init_guess_eigen = scaling_fac * rand(1, guesses);  % Initial guesses array

final_lambdas = [];
final_weights = [];
prevPop = [];

for gen = 1:generations
    fprintf("Generation %d\n", gen);

    [lambda_arr, weight_arr] = getEigenValWt_Xc_incorpor(init_guess_eigen, x_vec, prevPop, ...
         A, B, w1, b1, x_bc, y_bc, x_amp, y_amp, MaxIter, m2 , n);

    for i = 1:length(lambda_arr)
        lambda = lambda_arr(i);
        weight = weight_arr(:, i);

        % Check uniqueness using eigenvalue proximity
        is_unique = true;
        for k = 1:length(final_lambdas)
            if abs(final_lambdas(k) - lambda) < 0.5
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

printEigenModes(final_lambdas, final_weights, w1, b1)
%% Function: Generate L-shaped domain points
function [X, Y, xd] = generate_L_shape_points(N, maxi)
    
    X = zeros(N, 1);
    Y = zeros(N, 1);
    count = 0;
    while count < N
        x = 2 * rand - 1;
        y = 2 * rand - 1;
        if ~(x < 0 && y > 0) && x ~= 0 && y ~= 0
            count = count + 1;
            X(count) = x;
            Y(count) = y;
        end
    end

    xd = zeros(maxi, 2);
    count = 0;
    while count < maxi
        x = 2 * rand - 1;
        y = 2 * rand - 1;
        if ~(x < 0 && y > 0) && x ~= 0 && y ~= 0
            count = count + 1;
            xd(count, :) = [x, y];
        end
    end
end
warning('off', 'all'); % this helps to ignore all warnings 

clc;
clear;
close all;
rng(40);
tic;

% Parameters
r = 8;
n = 500;
L = 1;

% Random features
w1 = r * pi *(2* rand(n, 1)-1);
b1 = r * pi * (2*rand(n, 1)-1);

% Collocation points
m2 = 200;
delta = 0.01;
x_colloc = linspace(delta, L - delta, m2)';
x_vec = linspace(0 , L , m2)' ;
% Operator and features
d4y_dx4 = (cos(w1 * (x_colloc') + b1) .* (w1 .^ 4))';   % [m2 x n]
y_x     = cos(w1 * (x_colloc') + b1)';                  % [m2 x n]

% Boundary conditions
x_bc0 = [cos(w1 * 0 + b1)'; -sin(w1 * 0 + b1)' .* w1'];
x_bc1 = [-cos(w1 * L + b1)' .* (w1 .^ 2)'; sin(w1 * L + b1)' .* (w1 .^ 3)'];
x_bc = [x_bc0; x_bc1];  % [4 x n]

% Boundary target values
y_bc = zeros(4, 1);  

% Reference amplitude constraint
xd = L *rand; 

x_amp = cos(w1 * xd + b1)';  
y_amp = ones(size(x_amp, 1), 1);                

% Construct base matrices
A = d4y_dx4;    
B = y_x;  

% Initialize
final_lambdas = [];
final_weights = [];
prevPop = [];
guesses = 4;
tol_unique = 0.5;  % tolerance to consider lambdas as duplicates
MaxIter = 30;
generations = 5;
scaling_fac = 30;
init_eigen_guesses = scaling_fac*rand(1 ,guesses);
for gen = 1:generations
    fprintf("Generation %d\n", gen);
          
    % Get a population of eigenvalues and weight vectors
    [lambda_pop, weight_pop] = get_EigenValWt_Xc_incor(init_eigen_guesses, x_vec ,prevPop, ...
        A, B, w1, b1, x_bc, y_bc, x_amp, y_amp, MaxIter, m2 , n);

    % Loop through each candidate in the returned population
    for k = 1:length(lambda_pop)
        lambda = lambda_pop(k);
        weight = weight_pop(:, k);

        % Check for uniqueness
        is_new = true;
        for existing = final_lambdas
            if abs(lambda - existing) < tol_unique
                is_new = false;
                break;
            end
        end

        % If unique, store it
        if is_new
            final_lambdas(end+1) = lambda;
            final_weights(:, end+1) = weight;
            fprintf("Accepted new eigenvalue: %.6f (total = %d)\n", lambda, length(final_lambdas));
        else
            fprintf("Skipped duplicate eigenvalue: %.6f\n", lambda);
        end
    end

    % Update prevPop for orthogonality enforcement
    prevPop = final_weights;
    %  generate new population
    init_eigen_guesses = scaling_fac * rand(1, guesses);
    % init_guess_eigen =  lambda_pop + randn(1, guesses);

end

disp("Final eigenvalues:");
% Sort eigenvalues and corresponding weights
[final_lambdas, sortIdx] = sort(final_lambdas);           % ascending order
final_weights = final_weights(:, sortIdx);                % reorder columns to match

disp(final_lambdas);
fprintf("Elapsed time: %.4f seconds\n", toc);

printEigenModes(final_lambdas, final_weights, w1, b1);

function printEigenModes(final_lambdas, final_weights, w1, b1)
    % Function to plot eigenfunctions for each eigenvalue
    % Inputs:
    % - final_lambdas: vector of eigenvalues
    % - final_weights: corresponding weights [n x k]
    % - w1, b1: random feature parameters

    nPoints = 200;
    xvec = linspace(0, 1, nPoints)';  % Discretize the domain [0,1]

    % Create feature matrix over xvec
    Phi = cos(w1 * xvec' + b1)';  % [nPoints x n]

    figure;
    hold on;
    colors = lines(length(final_lambdas));  % distinct colors

    for i = 1:length(final_lambdas)
        w = final_weights(:, i);  % weight vector for i-th mode
        y = Phi * w;              % eigenfunction y(x)

        % Normalize for visual clarity
        y = y / norm(y);

        plot(xvec, y, 'DisplayName', sprintf('\\beta = %.4f', final_lambdas(i)), ...
            'Color', colors(i, :), 'LineWidth', 1.5);
    end

    xlabel('x');
    ylabel('Eigenfunction y(x)');
    title('Eigenmodes');
    legend show;
    grid on;
    hold off;
end


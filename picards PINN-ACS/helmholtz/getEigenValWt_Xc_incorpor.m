function [eigenVals, weights] = getEigenValWt_Xc_incorpor(eigen_guesses, x_vec, prevPop, ...
    A, B, w1, b1, x_bc, y_bc, x_amp, y_amp, MaxIter, m2 , n)
    tol = 1e-8;        

    numGuesses = length(eigen_guesses);

%% Adding orthogonality 

    numPrev = size(prevPop, 2);  % Number of previous eigenvectors (n x p)
    X_ortho = zeros(numPrev ,n) ;
    Y_ortho = zeros(numPrev , 1);
    if numPrev > 0
        Phi = cos(w1 * (x_vec') + b1)';  % [m x n]
        for k = 1:numPrev
            prevW2 = prevPop(:, k);           % [n x 1] (previous weight vector)
            u_j = (Phi * prevW2 );           % [m x 1] (evaluated u_j at collocation)
            X_ortho(k,:)  = (u_j') *Phi;
        end
    end
   
%% picards iteration for each guess 1 time 
    u_guess = zeros(size(x_vec ,1) , numGuesses); % x_vec is column vector 
    for i  =1: numGuesses
        eigen_guess = eigen_guesses(i);
        x_comb = [A - eigen_guess*B; 1e1* x_bc; x_amp];
        y_comb = [zeros(m2, 1); 1e1*y_bc; y_amp];
    
        % w2opt = pinv(x_comb)*y_comb ;
        w2opt = ((x_comb')*x_comb) \ ((x_comb') *y_comb);
        u_guess(: ,i) = (cos(w1*(x_vec')+ b1)') * w2opt; % intitial guess vector (mx1 )
    end 

    eigenVals = zeros(1, numGuesses);
    weights = zeros(n, numGuesses);
    
    Phi = cos(w1*(x_vec')+ b1)' ; % mX n
    
    for g = 1:numGuesses
        eigenVal = eigen_guesses(g);
        w2final = inf;
        u_g = u_guess(: ,g);
        for iter = 1:MaxIter
       
            res = A - eigenVal * B;
            %  combining all the residuals (pde  , bc , normalisaiton)
            coeff = 3/size(x_vec ,1);
            X_g = coeff.*((u_g' )* Phi) ;

            Xc = [res;  sqrt(1000)*x_bc; 1e2*X_g ; 1e4* X_ortho];
            Yc = [zeros(m2, 1); sqrt(1000)*y_bc; 1e2*1; 1e4*Y_ortho];
            %  optimal solution 
            % w2 = pinv(Xc) *Yc ;    
            w2 = ((Xc')*Xc) \((Xc') *Yc);
            %  updating the guess 
            u_g = (cos(w1*(x_vec') + b1)')* w2 ; 
                

            % Rayleigh update
            num = (A * w2)' * (B * w2) + (B * w2)' * (A * w2);
            den = 2 * (B * w2)' * (B * w2);
            if any(isnan(w2)) || any(isinf(w2))
                fprintf("NaN or Inf detected in weight vector at iter %d, guess %d", iter );
                break;
            end
            
            if abs(den) < 1e-12
                fprintf("Denominator near-zero at iter %d, guess %d, skipping update", iter);
                break;
            end
            newApproxEigen = num / den;

            if abs(newApproxEigen - eigenVal) < tol
                break;
            end

            eigenVal = newApproxEigen;
            w2final = w2;
        end

        % Store result for this guess
        eigenVals(g) = eigenVal;
        weights(:, g) = w2final;
    end
end

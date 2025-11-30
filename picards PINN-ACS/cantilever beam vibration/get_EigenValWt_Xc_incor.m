function [finalEigenVals, weights] = get_EigenValWt_Xc_incor(init_guess_eigen,x_vec , prevPop, ...
    A, B, w1, b1, x_bc, y_bc, x_amp, y_amp, MaxIter, m2 , n)

    tol = 1e-8;            

    numGuesses = length(init_guess_eigen);

    %%  adding the orthogonality 
    numPrev = size(prevPop, 2);  % Number of previous eigenvectors (n x p)

    x_ortho = zeros(numPrev , n);
    y_ortho = zeros(numPrev,1);
    if numPrev > 0
        Phi = cos(w1 * (x_vec') + b1)';  
        for k = 1:numPrev
            prevW2 = prevPop(:, k);         
            u_j = (Phi * prevW2 );          
            x_ortho(k,:) =  (u_j') * (Phi);  %  1Xm (mXn)
        end
    end

    %% picards  iteration 
    u_guesses = zeros(numel((x_vec')) ,numGuesses) ;
    for i = 1: numGuesses
        Xc = [A - (init_guess_eigen(i)^4)*B ; 1e6* x_bc; x_amp];
        Yc = [zeros(m2, 1); 1e6* y_bc;  y_amp];
        w2_g = pinv(Xc) * Yc ;
        u_guesses(:,i) = cos(w1*(x_vec') + b1)' * w2_g ;
    end

    finalEigenVals = [];
    weights = [];

    for guessIdx = 1:numGuesses
        eigenVal = init_guess_eigen(guessIdx);  
        w2final = inf;
        u_g = u_guesses( : ,guessIdx) ; % column vector 
        for iter = 1:MaxIter

            res = A - (eigenVal^4) * B;
            coeff = (1-0)/size(x_vec,1);
            X_p = cos(w1*(x_vec') + b1)'; % mxn
            X_p = coeff.*((u_g') * X_p) ; % mX1 * mX n

            Xc = [res; 1e6* x_bc;1e4* X_p ;1e8* x_ortho];
            Yc = [zeros(m2, 1); 1e6*y_bc; 1e4*1; 1e8*y_ortho];
        
            w2 = pinv(Xc)*Yc ;
            % Rayleigh quotient update
            num = (A * w2)' * (B * w2) + (B * w2)' * (A * w2);
            den = 2 * (B * w2)' * (B * w2);
            newApproxEigen = num / den;

            if abs(newApproxEigen - eigenVal^4) < tol
                break;
            end

            eigenVal = (abs(newApproxEigen))^(1/4);  % real + positive
            w2final = w2;
        end

        % Save this candidate
        finalEigenVals(end+1) = eigenVal;
        weights(:, end+1) = w2final;
    end
end


function printEigenModes(final_lambdas, l, b, final_weights, w1, b1)
    for i = 1:numel(final_lambdas)
        % Grid for rectangular domain [0, l] x [0, b]
        gridSize = 201;
        [xg, yg] = meshgrid(linspace(0, l, gridSize), linspace(0, b, gridSize));
        XY = [xg(:), yg(:)]';  % 2 x M

        w2 = final_weights(:, i);

        phi_valid = cos(w1 * XY + b1);  % n x M
        z_valid = (w2' * phi_valid)';   % M x 1
        z_valid = z_valid / norm(z_valid);  % Normalize

        % Reshape to grid for plotting
        Z = reshape(z_valid, gridSize, gridSize);

        % Plot eigenmode
        figure;
        surf(xg, yg, Z, 'EdgeColor', 'none');
        view(2); axis equal tight;
        shading interp;
        colorbar;
        colormap jet;
        % xlabel('x'); ylabel('y');
        % title(['Predicted Eigenmode over square Domain, eigenVal = ' , num2str(final_lambdas(i))]);
    end
end

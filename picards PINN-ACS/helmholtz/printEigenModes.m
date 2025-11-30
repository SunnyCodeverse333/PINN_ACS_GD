%  funciton displays  the eigenfuncitons 
function  printEigenModes(final_lambdas, final_weights, w1, b1)
    for i = 1: numel(final_lambdas)
        % === Final plotting over domain ===
        gridSize = 201;
        [xg, yg] = meshgrid(linspace(-1, 1, gridSize), linspace(-1, 1, gridSize));
        XY = [xg(:), yg(:)]';  % 2 x M

        % Mask out the top-left quadrant
        mask = ~(XY(1,:) < 0 & XY(2,:) > 0);
        w2 = final_weights(: ,i);
        % Evaluate sin features and reconstruct solution
        phi_valid = cos(w1 * XY(:,mask) + b1);  % n x M
        z_valid = (w2' * phi_valid)';          % M x 1
        z_valid = z_valid / norm(z_valid);     % Normalize for plotting

        % Build 2D matrix for visualization
        Z = nan(gridSize, gridSize);
        Z(mask') = z_valid;

        % Plot eigenmode
        figure;
        surf(xg, yg, Z, 'EdgeColor', 'none');
        view(2); axis equal tight;
        shading interp;
        colorbar;
        colormap jet;
        % xlabel('x'); ylabel('y');
        % title(['Predicted Eigenmode over L-Shaped Domain eigen value= ' ,num2str(final_lambdas(i))]);
    end 
    
end 
% function printEigenModes(final_lambdas, final_weights, w1, b1)
%     saveDir = 'C:\Users\HP\OneDrive\images\OneDrive\Documents\MATLAB\proj\final_organised\picards\Lshape\population\orthogonality incorporated in Xc\plots';

%     % Create directory if it does not exist
%     if ~exist(saveDir, 'dir')
%         mkdir(saveDir);
%     end

%     for i = 1:numel(final_lambdas)
%         % === Final plotting over domain ===
%         gridSize = 201;
%         [xg, yg] = meshgrid(linspace(-1, 1, gridSize), linspace(-1, 1, gridSize));
%         XY = [xg(:), yg(:)]';  % 2 x M

%         % Mask out the top-left quadrant
%         mask = ~(XY(1,:) < 0 & XY(2,:) > 0);
%         w2 = final_weights(:, i);
%         % Evaluate sin features and reconstruct solution
%         phi_valid = cos(w1 * XY(:,mask) + b1);  % n x M
%         z_valid = (w2' * phi_valid)';          % M x 1
%         z_valid = z_valid / norm(z_valid);     % Normalize for plotting

%         % Build 2D matrix for visualization
%         Z = nan(gridSize, gridSize);
%         Z(mask') = z_valid;

%         % Plot eigenmode
%         fig = figure('Visible', 'off');
%         surf(xg, yg, Z, 'EdgeColor', 'none');
%         view(2); axis equal tight;
%         shading interp;
%         colorbar;
%         colormap jet;
%         xlabel('x'); ylabel('y');
%         title(['Predicted Eigenmode over L-Shaped Domain eigenvalue= ' ,num2str(final_lambdas(i))]);

%         % Save the figure
%         filename = fullfile(saveDir, sprintf('Eigenmode_%02d_lambda_%.4f.png', i, final_lambdas(i)));
%         saveas(fig, filename);
%         close(fig);
%     end
% end

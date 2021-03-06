function [v,w,error] = backprop_gaussian(patterns, targets, hidden, alpha, eta, epochs, gridsize, x, y, plot)
    [outsize, ndata] = size(targets);
    [insize, ndata] = size(patterns); 
    w = randn(hidden, insize + 1); 
    v = randn(outsize, hidden + 1); 
    dw = 0; dv = 0;
    patterns(end+1,:) = ones(1, ndata); % Add bias row
    error = zeros(1, epochs);
    
    for epoch = 1:epochs
        % Foward pass
        hin = w * patterns;
        hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,ndata)];
        oin = v * hout;
        out = 2 ./ (1+exp(-oin)) - 1;
        % Backward pass
        delta_o = (out - targets) .* ((1 + out) .* (1 - out)) * 0.5;
        delta_h = (v' * delta_o) .* ((1 + hout) .* (1 - hout)) * 0.5;
        delta_h = delta_h(1:hidden, :);

        % Weight update
     
        dw = (dw .* alpha) - (delta_h * patterns') .* (1-alpha);
        dv = (dv .* alpha) - (delta_o * hout') .* (1-alpha);
        w = w + dw .* eta;
        v = v + dv .* eta;
        error(epoch) = sum(sum(abs((out) - targets)));
        %% Plot change
        if (plot)
            if (mod(epoch,100)== 0) fprintf('Epochs: #%d\n', epoch); end
            zz = reshape(out, gridsize, gridsize);
            mesh(x,y,zz);
            axis([-5 5 -5 5 -0.7 0.7]);
            drawnow;
        end
    end
end


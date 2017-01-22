function W = delta(patterns, targets, epochs, step, plot)
    
    if plot
        axis ([-2, 2, -2, 2], 'square');
    end
    originalpatterns = patterns;
    
    % Preprocess data
    [outsize, ndata] = size(targets);
    patterns(end+1,:) = ones(1, ndata); % Add bias row
    [insize, ndata] = size(patterns); 
    
    W = randn(outsize, insize); %Init W
    
    % Start training
    
    for c = 1:epochs
        dW = step*(targets-W*patterns)*patterns';
        W = W + dW;
        if plot
            plotSeparation(W, originalpatterns, targets);
        end
    end
    
    
    

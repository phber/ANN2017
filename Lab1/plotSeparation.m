function [] = plotSeparation(W, patterns, targets)
    p = W(1, 1:2);
    k = -W(1, size(patterns,1)+1) / (p*p');
    l = sqrt(p*p');
    
    plot( patterns(1, find(targets>0) ), ...
        patterns(2, find(targets>0) ), '*', ...
        patterns(1, find(targets<0) ), ...
        patterns(2, find(targets<0) ), '+', ...
        [p(1), p(1)]*k + [-p(2), p(2)]/l, ...
        [p(2), p(2)]*k + [p(1), -p(1)]/l, '-');
    
    drawnow;
    %pause(0.05);
end



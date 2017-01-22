clear all
clc

 %% Init variables
clear all
eta = 0.001;
epochs = 40;
alpha = 0.9;

%% 3.4 Linear separation
[patterns, targets] = sepdata();
W = delta(patterns, targets, 20, 0.001, true);

%% 4.2 Non Linear separation
[patterns, targets] = nsepdata();
hiddenMax = 60;
errors = ones(1, hiddenMax);
for hidden = 1:hiddenMax
    %fprintf('Number of hidden units: #%d\n', hidden);
    [v,w,error] = backprop(patterns, targets, hidden, alpha, eta, epochs);
    errors(hidden) = error(end);
end
plot(1:hiddenMax, errors)
xlabel('Hidden units')
ylabel('Misclassifications')
legend(sprintf('alpha=%.2f, eta=%.3f, epochs=%d',alpha, eta, epochs));

%% 4.3 Encoder problem
patterns = eye(8) * 2 - 1;
targets = patterns;
errors = [];
for c = 1:10
    [v,w,error] = backprop(patterns, targets, 3, alpha, eta, epochs);
    errors(end+1) = error(end);
end
errors



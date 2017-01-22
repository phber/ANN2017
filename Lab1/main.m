%% Setup
clear all
clc

%% 3.4 Linear separation
clear all
eta = 0.001;
epochs = 20;
[patterns, targets] = sepdata();
W = delta(patterns, targets, epochs, eta, true);
legend(sprintf('eta=%.3f, epochs=%d', eta, epochs));

%% 4.2 Non Linear separation
[patterns, targets] = nsepdata();
hiddenMax = 40;
eta = 0.1;
epochs = 100;
alpha = 0.9;
errors = ones(1, hiddenMax);
for hidden = 1:hiddenMax
    temperror = [];
    for count = 1:10
        %fprintf('Number of hidden units: #%d\n', hidden);
        [v,w,error] = backprop(patterns, targets, hidden, alpha, eta, epochs);
        temperror(end+1) = error(end);
    end
    errors(hidden) = mean(temperror);
end
plot(1:hiddenMax, errors)
xlabel('Hidden units')
ylabel('Misclassifications')
legend(sprintf('alpha=%.2f, eta=%.3f, epochs=%d',alpha, eta, epochs));
%% Error 4 hidden units
[patterns, targets] = nsepdata();
eta = 0.1;
epochs = 100;
alpha = 0.9;
[v,w,error] = backprop(patterns, targets,4 , alpha, eta, epochs);
plot(1:epochs, error)
xlabel('Epochs')
ylabel('Misclassifications')
legend(sprintf('alpha=%.2f, eta=%.3f, epochs=%d,hidden=%d',alpha, eta, epochs,4));
%% 4.3 Encoder problem
eta = 0.1;
epochs = 5000;
alpha = 0.9;
patterns = eye(8) * 2 - 1;
targets = patterns;
errors = [];
[v,w,error] = backprop(patterns, targets, 3, alpha, eta, epochs);
fprintf('Error=%d\n', error(end));
sign(w)
%% Plot encoding error
plot(1:epochs, error)
xlabel('Epochs')
ylabel('Misclassifications')
legend(sprintf('alpha=%.2f, eta=%.3f, epochs=%d',alpha, eta, epochs));

%% 5 Function approximation
eta = 0.01;
epochs = 1000;
alpha = 0.9;
hidden = 4;
[patterns, targets, gridsize, x, y, z] = gaussiandata();
[v,w,error] = backprop_gaussian(patterns, targets, hidden, alpha, eta, epochs, gridsize, x, y);





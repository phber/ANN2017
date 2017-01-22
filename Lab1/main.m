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
epochs = 1000;
alpha = 0.9;
errors = ones(1, hiddenMax);
for hidden = 1:hiddenMax
    fprintf('Number of hidden units: #%d\n', hidden);
    temperror = [];
    for count = 1:10
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
clear all
[patterns, targets] = nsepdata();
eta = 0.1;
epochs = 1000;
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
[v,w,error] = backprop(patterns, targets, 3, alpha, eta, epochs);
fprintf('Error=%d\n', error(end));
sign(w)
%% Plot encoding error
plot(1:epochs, error)
xlabel('Epochs')
ylabel('Misclassifications')
legend(sprintf('alpha=%.2f, eta=%.3f, epochs=%d',alpha, eta, epochs));

%% 5 Function approximation
clear all
eta = 0.15;
epochs = 200;
alpha = 0.9;
[patterns, targets, gridsize, x, y, z] = gaussiandata(0);

%% Hidden units error
hiddenMax = 50;
errors = ones(1, hiddenMax);
for hidden = 1:hiddenMax
    fprintf('Number of hidden units: #%d\n', hidden);
    temperror = zeros(1,10);
    for count = 1:10
        [v,w,error] = backprop_gaussian(patterns, targets, hidden, alpha, eta, epochs, gridsize, x, y, false);
        temperror(count) = error(end);
    end
    errors(hidden) = mean(temperror);
end
plot(1:hiddenMax, errors)
xlabel('Hidden units')
ylabel('L1 error')
legend(sprintf('alpha=%.2f, eta=%.3f, epochs=%d',alpha, eta, epochs));
%% Eta error
etas = 0.0001:0.0001:0.01;
errors = [];

for eta = etas
    fprintf('Eta: #%d\n', eta);
    temperror = zeros(1,10);
    for count = 1:10
        [v,w,error] = backprop_gaussian(patterns, targets, hidden, alpha, eta, epochs, gridsize, x, y, false);
        temperror(count) = error(end);
    end
    errors(end+1) = mean(temperror);
end
%%
plot(etas, errors)
xlabel('Eta')
ylabel('L1 Error')
legend(sprintf('alpha=%.2f,epochs=%d,hidden=%d',alpha, epochs,hidden));
%% Plot change
hidden = 2;
[v,w,error] = backprop_gaussian(patterns, targets, hidden, alpha, eta, epochs, gridsize, x, y, true);
legend(sprintf('alpha=%.2f,epochs=%d,hidden=%d,eta=%.3f',alpha, epochs,hidden,eta));
error(end)
%plot(1:epochs, error)


%% 6 Generalization
clear all
eta = 0.05;
epochs = 500;
alpha = 0.9;
n = 25;
hidden = 30;

% Keep n samples
[patterns, targets, gridsize, x, y, z] = gaussiandata();
[~, ndata] = size(targets);
permute = randperm(ndata);
new_patterns = patterns(:, permute);
new_targets = targets(:, permute);
new_patterns = new_patterns(:, 1:n);
new_targets = new_targets(:, 1:n);
    
[v,w,error] = backprop_gaussian(new_patterns, new_targets, hidden, alpha, eta, epochs, gridsize, x, y, false);

% Test error
[z_row, z_col] = size(z);
hin = w * [patterns; ones(1, ndata)];
hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,ndata)];
oin = v * hout;
out = 2 ./ (1+exp(-oin)) - 1;
zz = reshape(out, z_row, z_col);
mesh(x, y, zz);
axis([-5 5 -5 5 -0.7 0.7]);
drawnow;
legend(sprintf('n=%d,hidden=%d,alpha=%.2f, eta=%.3f, epochs=%d',n,hidden,alpha, eta, epochs));



clear all; 
clc;
% loading datasets
images = loadMNISTImages('train_images');
labels = loadMNISTLabels('train_labels');

% pre-processing lables, same as hard margin
for i = 1:60000
    if labels(i) == 0
        labels(i) = 1;
    else 
        labels(i) = -1;
    end
end

% SVM for first 1000 data points to reduce computation time
N = 1000; % number of data points
x = images(:,1:N);
y = labels(1:N);
C = 10; % constraint for alpha

% solving w and b with cvx
cvx_begin
    variables w(784,1) b(1) zeta(1,N)
    minimize ( 0.5*w'*w + C*ones(1,N)*zeta')
    subject to
        y'.*(w'*x + b*ones(1,N)) - ones(1,N) + zeta >= 0;
        zeta >= 0;
cvx_end
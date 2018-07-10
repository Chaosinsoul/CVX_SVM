clear all;
clc;
% loading datasets
images = loadMNISTImages('train_images');
labels = loadMNISTLabels('train_labels');

% pre-processing lables
% dividing the data into two groups, zero and non zero
% label them as {1, -1}
for i = 1:60000
    if labels(i) == 0
        labels(i) = 1;
    else 
        labels(i) = -1;
    end
end

% SVM for first 1000 data points to reduce computation time
% also works well on the whole dataset
N = 1000;
X = images(:,1:N);
Yi = labels(1:N);

% solving minimization problem with cvx package
% and get the optimal value for w and b
cvx_begin
    variables w(784,1) b
    minimize ( 0.5*w'*w )
    subject to
      Yi.*(X'*w + b*ones(N,1)) -ones(N,1) >= 0;
cvx_end



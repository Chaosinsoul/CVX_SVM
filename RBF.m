clear all;
clc;
% load training set
images = loadMNISTImages('train_images');
labels = loadMNISTLabels('train_labels');

% pre-processing
for i = 1:60000
    if labels(i) == 0
        labels(i) = 1;
    else
        labels(i) = -1;
    end
end

% select the first 1000 points
N = 1000;
X = images(:,1:N);
y = labels(1:N);

% define radial basis function
k = @(x,z) exp( -sum((x*ones(1,size(z,2))-z).^2) / 500)'

for t =1:N
    for s = t:N
        K(t,s) = k(X(:,t),X(:,s));
        K(s,t) = K(t,s);
    end
end

% solving a(alpha)
cvx_begin
    variables a(N,1)
    minimize( 0.5*(a.*y)' * K * (a.*y) - ones(1,N) * a)
    subject to
        ones(1,N) * (a.*y) == 0
        a >= 0
cvx_end
% find the index alpha > 0, corresponding to the support vectors
ind = find(a > 0.00000001);

% define the function w' * x
wphi = @(xstar) ones(1,N)*(a.*y.*k(xstar,X));
b=0;
for i = 1:length(ind)
    b = b + 1/y(ind(i)) - wphi(X(:,ind(i)));
end
b = b / length(ind); % get the averaged b

% define ystar = w' * X + b
ystar = @(xstar) wphi(xstar) + b
    
    






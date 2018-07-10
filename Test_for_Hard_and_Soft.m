% loading test data
test_images = loadMNISTImages('test_images');
test_labels = loadMNISTLabels('test_labels'); 

% pre-processing 
for i = 1:10000
    if test_labels(i) == 0
        test_labels(i) = 1;
    else 
        test_labels(i) = -1;
    end
end

% check accuracy
test_y = zeros(10000,1); % predicted value
for i = 1:10000
    if test_images(:,i)' * w + b >= 1
        test_y(i) = 1;
    elseif test_images(:,i)' * w + b <= -1
        test_y(i) = -1;  
    else
        test_y(i) = 0;
    end
end

% if test_labels and test_y are the same, count += 1
count = 0;
for i = 1:10000
    if test_labels(i)*test_y(i) == 1
        count = count + 1;
    end
end

accuracy = count / 10000
        
 
    
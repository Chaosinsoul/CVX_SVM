% load test data
test_images = loadMNISTImages('test_images');
test_labels = loadMNISTLabels('test_labels');

% pre-processing
for i = 1:10000
    if test_labels == 0
        test_labels(i) = 1;
    else
        test_labels(i) = -1;
    end
end

% predict and classify test data
test_y = zeros(10000,1);
for i = 1:10000
    if ystar(test_images(:,i)) >= 1
        test_y(i) = 1;
    elseif ystar(test_images(:,i)) <= -1
        test_y(i) = -1;
    else
        test_y(i) = 0;
    end
end

% calculate the accuracy
count = 0;
for i = 1:10000
    if test_labels(i) * test_y(i) == 1
        count = count + 1;
    end
end

accuracy = count / 10000
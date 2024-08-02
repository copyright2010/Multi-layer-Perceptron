dataset_train = [2.7810836 2.550537003 0; 
            1.465489372 2.362125076 0;
            3.396561688 4.400293529 0; 
            1.38807019 1.850220317 0;
            3.06407232 3.005305973 0; 
            7.627531214 2.759262235 1;
            5.332441248 2.088626775 1; 
            6.922596716 1.77106367 1;
            8.675418651 -0.242068655 1; 
            7.673756466 3.508563011 1];

dataset_test = [8.673756466 4.508563011 1;
               2.465489372 3.362125076 0;
            4.396561688 5.400293529 0; 
            2.38807019 2.850220317 0;
            4.06407232 4.005305973 0; 
            8.627531214 3.759262235 1;
            6.332441248 3.088626775 1; 
            7.922596716 2.77106367 1;
            9.675418651 -1.242068655 1; 
            3.7810836 3.550537003 0];
        
%input = [0 0; 0 1; 1 0; 1 1];
% input = dataset_train(:,1:2);
% input_test = dataset_test(:,1:2);
% %desired_out = [0;1;1;1];
% desired_out = dataset_train(:,3);

% activate bias and coefficient 
bias = -1;
coeff = 0.7;
% rand('state',sum(100*clock));
% 
% plot(input(1:5,1),input(1:5,2), 'o')
% hold on
% plot(input(6:10,1),input(6:10,2), 'x')
% hold on

[data,T] = iris_dataset;

set = [data(1,100:150);data(3,100:150)];
set2 = [data(1,50:100);data(3,50:100)];
plot(set(1,:),set(2,:),'rx')
hold on
plot(set2(1,:),set2(2,:),'bo')
xlabel('Sepal Length')
ylabel('Petal Length')
legend('Versicolor','Virginica','Location','northwest')

% Non-linearly seperated data

data_train_fail = [data(1,51:80),data(1,121:150);data(3,51:80),data(3,121:150)]';
targets_fail = [T(2,51:80),T(2,121:150)]';

%data_train_fail_2 = data_train_fail;
targets_fail_2 = targets_fail;

data_test_fail = [data(1,81:120);data(3,81:120)]';
test_targets_fail = T(2,81:120)';
%data_test_fail_2 = data_test_fail;

% plot(data_train_fail(1:30,1), data_train_fail(1:30,2), 'rx')
% hold on
% plot(data_train_fail(31:60,1),data_train_fail(31:60,2), 'bo')
% xlabel('Sepal Length')
% ylabel('Petal Length')
% legend('Versicolor','Virginica','Location','northwest')
% 
% plot(data_train_fail_2(1:20,1), data_train_fail_2(1:20,2), 'rx')
% hold on
% plot(data_train_fail_2(21:40,1),data_train_fail_2(21:40,2), 'bo')
% xlabel('Sepal Length')
% ylabel('Petal Length')
% legend('Versicolor','Virginica','Location','northwest')

% Linearly seperated data
% Training data
% select Sepal and Petal length for Setosa and Versicolor dataset
data_train = [data(1,1:30),data(1,71:100);data(3,1:30),data(3,71:100)]';
targets_train = [T(1,1:30),T(1,71:100)]';

% Testing data
data_test = [data(1,31:70);data(3,31:70)]';
test_targets = T(1,31:70)';

plot(data_train(1:30,1),data_train(1:30,2),'rx')
hold on
plot(data_train(31:60,1),data_train(31:60,2),'bo')
xlabel('Sepal Length')
ylabel('Petal Length')
legend('Setosa','Versicolor','Location','northwest')

figure
plot(data_test(1:20,1),data_test(1:20,2),'rx')
hold on
plot(data_test(21:40,1),data_test(21:40,2),'bo')
xlabel('Sepal Length')
ylabel('Petal Length')
legend('Setosa','Versicolor','Location','northwest')

test = [0 0; 0 1 ; 1 0; 1 1];
plot(test,'.')
targets = [0; 1; 1; 0];

% train perceptron function and calculate the weights. 
%[weights,out] = perceptron_train(data_train_fail_2, targets_fail, bias, coeff);
[weights,out] = perceptron_train(test, targets, bias, coeff);

% test accuracy of the perceptron learning
% learning_accuracy = 100 - abs(((sum(abs(out - targets_fail))/...
%     length(out))*100))

% validate perceptron accuracy
out_validate = p_validate(data_test_fail, weights, bias);

% accuracy_perc = 100 - abs(((sum(abs(out_validate - test_targets_fail))/...
%     length(out_validate))*100))

function [weights,out] = perceptron_train(input, desired_out, bias, coeff)
    weights = -1*2.*rand(3,1); 
    iterations = 1000000;
    for i = 1:iterations
         out = zeros(4,1);
         for j = 1:length(input)
              y = bias*weights(1,1)+...
                   input(j,1)*weights(2,1)+input(j,2)*weights(3,1);
              out(j) = 1/(1+exp(-y));
              delta = desired_out(j)-out(j);
              weights(1,1) = weights(1,1)+coeff*bias*delta;
              weights(2,1) = weights(2,1)+coeff*input(j,1)*delta;
              weights(3,1) = weights(3,1)+coeff*input(j,2)*delta;
         end
    end
end 


%% Validate perceptron
function out_round = p_validate(input_test, weights, bias)
    iterations = 100;
    for i = 1:iterations
        out_validate = zeros(20,1);
         for j = 1:length(input_test)
              y = bias*weights(1,1)+...
                   input_test(j,1)*weights(2,1)+input_test(j,2)*weights(3,1);
               out_validate(j) = 1/(1+exp(-y));
         end
        out_round = round(out_validate);
    end
end

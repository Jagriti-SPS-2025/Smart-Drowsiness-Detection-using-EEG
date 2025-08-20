%% ===============================
%   EEG Drowsiness Detection - Model Testing
%   Description: 
%   - Loads trained SVM model
%   - Tests on new dataset (with labels in last column)
%   - Prints predictions (Active/Drowsy) for each row
%   - Computes Accuracy and F1 score
%% ===============================

clc; clear;

%% Step 1: Load trained model
load('SVM_trained_Model.mat', 'svmModel');

%% Step 2: Load new test data (last column = true labels)
data = readmatrix('mixed_newdata_testing.csv');
X_test = data(:, 1:27);   % First columns = features
y_true = data(:, end);    % Last column = true labels

%% Step 3: Predict using trained SVM
y_pred = predict(svmModel, X_test);

disp('Predictions for each row in mixed_newdata_testing.csv:');
for i = 1:length(y_pred)
    if y_pred(i) == 0
        fprintf('Row %d: Drowsy\n', i);
    else
        fprintf('Row %d: Active\n', i);
    end
end

%% Step 4: Accuracy calculation
accuracy = sum(y_pred == y_true) / length(y_true) * 100;
fprintf('Accuracy: %.2f%%\n', accuracy);

%% Step 5: F1 Score calculation
TP = sum((y_pred==1) & (y_true==1));
FP = sum((y_pred==1) & (y_true==0));
FN = sum((y_pred==0) & (y_true==1));

precision = TP / (TP + FP);
recall    = TP / (TP + FN);
F1        = 2 * (precision * recall) / (precision + recall);

fprintf('F1 Score: %.2f\n', F1);

%% ===============================
%  SVM Model Training and Evaluation
%  Description: 
%   - Loads EEG features from CSV
%   - Trains RBF SVM with hyperparameter tuning
%   - Evaluates using 5-fold CV (Accuracy, F1 score, Confusion Matrix, ROC)
%   - Saves trained model & metrics to MAT file
%% ===============================

%% Step 1: Load dataset
data = readmatrix('features.csv');   % Features file
X = data(:, 1:end-1);                % Features
y = data(:, end);                    % Labels (0 = Drowsy, 1 = Active)

%% Step 2: Train SVM with RBF kernel and automatic tuning
svmModel = fitcsvm(X, y, ...
    'KernelFunction', 'rbf', ...
    'Standardize', true, ...
    'ClassNames', unique(y), ...
    'OptimizeHyperparameters', {'BoxConstraint','KernelScale'}, ...
    'HyperparameterOptimizationOptions', struct( ...
        'AcquisitionFunctionName','expected-improvement-plus', ...
        'ShowPlots', false, ...
        'Verbose', 0, ...
        'Kfold', 5));

%% Step 3: Cross-validated accuracy
cvModel = crossval(svmModel, 'KFold', 5);
loss = kfoldLoss(cvModel);  % Classification error
accuracy = (1 - loss) * 100;
fprintf('Cross-validated Accuracy: %.2f%%\n', accuracy);

%% Step 4: Predictions for F1 score
y_pred = kfoldPredict(cvModel);

tp = sum((y_pred == 1) & (y == 1));
fp = sum((y_pred == 1) & (y == 0));
fn = sum((y_pred == 0) & (y == 1));

precision = tp / (tp + fp);
recall = tp / (tp + fn);
f1_score = 2 * (precision * recall) / (precision + recall);

fprintf('F1 Score: %.2f\n', f1_score);

%% Step 5: Confusion Matrix
figure;
cm = confusionchart(y, y_pred);
cm.Title = 'Confusion Matrix - 5-Fold CV SVM';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

%% Step 6: ROC Curve
[~, scores] = kfoldPredict(cvModel);
[Xroc, Yroc, ~, AUC] = perfcurve(y, scores(:,2), 1);

figure;
plot(Xroc, Yroc, 'LineWidth', 2);
xlabel('False Positive Rate'); ylabel('True Positive Rate');
title(sprintf('ROC Curve (AUC = %.2f)', AUC));
grid on;

%% Step 7: Accuracy per Fold
foldAcc = zeros(cvModel.KFold, 1);
for k = 1:cvModel.KFold
    y_fold_pred = predict(cvModel.Trained{k}, X(cvModel.Partition.test(k), :));
    y_fold_true = y(cvModel.Partition.test(k), :);
    foldAcc(k) = sum(y_fold_pred == y_fold_true) / numel(y_fold_true) * 100;
end

figure;
bar(foldAcc, 'FaceColor', [0.2 0.6 0.8]);
xlabel('Fold Number'); ylabel('Accuracy (%)');
title('Accuracy per Fold'); grid on;

%% Step 8: Save trained model and results
save('SVM_trained_Model.mat', 'svmModel', 'cvModel', 'X', 'y', 'accuracy', 'f1_score');

clc
clear 
close all

% IMPORTING THE DATA - Make sure you have correct file
A = readmatrix("data_part_1.csv");

opts = detectImportOptions("data_part_1.csv");

% Data = variable data (y), Wavelenghts = wavelenght data (x)
% Data division
Data = A(:,2:21);
Wavelenghts = A(:,22:end);
%boxplot(Data)

Data = normalize(Data, "center");
% Center and scale data to have standard deviation 1

Wavelenghts = normalize(Wavelenghts, "zscore");
%boxplot(Data)

medians = mean(Wavelenghts,1)
X = A(1,22:end);
plot(X,medians)

% Adding data togeter so the data division is done with corresponding rows
Dataset = cat(2,Data, Wavelenghts);
%Dataset = rmmissing(Dataset); % <- Basically removes all the data so this
%is not a solution in our case

% Dividing the datasets (matrix is rotated so the division is done by
% observations
[train_data, cal_data, test_data] = dividerand(Dataset.', 0.6,0.2,0.2);


% Rotating the matrix back
train_data = train_data.';
cal_data   = cal_data.';
test_data  = test_data.';

numArrays = size(Data,2);

% Creating cell arrays for the each set 
train_sets = cell(numArrays,1);
cal_sets = cell(numArrays,1);
test_sets = cell(numArrays,1);

% Creation of train/cal/test sets for each variable and removing missing
% values by each individual variable

for v = 1:numArrays

    xy_data_train = cat(2,train_data(:,v),train_data(:,21:end));
    xy_data_cal = cat(2,cal_data(:,v),cal_data(:,21:end));
    xy_data_test = cat(2,test_data(:,v),test_data(:,21:end));
    
    % Removing missing values of each trait and corresponding wavelenght
    % data on each set
    xy_data_train = rmmissing(xy_data_train);
    xy_data_cal = rmmissing(xy_data_cal);
    xy_data_test = rmmissing(xy_data_test);

    train_sets{v}   = xy_data_train;
    cal_sets{v}     = xy_data_cal;
    test_sets{v}    = xy_data_test;

    clear xy_data_train
    clear xy_data_cal
    clear xy_data_test
end





%% Creating PLS models 

% Creating cell arrays for different regression outcomes
numArrays = size(Data,2);
models_X = cell(numArrays,1);
models_Y = cell(numArrays,1);
models_XS = cell(numArrays,1);
models_YS = cell(numArrays,1);
models_BETA = cell(numArrays,1);
models_stats = cell(numArrays,1);


for i = 1:size(train_sets,1)

    XY = train_sets{i};

    [XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(XY(:,2:end), XY(:,1),50);
    
    % Values are stored for debugging and evaluation purposes
    models_X{i} = XL;
    models_Y{i} = YL;
    models_XS{i} = XS;
    models_YS{i} = YS;
    models_BETA{i} = BETA;
    models_stats{i} = stats;

    % Clearing the matrix for new set
    clear XY
end


%% Calibrating the PLS model

models_RSQ_cal = cell(numArrays,1);


% Calibration of model
for k = 1:size(cal_sets,1)
    
    XY = cal_sets{k};

    [XL,YL,XS,YS,BETA,PCTVAR,MSE,stats]= plsregress(XY(:,2:end), XY(:,1),50);
    
    % Fitting 
    Ypred_cal = [ones(size(XY(:,2:end) ,1),1) XY(:,2:end)]*BETA;

    TSS = sum((XY(:,1)-mean(XY(:,1))).^2);
    RSS = sum((XY(:,1)-Ypred_cal).^2);

    models_RSQ_cal{k}= 1 - RSS/TSS
    clear XY Ypred_cal
end

%% Evaluating the PLS models

% Cell array for R2 scores
models_RSQ_val = cell(numArrays,1);
pp = 1;
figure
for p = 1:size(test_sets,1)

    XY = test_sets{p};

    Ypred_val = [ones(size(XY(:,2:end) ,1),1) XY(:,2:end)]*models_BETA{p};

    TSSVal = sum((XY(:,1)-mean(XY(:,1))).^2);
    RSSVal = sum((XY(:,1)-Ypred_val).^2);
    models_RSQ_val{p} = 1 - RSSVal/TSSVal
    
    subplot(2,10,pp); pp = pp+1;

    % Ypred and Y to observations plot
    %plot(Ypred_val, ":")
    %hold on
    %plot(XY(:,1),":");
    %title(['Variable ' num2str(p) ' plot'])
    %xlabel('Observations')
    %ylabel('Variable value')

    % Y/Y_pred plot
    axis tight
    scatter(XY(:,1),Ypred_val)
    hold on
    plot(-30:30,-30:30) % Reference y-line (1-dimensional line)
    title(['Variable ' num2str(p) ' plot'])
    xlabel('Y- real')
    ylabel('Y- predicted')


    clear XY
end
%% Variance explained in x by nr of components in percentage
hold off
figure
plot(cumsum(PCTVAR(2,:)))

%% Five worst models

% The worst models change a bit since the division to sets is done randomly
% which affects training etc. 

% Store indeces to models for determining the ranking
for h = 1:length(models_RSQ_val)
    models_RSQ_val{h,2} = h;
end

% Sorting according to worst Q^2 scores
worst_models1 = sortrows(models_RSQ_val,1)

%% PCA on 5 worst models

worst_models(1) = cal_sets(worst_models1{1,2});
worst_models(2) = cal_sets(worst_models1{2,2});
worst_models(3) = cal_sets(worst_models1{3,2});
worst_models(4) = cal_sets(worst_models1{4,2});
worst_models(5) = cal_sets(worst_models1{5,2});

% Cell array for new calibration sets where values over warning are excluded
cal_sets_new = cell(5,1);

figure
ii = 1;
for i = 1:size(worst_models,2)

    [model.P, model.T, model.latent, model.tsq, model.expl] = pca(worst_models{i}, 'Centered', false, 'NumComponents', 50)

    % T2
    % Setting limits
    model.T2            = t2comp(worst_models{i}, model.P, model.latent, i)
    model.T2warning     = mean(model.T2) + 2 * std(model.T2);
    model.T2alarm       = mean(model.T2) + 3 * std(model.T2);

    subplot(2,5,ii); ii = ii+1;
    scatter(1:length(model.T2), model.T2)
    hold on
    plot([1, length(worst_models{i})], [model.T2warning, model.T2warning], '--');
    plot([1, length(worst_models{i})], [model.T2alarm , model.T2alarm ], '--');
    legend(["Scores", "Warning", "Alarm"]);
    title(["Variable " worst_models1{i,2} " T2"])
    xlim([0 1200])
    
    % SPEx
    % Setting limits
    model.SPEx            = qcomp(worst_models{i}, model.P, i);
    model.SPExwarning     = mean(model.SPEx) + 2 * std(model.SPEx);
    model.SPExalarm       = mean(model.SPEx) + 3 * std(model.SPEx);

    subplot(2,5,ii); ii = ii+1;
    scatter(1:length(model.SPEx ), model.SPEx, 'g.');
    hold on
    plot([1, length(worst_models{i})], [model.T2warning, model.T2warning], '--');
    plot([1, length(worst_models{i})], [model.T2alarm , model.T2alarm ], '--');
    legend(["Scores", "Warning", "Alarm"]);
    title(["Variable " worst_models1{i,2} " SPEx"])
    xlim([0 1200])

    % Deleting the exceeding variables
    cal_sets_new{worst_models1{i,2}} = cal_sets{worst_models1{i,2}}
    idx = find(model.T2 > model.T2warning);
    cal_sets_new{worst_models1{i,2}}(idx,:) = [];

    clear idx
end


%% T2 from PLS MODEL
% Assinging the models stats (T2 scores etc. to 5 worst models)
worst_modelst2(1) = models_stats(worst_models1{1,2});
worst_modelst2(2) = models_stats(worst_models1{2,2});
worst_modelst2(3) = models_stats(worst_models1{3,2});
worst_modelst2(4) = models_stats(worst_models1{4,2});
worst_modelst2(5) = models_stats(worst_models1{5,2});

figure
ii = 1;
for i = 1:size(worst_modelst2,2)
    
    % Warnings and alarms
   
    model.T2warning   = mean(worst_modelst2{i}.T2) + 2 * std(worst_modelst2{i}.T2);
    model.T2alarm   = mean(worst_modelst2{i}.T2) + 3 * std(worst_modelst2{i}.T2);

    subplot(1,5,ii); ii = ii+1;
    scatter(1:length(worst_modelst2{i}.T2), worst_modelst2{i}.T2)
    hold on
    plot([1, length(worst_modelst2{i}.T2)], [model.T2warning, model.T2warning], '--');
    plot([1, length(worst_modelst2{i}.T2)], [model.T2alarm , model.T2alarm ], '--');
    legend(["Scores", "Warning", "Alarm"]);
    title(["Variable " worst_models1{i,2} " model"])
end
hold off


%% Re-calibrating the 5 worst PLS model with new cal set

models_RSQ_cal_new = cell(numArrays,1);


% Calibration of model
for k = 1:size(cal_sets_new,1)
    
    % To only use worst models (others are empty)
    if isempty(cal_sets_new{k})
        continue
    end

    XY = cal_sets_new{k};

    [XL,YL,XS,YS,BETA,PCTVAR,MSE,stats]= plsregress(XY(:,2:end), XY(:,1),50);
    
    % Values are re-stored for debugging and evaluation purposes
    models_X{i} = XL;
    models_Y{i} = YL;
    models_XS{i} = XS;
    models_YS{i} = YS;
    models_BETA{i} = BETA;
    models_stats{i} = stats;

    % Fitting 
    Ypred_cal = [ones(size(XY(:,2:end) ,1),1) XY(:,2:end)]*BETA;

    TSS = sum((XY(:,1)-mean(XY(:,1))).^2);
    RSS = sum((XY(:,1)-Ypred_cal).^2);

    models_RSQ_cal_new{k}= 1 - RSS/TSS
    clear XY Ypred_cal
end

%% Re evaluating the 5 worst PLS models

models_RSQ_val_new = cell(numArrays,1);
pp = 1;
figure
for p = 1:size(test_sets,1)

    XY = test_sets{p};

    Ypred_val = [ones(size(XY(:,2:end) ,1),1) XY(:,2:end)]*models_BETA{p};

    TSSVal2 = sum((XY(:,1)-mean(XY(:,1))).^2);
    RSSVal2 = sum((XY(:,1)-Ypred_val).^2);
    models_RSQ_val_new{p} = 1 - RSSVal2/TSSVal2
    
    subplot(2,10,pp); pp = pp+1;

    % Ypred and Y to observations plot
    %plot(Ypred_val, ":")
    %hold on
    %plot(XY(:,1),":");
    %title(['Variable ' num2str(p) ' plot'])
    %xlabel('Observations')
    %ylabel('Variable value')

    % Y/Y_pred plot
    axis tight
    scatter(XY(:,1),Ypred_val)
    hold on
    plot(-30:30,-30:30) % Reference y-line (1-dimensional line)
    title(['Variable ' num2str(p) ' plot'])
    xlabel('Y- real')
    ylabel('Y- predicted')


    clear XY
end





% From course materials
function T2     = t2comp(data, loadings, latent, comp)
score       = data * loadings(:,1:comp);
standscores = bsxfun(@times, score(:,1:comp), 1./sqrt(latent(1:comp,:))');
T2          = sum(standscores.^2,2);
end

function Qfac   = qcomp(data, loadings, comp)
score       = data * loadings(:,1:comp);
reconstructed = score * loadings(:,1:comp)';
residuals   = bsxfun(@minus, data, reconstructed);
Qfac        = sum(residuals.^2,2);
end

%EOF
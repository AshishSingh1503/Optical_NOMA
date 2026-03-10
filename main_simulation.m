clc;
clear;
close all;
rng(42);   % Fix random seed for reproducibility

%% ============================
%  SYSTEM PARAMETERS
%% ============================

N          = 30000;            % Number of transmitted bits per simulation run
P1         = 0.8;              % Power allocation - far user (User 1)
P2         = 0.2;              % Power allocation - near user (User 2)

distance_vec   = 1:2:30;       % Link distance range (meters)
turbidity_vec  = [0.056, 0.15, 0.398];   % Jerlov water types: Pure Sea / Clear Ocean / Coastal
turbidity_name = {'Pure Sea Water', 'Clear Ocean Water', 'Coastal Water'};

noise_var  = 0.02;             % AWGN noise variance

% Gamma-Gamma turbulence parameters
alpha = 3;
beta  = 2;

num_dist = length(distance_vec);
num_turb = length(turbidity_vec);

% BER matrices: rows = distances, columns = water types
BER_SIC = zeros(num_dist, num_turb);
BER_CNN = zeros(num_dist, num_turb);

%% ============================
%  GENERATE RANDOM BITS
%% ============================

bits1 = randi([0 1], N, 1);
bits2 = randi([0 1], N, 1);

%% ============================
%  BPSK MODULATION
%% ============================

x1 = 2*bits1 - 1;
x2 = 2*bits2 - 1;

%% ============================
%  NOMA SUPERPOSITION
%% ============================

X_tx = sqrt(P1)*x1 + sqrt(P2)*x2;

%% ============================
%  DATASET GENERATION FOR CNN
%  (covers all distances & turbidity levels for robust training)
%% ============================

dataset_size = 60000;   % Increased for better generalisation across channel conditions

inputs = zeros(dataset_size, 1);
labels = zeros(dataset_size, 2);

for i = 1:dataset_size

    b1 = randi([0 1]);
    b2 = randi([0 1]);

    s1 = 2*b1 - 1;
    s2 = 2*b2 - 1;

    tx = sqrt(P1)*s1 + sqrt(P2)*s2;

    % Randomly sample a water type and distance during training
    % so the CNN learns to generalise across all channel conditions
    c_rand = turbidity_vec(randi(num_turb));
    d_rand = distance_vec(randi(num_dist));

    % Beer-Lambert attenuation
    h = exp(-c_rand * d_rand);

    % Gamma-Gamma turbulence
    g1 = gamrnd(alpha, 1/alpha);
    g2 = gamrnd(beta,  1/beta);
    turbulence = g1 * g2;

    noise = sqrt(noise_var) * randn;

    y = h * turbulence * tx + noise;

    inputs(i)   = y;
    labels(i,:) = [b1, b2];

end

%% ============================
%  DATASET SPLIT  (70 / 15 / 15)
%% ============================

train_end = floor(0.70 * dataset_size);
val_end   = floor(0.85 * dataset_size);

trainX = inputs(1:train_end);
trainY = labels(1:train_end, :);

valX = inputs(train_end+1 : val_end);
valY = labels(train_end+1 : val_end, :);

testX = inputs(val_end+1 : end);
testY = labels(val_end+1 : end, :);

%% ============================
%  CNN ARCHITECTURE
%  featureInputLayer is correct for scalar (non-sequential) inputs.
%  Output: 2 continuous values in [0,1] via sigmoid-style FC + regressionLayer.
%% ============================

layers = [

    featureInputLayer(1, 'Name', 'input')           % Scalar received sample

    fullyConnectedLayer(64, 'Name', 'fc1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')

    fullyConnectedLayer(128, 'Name', 'fc2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')

    fullyConnectedLayer(64, 'Name', 'fc3')
    reluLayer('Name', 'relu3')

    fullyConnectedLayer(32, 'Name', 'fc4')
    reluLayer('Name', 'relu4')

    fullyConnectedLayer(2, 'Name', 'fc_out')        % 2 outputs: decoded bits for User 1 & User 2

    % Clamp outputs to [0,1] so they can be threshold-detected as bits.
    % tanhLayer scaled, or simply use regressionLayer and clamp predictions.
    regressionLayer('Name', 'output')

];

%% ============================
%  TRAINING OPTIONS
%% ============================

options = trainingOptions('adam', ...
    'MaxEpochs',          25, ...
    'MiniBatchSize',      256, ...
    'InitialLearnRate',   1e-3, ...
    'LearnRateSchedule',  'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 10, ...
    'Shuffle',            'every-epoch', ...
    'ValidationData',     {trainX(1:2000), trainY(1:2000,:)}, ...  % small fixed validation subset
    'ValidationFrequency', 50, ...
    'Plots',              'training-progress', ...
    'Verbose',            false);

% NOTE: ValidationData expects the same format as training data.
% Using a small slice of trainX/trainY here to avoid format mismatch.
% Replace with {valX, valY} if using a version that supports it directly.

%% ============================
%  TRAIN CNN
%% ============================

fprintf('Training CNN... please wait.\n');
net = trainNetwork(trainX, trainY, layers, options);
fprintf('Training complete.\n\n');

%% ============================
%  BER SIMULATION
%  Loop over ALL turbidity levels and distances
%% ============================

for t = 1:num_turb

    c = turbidity_vec(t);

    for k = 1:num_dist

        d = distance_vec(k);

        % Beer-Lambert attenuation
        h = exp(-c * d);

        % Gamma-Gamma turbulence  (vectorised over N samples)
        g1 = gamrnd(alpha, 1/alpha, [N, 1]);
        g2 = gamrnd(beta,  1/beta,  [N, 1]);
        turbulence = g1 .* g2;

        noise = sqrt(noise_var) * randn(N, 1);

        y = h .* turbulence .* X_tx + noise;

        %% --- SIC DECODER ---
        % Step 1: Detect stronger user (User 1 - high power)
        x1_hat_sic = sign(y);

        % Step 2: Cancel User 1 and detect User 2
        y2          = y - sqrt(P1) * x1_hat_sic;
        x2_hat_sic  = sign(y2);

        % Convert BPSK symbols back to bits
        bits1_hat_sic = (x1_hat_sic + 1) / 2;
        bits2_hat_sic = (x2_hat_sic + 1) / 2;

        BER_SIC(k, t) = (sum(bits1 ~= bits1_hat_sic) + sum(bits2 ~= bits2_hat_sic)) / (2*N);

        %% --- CNN DECODER ---
        pred = predict(net, y);         % Output shape: (N x 2), values in [0,1]
        pred_bits = pred > 0.5;         % Hard decision threshold

        bits1_hat_cnn = pred_bits(:, 1);
        bits2_hat_cnn = pred_bits(:, 2);

        BER_CNN(k, t) = (sum(bits1 ~= bits1_hat_cnn) + sum(bits2 ~= bits2_hat_cnn)) / (2*N);

    end

    fprintf('Done turbidity: %s\n', turbidity_name{t});

end

%% ============================
%  PLOT: BER vs DISTANCE
%  (one figure per water type, all on same axes for comparison)
%% ============================

colors_sic = {'r-o', 'm-o', 'k-o'};
colors_cnn = {'b-s', 'c-s', 'g-s'};

figure('Name', 'BER vs Distance', 'NumberTitle', 'off');
hold on;

for t = 1:num_turb
    semilogy(distance_vec, BER_SIC(:,t), colors_sic{t}, 'LineWidth', 2, ...
        'DisplayName', ['SIC - ' turbidity_name{t}]);
    semilogy(distance_vec, BER_CNN(:,t), colors_cnn{t}, 'LineWidth', 2, ...
        'DisplayName', ['CNN - ' turbidity_name{t}]);
end

grid on;
xlabel('Distance (m)', 'FontSize', 12);
ylabel('Bit Error Rate (BER)', 'FontSize', 12);
title('BER vs Link Distance — UWOC NOMA System', 'FontSize', 13);
legend('Location', 'best');
hold off;

%% ============================
%  PLOT: EYE DIAGRAMS
%% ============================

eye_configs = struct( ...
    'c',    {turbidity_vec(1), turbidity_vec(3)}, ...
    'name', {turbidity_name{1}, turbidity_name{3}} ...
);

d_eye = 10;   % Fixed distance for eye diagram

for e = 1:2

    c_eye = eye_configs(e).c;
    h_eye = exp(-c_eye * d_eye);

    g1_e = gamrnd(alpha, 1/alpha, [N, 1]);
    g2_e = gamrnd(beta,  1/beta,  [N, 1]);
    turb_e = g1_e .* g2_e;

    noise_e = sqrt(noise_var) * randn(N, 1);
    y_eye   = h_eye .* turb_e .* X_tx + noise_e;

    figure('Name', ['Eye Diagram - ' eye_configs(e).name], 'NumberTitle', 'off');
    eyediagram(y_eye(1:2000), 20);
    title(['Eye Diagram — ' eye_configs(e).name], 'FontSize', 13);

end

%% ============================
%  OPTIONAL: TEST SET EVALUATION
%% ============================

pred_test  = predict(net, testX);
pred_test_bits = pred_test > 0.5;

BER_test = sum(sum(pred_test_bits ~= testY)) / (2 * size(testY, 1));
fprintf('\nCNN Test Set BER: %.4f\n', BER_test);
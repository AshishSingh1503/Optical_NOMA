clc;
clear;
close all;

%% ============================
% SYSTEM PARAMETERS
%% ============================

N = 30000;                    % Number of transmitted bits
P1 = 0.8;                     % Power allocation (far user)
P2 = 0.2;                     % Power allocation (near user)

distance_vec = 1:2:30;        % Link distance (meters)

% Jerlov water types
turbidity_vec = [0.056 0.15 0.398];

noise_var = 0.02;

% Gamma-Gamma turbulence parameters
alpha = 3;
beta = 2;

BER_SIC = zeros(length(distance_vec),1);
BER_CNN = zeros(length(distance_vec),1);

%% ============================
% GENERATE RANDOM BITS
%% ============================

bits1 = randi([0 1],N,1);
bits2 = randi([0 1],N,1);

%% ============================
% BPSK MODULATION
%% ============================

x1 = 2*bits1 - 1;
x2 = 2*bits2 - 1;

%% ============================
% NOMA SUPERPOSITION
%% ============================

X_tx = sqrt(P1)*x1 + sqrt(P2)*x2;

%% ============================
% DATASET GENERATION FOR CNN
%% ============================

dataset_size = 20000;

inputs = zeros(dataset_size,1);
labels = zeros(dataset_size,2);

c = 0.15;

for i = 1:dataset_size

    b1 = randi([0 1]);
    b2 = randi([0 1]);

    s1 = 2*b1 - 1;
    s2 = 2*b2 - 1;

    tx = sqrt(P1)*s1 + sqrt(P2)*s2;

    % Random distance for diversity
    d_rand = 5 + 20*rand;
    h = exp(-c*d_rand);

    % Gamma-Gamma turbulence
    g1 = gamrnd(alpha,1/alpha);
    g2 = gamrnd(beta,1/beta);

    turbulence = g1*g2;

    noise = sqrt(noise_var)*randn;

    y = h*turbulence*tx + noise;

    inputs(i) = y;

    labels(i,:) = [b1 b2];

end

%% ============================
% DATASET SPLIT
%% ============================

train_ratio = 0.7;
val_ratio = 0.15;

train_end = floor(train_ratio*dataset_size);
val_end = floor((train_ratio+val_ratio)*dataset_size);

trainX = inputs(1:train_end);
trainY = labels(1:train_end,:);

valX = inputs(train_end+1:val_end);
valY = labels(train_end+1:val_end,:);

testX = inputs(val_end+1:end);
testY = labels(val_end+1:end,:);

%% ============================
% CNN ARCHITECTURE
%% ============================

layers = [

sequenceInputLayer(1)

convolution1dLayer(5,16,'Padding','same')
batchNormalizationLayer
reluLayer

convolution1dLayer(3,32,'Padding','same')
batchNormalizationLayer
reluLayer

convolution1dLayer(3,64,'Padding','same')
reluLayer

fullyConnectedLayer(32)
reluLayer

fullyConnectedLayer(2)
sigmoidLayer

regressionLayer

];

%% ============================
% TRAINING OPTIONS
%% ============================

options = trainingOptions('adam', ...
    'MaxEpochs',25,...
    'MiniBatchSize',128,...
    'Shuffle','every-epoch',...
    'ValidationData',{valX,valY},...
    'Plots','training-progress',...
    'Verbose',false);

%% ============================
% TRAIN CNN NETWORK
%% ============================

net = trainNetwork(trainX,trainY,layers,options);

%% ============================
% BER SIMULATION
%% ============================

c = 0.15;

for k = 1:length(distance_vec)

    d = distance_vec(k);

    % Beer-Lambert attenuation
    h = exp(-c*d);

    % Gamma-Gamma turbulence
    g1 = gamrnd(alpha,1/alpha,[N 1]);
    g2 = gamrnd(beta,1/beta,[N 1]);

    turbulence = g1 .* g2;

    noise = sqrt(noise_var)*randn(N,1);

    y = h .* turbulence .* X_tx + noise;

    %% SIC DECODER

    x1_hat = sign(y);
    x1_hat(x1_hat==0) = 1;

    y2 = y - sqrt(P1)*h*mean(turbulence).*x1_hat;

    x2_hat = sign(y2);
    x2_hat(x2_hat==0) = 1;

    bits1_hat = (x1_hat+1)/2;
    bits2_hat = (x2_hat+1)/2;

    BER_SIC(k) = (sum(bits1~=bits1_hat) + sum(bits2~=bits2_hat))/(2*N);

    %% CNN DECODER

    pred = predict(net,y);

    pred_bits = pred > 0.5;

    BER_CNN(k) = sum(sum(pred_bits ~= [bits1 bits2]))/(2*N);

end

%% ============================
% BER VS DISTANCE GRAPH
%% ============================

figure('Position',[100 100 800 600])

semilogy(distance_vec,BER_SIC,'r-o','LineWidth',2,'MarkerSize',8)
hold on
semilogy(distance_vec,BER_CNN,'b-s','LineWidth',2,'MarkerSize',8)

grid on
set(gca,'FontSize',12)

xlabel('Distance (m)','FontSize',14,'FontWeight','bold')
ylabel('Bit Error Rate','FontSize',14,'FontWeight','bold')

legend('Traditional SIC','CNN Decoder','Location','best','FontSize',12)

title('BER vs Link Distance in UWOC NOMA System','FontSize',16,'FontWeight','bold')

%% ============================
% EYE DIAGRAM (PURE SEA)
%% ============================

c = 0.056;
d = 10;

h = exp(-c*d);

g1 = gamrnd(alpha,1/alpha,[N 1]);
g2 = gamrnd(beta,1/beta,[N 1]);

turbulence = g1 .* g2;

noise = sqrt(noise_var)*randn(N,1);

y_eye = h .* turbulence .* X_tx + noise;

figure('Position',[100 100 800 600])
eyediagram(y_eye(1:2000),20)
title('Eye Diagram - Pure Sea Water (c=0.056)','FontSize',14,'FontWeight','bold')

%% ============================
% EYE DIAGRAM (COASTAL WATER)
%% ============================

c = 0.398;

h = exp(-c*d);

g1 = gamrnd(alpha,1/alpha,[N 1]);
g2 = gamrnd(beta,1/beta,[N 1]);

turbulence = g1 .* g2;

noise = sqrt(noise_var)*randn(N,1);

y_eye2 = h .* turbulence .* X_tx + noise;

figure('Position',[100 100 800 600])
eyediagram(y_eye2(1:2000),20)
title('Eye Diagram - Coastal Water (c=0.398)','FontSize',14,'FontWeight','bold')

%% ============================
% DISPLAY RESULTS
%% ============================

fprintf('\n========== SIMULATION RESULTS ==========\n');
fprintf('Minimum BER (SIC): %.4e at %.0f m\n', min(BER_SIC), distance_vec(find(BER_SIC==min(BER_SIC),1)));
fprintf('Minimum BER (CNN): %.4e at %.0f m\n', min(BER_CNN), distance_vec(find(BER_CNN==min(BER_CNN),1)));
fprintf('Average BER (SIC): %.4e\n', mean(BER_SIC));
fprintf('Average BER (CNN): %.4e\n', mean(BER_CNN));
fprintf('Performance Gain: %.2f%%\n', (1-mean(BER_CNN)/mean(BER_SIC))*100);
fprintf('========================================\n\n');

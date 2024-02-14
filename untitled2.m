%%
numFramesPerModType = 4096
percentTrainingSamples = 75;
percentValidationSamples = 12.5;
percentTestSamples = 12.5;

sps = 8;                % Samples per symbol
spf = 1024;             % Samples per frame
symbolsPerFrame = spf / sps;
fs = 200e3;             % Sample rate
fc = [902e6 100e6];     % Center frequencies
SNRj = [-20 -18 -16 -14 -12 -10 -8 -6 -4 -2 0 2 4 6 8 10 12 14 16 18 20];
modulationTypes = categorical(["BPSK", "QPSK", "8PSK", ...
  "16QAM", "64QAM", "PAM4", "GFSK", "CPFSK", ...
  "B-FM", "DSB-AM", "SSB-AM"]);
%%
% Set the random number generator to a known state to be able to regenerate
% the same frames every time the simulation is run
rng(0)

tic

numModulationTypes = length(modulationTypes);

transDelay = 50;
%pool = getPoolSafe();
%if ~isa(pool,"parallel.ClusterPool")
%  dataDirectory = fullfile(tempdir,"ModClassDataFiles");
%else
dataDirectory = uigetdir("","Select network location to save data files");
%end
disp("Data file directory is " + dataDirectory)
fileNameRoot = "frame";

% Check if data files exist
dataFilesExist = false;
if exist(dataDirectory,'dir')
  files = dir(fullfile(dataDirectory,sprintf("%s*",fileNameRoot)));
  if length(files) == numModulationTypes*numFramesPerModType
    dataFilesExist = true;
  end
end

if ~dataFilesExist
  disp("Generating data and saving in data files...")
  [success,msg,msgID] = mkdir(dataDirectory);
  if ~success
    error(msgID,msg)
  end
  for j = 1:length(SNRj)
          channel = helperModClassTestChannel(...
      'SampleRate', fs, ...
      'SNR', SNRj(j), ...
      'PathDelays', [0 1.8 3.4] / fs, ...
      'AveragePathGains', [0 -2 -10], ...
      'KFactor', 4, ...
      'MaximumDopplerShift', 4, ...
      'MaximumClockOffset', 5, ...
      'CenterFrequency', 902e6)

      for modType = 1:numModulationTypes
        elapsedTime = seconds(toc);
        elapsedTime.Format = 'hh:mm:ss';
        fprintf('%s - Generating %s frames\n', ...
          elapsedTime, modulationTypes(modType))
        
        label = modulationTypes(modType);
        numSymbols = (numFramesPerModType / sps);
        dataSrc = helperModClassGetSource(modulationTypes(modType), sps, 2*spf, fs);
        modulator = helperModClassGetModulator(modulationTypes(modType), sps, fs);
        if contains(char(modulationTypes(modType)), {'B-FM','DSB-AM','SSB-AM'})
          % Analog modulation types use a center frequency of 100 MHz
          channel.CenterFrequency = 100e6;
        else
          % Digital modulation types use a center frequency of 902 MHz
          channel.CenterFrequency = 902e6;
        end
        
        for p=1:numFramesPerModType
          % Generate random data
          x = dataSrc();
          
          % Modulate
          y = modulator(x);
          
          % Pass through independent channels
          rxSamples = channel(y);
          
          % Remove transients from the beginning, trim to size, and normalize
          frame = helperModClassFrameGenerator(rxSamples, spf, spf, transDelay, sps);
          SNR = SNRj(j);
          % Save data file
          fileName = fullfile(dataDirectory,...
            sprintf("%s_%s_%i_%03d",fileNameRoot,modulationTypes(modType),SNRj(j),p));
          save(fileName,"frame","label","SNR")
        end
      end
  end
else
  disp("Data files exist. Skip data generation.")
end
%%
allSignals = zeros(length(modulationTypes),1024,4096);
for modType = 1:length(modulationTypes)
  % Initialize an array to store all signalsd
  for i = 1:numFramesPerModType
    % Load data for the first frame
    fileName = fullfile(dataDirectory, ...
      sprintf("frame_%s_%i_%03d.mat", modulationTypes(modType), SNRj(length(SNRj)), i));
    load(fileName, 'frame');
    % Extract signal values and concatenate to the array
    size(frame)
    allSignals(modType,:,i) = frame(:);
  end
  % Plot combined histogram
end

for modType = 1:length(modulationTypes)
  subplot(2,length(modulationTypes),modType);
  histogram(real(allSignals(modType,:,:)), 'Normalization', 'pdf', 'EdgeColor', 'red');
  title(sprintf('Combined PDF of %s Signals over all SNR levels', modulationTypes(modType)));
  xlabel('Signal Values');
  ylabel('Probability Density');
  grid on;
  subplot(2,length(modulationTypes),length(modulationTypes)+modType)
  histogram(imag(allSignals(modType,:,:)), 'Normalization', 'pdf', 'EdgeColor', 'blue');
  title(sprintf('Combined PDF of %s Signals over all SNR levels', modulationTypes(modType)));
  xlabel('Signal Values');
  ylabel('Probability Density');
  grid on;
end
%%
% Plot the amplitude of the real and imaginary parts of the example frames
% against the sample number
helperModClassPlotTimeDomain(dataDirectory,modulationTypes,fs)

%%
frameDS = signalDatastore(dataDirectory,'SignalVariableNames',["frame", "label", "SNR"]);
%%
splitPercentages = [percentTrainingSamples,percentValidationSamples,percentTestSamples];
[trainDSTrans,validDSTrans,testDSTrans] = helperModClassSplitData(frameDS,splitPercentages);
% Read the training and validation frames into the memory
pctExists = parallelComputingLicenseExists();
trainFrames = transform(trainDSTrans, @helperModClassReadFrame);
rxTrainFrames = readall(trainFrames,"UseParallel",pctExists);
rxTrainFrames = permute(cat(4, rxTrainFrames{:}), [2 1 3 4]);
rxTrainFrames = squeeze(rxTrainFrames);
validFrames = transform(validDSTrans, @helperModClassReadFrame);
rxValidFrames = readall(validFrames,"UseParallel",pctExists);
rxValidFrames = permute(cat(4, rxValidFrames{:}), [2 1 3 4]);
rxValidFrames = squeeze(rxValidFrames);

% Read the training and validation labels into the memory
trainLabels = transform(trainDSTrans, @helperModClassReadLabel);
rxTrainLabels = readall(trainLabels,"UseParallel",pctExists);
validLabels = transform(validDSTrans, @helperModClassReadLabel);
rxValidLabels = readall(validLabels,"UseParallel",pctExists);
%%
filter_size = 23;
num_classes = 11;
modClassNet = [
    sequenceInputLayer([1024 2])

    convolution1dLayer(filter_size, 16,'Padding', 11)
    batchNormalizationLayer()
    reluLayer
    maxPooling1dLayer(filter_size,'Stride', 2, 'Padding', 11)

    convolution1dLayer(filter_size, 24,'Padding', 11)
    batchNormalizationLayer()
    reluLayer
    maxPooling1dLayer(filter_size,'Stride', 2, 'Padding', 11)

    convolution1dLayer(filter_size, 32,'Padding', 11)
    batchNormalizationLayer()
    reluLayer
    maxPooling1dLayer(filter_size,'Stride', 2, 'Padding', 11)
    
    convolution1dLayer(filter_size, 48,'Padding', 11)
    batchNormalizationLayer()
    reluLayer
    maxPooling1dLayer(filter_size,'Stride', 2, 'Padding', 11)

    convolution1dLayer(filter_size, 64,'Padding', 11)
    batchNormalizationLayer()
    reluLayer
    maxPooling1dLayer(filter_size,'Stride', 2, 'Padding', 11)

    convolution1dLayer(filter_size, 96,'Padding', 11)
    batchNormalizationLayer()
    reluLayer

    averagePooling1dLayer(32,'Stride', 2,'Padding', 0)
    dropoutLayer(0.5)
    flattenLayer
    fullyConnectedLayer(num_classes)
    softmaxLayer
    classificationLayer
]
%analyzeNetwork(layers)

%%
maxEpochs = 12;
miniBatchSize = 256;
options = helperModClassTrainingOptions(maxEpochs,miniBatchSize,...
  numel(rxTrainLabels),rxValidFrames,rxValidLabels');
%%
rxTrainFrames(:,:,1)
%%
elapsedTime = seconds(toc);
elapsedTime.Format = 'hh:mm:ss';
fprintf('%s - Training the network\n', elapsedTime)
trainedNet = trainNetwork(rxTrainFrames,rxTrainLabels',modClassNet,options);

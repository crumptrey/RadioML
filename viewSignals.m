% Specify the directory where data files are stored
dataDirectory = 'Data_2/';

% List of modulation types and SNR levels
modulationTypes = categorical(["BPSK", "QPSK", "8PSK", ...
  "16QAM", "64QAM", "PAM4", "GFSK", "CPFSK", ...
  "B-FM", "DSB-AM", "SSB-AM"]);
SNRj = [-20 -18 -16 -14 -12 -10 -8 -6 -4 -2 0 2 4 6 8 10 12 14 16 18 20];
numFramesPerModType = 4096
allSignals = zeros(length(modulationTypes),1024,4096);
% Plot combined histogram for each modulation type and SNR
for modType = 1:length(modulationTypes)
  % Initialize an array to store all signalsd
  for i = 1:numFramesPerModType
    % Load data for the first frame
    fileName = fullfile(dataDirectory, ...
      sprintf("frame_%s_%i_%03d.mat", modulationTypes(modType), SNRj(length(SNRj)), i));
    load(fileName, 'frame');
    % Extract signal values and concatenate to the array
    allSignals(modType,:,i) = frame(:);
  end
  % Plot combined histogram
end

for modType = 1:length(modulationTypes)
  subplot(2,length(modulationTypes),modType);
  histogram(real(allSignals(modType,:,:)), 'Normalization', 'pdf', 'EdgeColor', 'black');
  title(sprintf('Combined PDF of %s Signals over all SNR levels', modulationTypes(modType)));
  xlabel('Signal Values');
  ylabel('Probability Density');
  grid on;
  subplot(2,length(modulationTypes),length(modulationTypes)+modType)
  histogram(imag(allSignals(modType,:,:)), 'Normalization', 'pdf', 'EdgeColor', 'black');
  title(sprintf('Combined PDF of %s Signals over all SNR levels', modulationTypes(modType)));
  xlabel('Signal Values');
  ylabel('Probability Density');
  grid on;
end

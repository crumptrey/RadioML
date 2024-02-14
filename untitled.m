net = Snoap_CNN(11);

% Generate sample data (adjust based on your input size)
sample_data = randn(2, 1024);  % Adjust the size accordingly

% Perform a forward pass through the network
output = net.forward(sample_data);

% Display the output
disp(output);
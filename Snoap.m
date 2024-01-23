filter_size = 23;
num_classes = 11;
layers = [
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
analyzeNetwork(layers)

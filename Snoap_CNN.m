classdef Snoap_CNN < matlab.mixin.SetGet
    properties
        ConvLayer1
        BatchNorm1
        ReLU1
        MaxPool1
        
        ConvLayer2
        BatchNorm2
        ReLU2
        MaxPool2
        
        ConvLayer3
        BatchNorm3
        ReLU3
        MaxPool3
        
        ConvLayer4
        BatchNorm4
        ReLU4
        MaxPool4
        
        ConvLayer5
        BatchNorm5
        ReLU5
        MaxPool5
        
        ConvLayer6
        BatchNorm6
        ReLU6
        
        AvgPool
        Dropout
        FullyConnected
        Softmax
    end
    
    methods
        function obj = Snoap_CNN(num_classes)
            filter_size = 23;
            
            obj.ConvLayer1 = convolution1dLayer(filter_size, 16, 'Padding', 11);
            obj.BatchNorm1 = batchNormalizationLayer;
            obj.ReLU1 = reluLayer;
            obj.MaxPool1 = maxPooling1dLayer(filter_size, 'Stride', 2, 'Padding', 11);
            
            obj.ConvLayer2 = convolution1dLayer(filter_size, 24, 'Padding', 11);
            obj.BatchNorm2 = batchNormalizationLayer;
            obj.ReLU2 = reluLayer;
            obj.MaxPool2 = maxPooling1dLayer(filter_size, 'Stride', 2, 'Padding', 11);
            
            obj.ConvLayer3 = convolution1dLayer(filter_size, 32, 'Padding', 11);
            obj.BatchNorm3 = batchNormalizationLayer;
            obj.ReLU3 = reluLayer;
            obj.MaxPool3 = maxPooling1dLayer(filter_size, 'Stride', 2, 'Padding', 11);
            
            obj.ConvLayer4 = convolution1dLayer(filter_size, 48, 'Padding', 11);
            obj.BatchNorm4 = batchNormalizationLayer;
            obj.ReLU4 = reluLayer;
            obj.MaxPool4 = maxPooling1dLayer(filter_size, 'Stride', 2, 'Padding', 11);
            
            obj.ConvLayer5 = convolution1dLayer(filter_size, 64, 'Padding', 11);
            obj.BatchNorm5 = batchNormalizationLayer;
            obj.ReLU5 = reluLayer;
            obj.MaxPool5 = maxPooling1dLayer(filter_size, 'Stride', 2, 'Padding', 11);
            
            obj.ConvLayer6 = convolution1dLayer(filter_size, 96, 'Padding', 11);
            obj.BatchNorm6 = batchNormalizationLayer;
            obj.ReLU6 = reluLayer;
            
            obj.AvgPool = averagePooling1dLayer(32, 'Stride', 2, 'Padding', 0);
            obj.Dropout = dropoutLayer(0.5);
            obj.FullyConnected = fullyConnectedLayer(num_classes);
            obj.Softmax = softmaxLayer;
        end
        
        function out = forward(obj, x)
            out = obj.ConvLayer1(x);
            out = obj.BatchNorm1(out);
            out = obj.ReLU1(out);
            out = obj.MaxPool1(out);
            
            out = obj.ConvLayer2(out);
            out = obj.BatchNorm2(out);
            out = obj.ReLU2(out);
            out = obj.MaxPool2(out);
            
            out = obj.ConvLayer3(out);
            out = obj.BatchNorm3(out);
            out = obj.ReLU3(out);
            out = obj.MaxPool3(out);
            
            out = obj.ConvLayer4(out);
            out = obj.BatchNorm4(out);
            out = obj.ReLU4(out);
            out = obj.MaxPool4(out);
            
            out = obj.ConvLayer5(out);
            out = obj.BatchNorm5(out);
            out = obj.ReLU5(out);
            out = obj.MaxPool5(out);
            
            out = obj.ConvLayer6(out);
            out = obj.BatchNorm6(out);
            out = obj.ReLU6(out);
            
            out = obj.AvgPool(out);
            %out = reshape(out, [], 1);
            out = obj.Dropout(out);
            out = obj.FullyConnected(out);
            out = obj.Softmax(out);
        end
    end
end

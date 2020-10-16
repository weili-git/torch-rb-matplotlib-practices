require 'torch'
require 'torchvision'
require 'matplotlib/pyplot'

# Hyper Parameters
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
plt = Matplotlib::Pyplot
$xx

train_dataset = TorchVision::Datasets::MNIST.new(
    './mnist',          # root: File.join(__dir__, "data")
    train: true,        # 60000
    download: false,    # true for the first time
    transform: TorchVision::Transforms::Compose.new([
        TorchVision::Transforms::ToTensor.new
    ])
)

# plot one example
# plt.imshow(train_dataset[0][0].squeeze.to_a, cmap: 'gray')  # data[0]
# plt.title('%i' % train_dataset[0][1])                       # label[0]
# plt.show

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Torch::Utils::Data::DataLoader.new(train_dataset, batch_size: BATCH_SIZE, shuffle: true)

# pick 2000 samples to speed up testing
test_dataset = TorchVision::Datasets::MNIST.new(
    './mnist',
    train: false,
    download: false,
    transform: TorchVision::Transforms::Compose.new([
        TorchVision::Transforms::ToTensor.new
    ])
)
test_x = test_dataset.data[0..1999].unsqueeze(1).type(:float) / 255.0   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_dataset.targets[0..1999]

class CNN < Torch::NN::Module
    def initialize
        super()
        @conv1 = Torch::NN::Sequential.new(
            Torch::NN::Conv2d.new(1, 16, 5, stride: 1, padding: 2), 
            # in_channels, out_channels, kernel_size, stride, same padding = (k-1)/2
            Torch::NN::ReLU.new,    # => (16, 28, 28) 
            Torch::NN::MaxPool2d.new(2)         # => (16, 14, 14), kernel_size: 2
        )
        @conv2 = Torch::NN::Sequential.new(
            Torch::NN::Conv2d.new(16, 32, 5, stride: 1, padding: 2),    # => (16, 14, 14)
            Torch::NN::ReLU.new,    # => (32, 14, 14)
            Torch::NN::MaxPool2d.new(2)         # => (32, 7, 7), kernel_size: 2
        )
        @out = Torch::NN::Linear.new(32*7*7, 10)
    end

    def forward(x)
        x = @conv1.call(x)
        x = @conv2.call(x)          # (batch, 32, 7, 7)
        x = x.view(x.shape[0], -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = @out.call(x)       # (batch, 10)
        $xx = x
        return output            # return x for visualization

    end
end

cnn = CNN.new
p cnn

optimizer = Torch::Optim::Adam.new(cnn.parameters, lr: LR)
loss_func = Torch::NN::CrossEntropyLoss.new

EPOCH.times do |epoch|
    train_loader.each_with_index do |(x, y), step|
        output = cnn.call(x)
        loss = loss_func.call(output, y)

        optimizer.zero_grad
        loss.backward
        optimizer.step
        
        if step%50==0
            test_output, last_layer = cnn.call(test_x), $xx
            pred_y = Torch.max(test_output, 1)[1]  # max index
            accuracy = pred_y.eq(test_y).type(:int).sum.to_f / test_y.size[0].to_f
            p "Epoch: #{epoch} | train loss: #{loss.to_f} | test accuracy: #{accuracy}"
        end
    end
end

# print 10 predictions from test data
test_output, _ = cnn.call(test_x[0..9]), $xx
pred_y = Torch.max(test_output, 1)[1].to_a
p "#{pred_y}, prediction number"
p "#{test_y[0..9].to_a}, real number"

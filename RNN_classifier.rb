require 'torch'
require 'torchvision'
require 'matplotlib/pyplot'


# Torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
TIME_STEP = 28          # rnn time step / image height
INPUT_SIZE = 28         # rnn input size / image width
LR = 0.01               # learning rate


# Mnist digital dataset
train_data = TorchVision::Datasets::MNIST.new(
    './mnist/',
    train: true,                                        # this is training data
    transform: TorchVision::Transforms::ToTensor.new,   # Converts a PIL.Image or numpy.ndarray to
                                                        # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download: false                                     # download it if you don't have it
)

# Data Loader for easy mini-batch return in training
train_loader = Torch::Utils::Data::DataLoader.new(train_data, batch_size: BATCH_SIZE, shuffle: true)

# convert test data into Variable, pick 2000 samples to speed up testing
test_data = TorchVision::Datasets::MNIST.new(
    './mnist',
    train: false,
    download: false,
    transform: TorchVision::Transforms::ToTensor.new
)
test_x = test_data.data[0..1999].unsqueeze(1).type(:float) / 255.0   # shape (2000, 28, 28) value in range(0,1)
test_y = test_data.targets[0..1999]                                  # covert to numpy array


class RNN < Torch::NN::Module
    def initialize
        super()
        @rnn = Torch::NN::LSTM.new(INPUT_SIZE, 64, num_layers: 1, batch_first: true)
        @out = Torch::NN::Linear.new(64, 10)
    end

    def forward(x)
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = @rnn.call(x)   # None represents zero initial hidden state
        # choose r_out at the last time step
        out = @out.call(r_out[0..(r_out.size[0]-1), -1, 0..(r_out.size[2]-1)])
        return out
    end
end


rnn = RNN.new()
puts rnn

optimizer = Torch::Optim::Adam.new(rnn.parameters, lr: LR)   # optimize all cnn parameters
loss_func = Torch::NN::CrossEntropyLoss.new                  # the target label is not one-hotted

# training and testing
EPOCH.times do |epoch|
    train_loader.each_with_index do |(b_x, b_y), step|  # gives batch data
        b_x = b_x.view(-1, 28, 28)                      # reshape x to (batch, time_step, input_size)
        
        output = rnn.call(b_x)                          # rnn output
        loss = loss_func.call(output, b_y)              # cross entropy loss 
        optimizer.zero_grad                             # clear gradients for this training step
        loss.backward                                   # backpropagation, compute gradients
        optimizer.step                                  # apply gradients
        
        if step % 50 == 0
            test_output = rnn.call(test_x.view(-1, 28, 28))                   # (samples, time_step, input_size)
            pred_y = Torch.max(test_output, 1)[1]
            accuracy = pred_y.eq(test_y).type(:int).sum.to_f / test_y.size[0].to_f
            p "Epoch: #{epoch} | train loss: #{loss.to_f} | test accuracy: #{accuracy}"
        end
    end
end

# print 10 predictions from test data
test_output = rnn.call(test_x[0..9].view(-1, 28, 28))
pred_y = Torch.max(test_output, 1)[1].to_a
p "#{pred_y}, prediction number"
p "#{test_y[0..9].to_a}, real number"

require 'torch'
require 'torchvision'
require 'matplotlib/pyplot'
require 'matplotlib/axes_3d'

plt = Matplotlib::Pyplot


# Hyper Parameters
EPOCH = 10
BATCH_SIZE = 64
LR = 0.005
DOWNLOAD_MNIST = false
N_TEST_IMG = 5

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


class AutoEncoder < Torch::NN::Module
    def initialize
        super()
        @encoder = Torch::NN::Sequential.new(
            Torch::NN::Linear.new(28*28, 128),
            Torch::NN::Tanh.new(),
            Torch::NN::Linear.new(128, 64),
            Torch::NN::Tanh.new(),
            Torch::NN::Linear.new(64, 12),
            Torch::NN::Tanh.new(),
            Torch::NN::Linear.new(12, 3)   # compress to 3 features which can be visualized in plt
        )
        @decoder = Torch::NN::Sequential.new(
            Torch::NN::Linear.new(3, 12),
            Torch::NN::Tanh.new(),
            Torch::NN::Linear.new(12, 64),
            Torch::NN::Tanh.new(),
            Torch::NN::Linear.new(64, 128),
            Torch::NN::Tanh.new(),
            Torch::NN::Linear.new(128, 28*28),
            Torch::NN::Sigmoid.new()       # compress to a range (0, 1)
        )
    end

    def forward(x)
        encoded = @encoder.call(x)
        decoded = @decoder.call(encoded)
        return encoded, decoded
    end
end


autoencoder = AutoEncoder.new()

optimizer = Torch::Optim::Adam.new(autoencoder.parameters, lr: LR)
loss_func = Torch::NN::MSELoss.new

# initialize figure
f, a = plt.subplots(2, N_TEST_IMG, figsize: [5, 2])
plt.ion   # continuously plot

# original data (first row) for viewing
view_data = train_data.data[0..(N_TEST_IMG-1)].view(-1, 28*28).type(:float) / 255.0

N_TEST_IMG.times do |i|
    a[0][i].imshow( view_data[i].view([28, 28]).to_a, cmap='gray')
    a[0][i].set_xticks([]); a[0][i].set_yticks([])
end

EPOCH.times do |epoch|
    train_loader.each_with_index do |(x, b_label), step|
        b_x = x.view(-1, 28*28)   # batch x, shape (batch, 28*28)
        b_y = x.view(-1, 28*28)   # batch y, shape (batch, 28*28)

        encoded, decoded = autoencoder.call(b_x)

        loss = loss_func.call(decoded, b_y) # mean square error
        optimizer.zero_grad                 # clear gradients for this training step
        loss.backward                       # backpropagation, compute gradients
        optimizer.step                      # apply gradients

        if step % 100 == 0
            p "Epoch: #{epoch}| train loss: #{loss.data.to_a}"

            # plotting decoded image (second row)
            _, decoded_data = autoencoder.call(view_data)
            N_TEST_IMG.times do |i|
                a[1][i].clear
                a[1][i].imshow( decoded_data[i].view(28, 28).to_a, cmap='gray')
                a[1][i].set_xticks([]); a[1][i].set_yticks([])
            end
            plt.draw
            plt.pause(0.05)
        end
    end
end
plt.ioff
plt.show

# visualize in 3D plot
view_data = train_data.data[0..199].view(-1, 28*28).type(:float) / 255.0
encoded_data, _ = autoencoder.call(view_data)
fig = plt.figure(2); ax = Matplotlib::Axes3D.new(fig)
X, Y, Z = encoded_data[0..encoded_data.shape[0], 0].to_a, encoded_data[0..encoded_data.shape[0], 1].to_a, encoded_data[0..encoded_data.shape[0], 2].to_a
values = train_data.targets[0..199].to_a
X.zip(Y, Z, values).each do |x, y, z, s|
    c = plt.cm.rainbow((255*s/9).to_i); ax.text(x, y, z, s, backgroundcolor: c)
end
ax.set_xlim(X.min, X.max); ax.set_ylim(Y.min, Y.max); ax.set_zlim(Z.min, Z.max)
plt.show

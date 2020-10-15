require 'torch'
require 'matplotlib/pyplot'

x = Torch.unsqueeze(Torch.linspace(-1, 1, 100), 1)
y = x.pow(2)*5 + Torch.rand(x.size)*0.2
plt = Matplotlib::Pyplot

def save(x, y, plt)
    net1 = Torch::NN::Sequential.new(
        Torch::NN::Linear.new(1, 10),
        Torch::NN::ReLU.new,  # Torch::NN::F.relu(x)
        Torch::NN::Linear.new(10, 1)
    )
    optimizer = Torch::Optim::SGD.new(net1.parameters, lr: 0.2)
    loss_func = Torch::NN::MSELoss.new

    100.times do
        prediction = net1.call(x)
        loss = loss_func.call(prediction, y)  # .call

        optimizer.zero_grad
        loss.backward
        optimizer.step
    end
    # Torch.save(net1, "net.pth")                 # entire net / unavailable
    Torch.save(net1.state_dict, "net_params.pth") # parameters
    # plot result
    prediction = net1.call(x)
    plt.figure(1, figsize: [10, 3])
    plt.subplot(131)
    plt.title("net1")
    plt.scatter(x.to_a, y.to_a)
    plt.plot(x.to_a, prediction.to_a, 'r-', lw=5)
end


# def restore_net(x, y, plt)
#     net2 = Torch.load("net.pth")
#     # plot result
#     prediction = net2.call(x)
#     plt.subplot(132)
#     plt.title("net2")
#     plt.scatter(x.to_a, y.to_a)
#     plt.plot(x.to_a, prediction.to_a, 'r-', lw=5)
# end


def restore_params(x, y, plt)
    net3 = Torch::NN::Sequential.new(
        Torch::NN::Linear.new(1, 10),
        Torch::NN::ReLU.new,
        Torch::NN::Linear.new(10, 1)
    )
    net3.load_state_dict(Torch.load("net_params.pth"))
    # plot result
    prediction = net3.call(x)
    plt.subplot(133)
    plt.title("net3")
    plt.scatter(x.to_a, y.to_a)
    plt.plot(x.to_a, prediction.to_a, 'r-', lw=5)
    plt.show
end

save(x, y, plt)
# restore_net(x, y, net)
restore_params(x, y, plt)

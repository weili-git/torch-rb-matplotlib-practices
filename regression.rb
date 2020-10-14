require 'torch'
require 'matplotlib/pyplot'

x = Torch.unsqueeze(Torch.linspace(-1, 1, 100), 1)  # dim: 1
y = x.pow(2) + Torch.rand(x.size)*0.2  # 0.2*Torch.rand()  dont work

class MyNet < Torch::NN::Module
    def initialize(n_features, n_hidden, n_output)
        super()  # by default, super=super(a, b, c)
        @hidden = Torch::NN::Linear.new(n_features, n_hidden)
        @predict = Torch::NN::Linear.new(n_hidden, n_output)
    end

    def forward(x)
        x = @hidden.call(x)
        x = Torch::NN::F.relu(x)
        x = @predict.call(x)  # 
        return x
    end
end

net = MyNet.new(1, 10, 1)
print(net)

plt = Matplotlib::Pyplot
plt.ion  # start real time drawing
plt.show

opimizer = Torch::Optim::SGD.new(net.parameters, lr: 0.3)
loss_func = Torch::NN::MSELoss.new

epochs = 200

1.upto(epochs) do |epoch|
    prediction = net.call(x)

    loss = loss_func.call(prediction, y)

    opimizer.zero_grad  # clear the last gradient.
    loss.backward
    opimizer.step

    if epoch%5==0
        plt.cla
        plt.scatter(x.to_a, y.to_a)
        plt.plot(x.to_a, prediction.to_a, 'r-', lw: 5)
        plt.text(0.5, 0, "Loss:%.4f" % loss.to_a[0], fontdict: {'size': 20, 'color': 'red'})
        plt.pause(0.1)
    end
end

plt.ioff
plt.show

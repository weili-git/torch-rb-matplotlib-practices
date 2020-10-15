require 'torch'
require 'matplotlib/pyplot'


Torch.manual_seed(1)    # reproducible

n_data = Torch.ones(100, 2)     # 100 (1, 1)
x0 = Torch.normal(n_data*2, 1)  # x0 = n_data*2 + Torch.randn(n_data.size)
y0 = Torch.zeros(100)           # label 0
x1 = Torch.normal(n_data*(-2), 1)   # x1 = n_data*(-2) + Torch.randn(n_data.size)  
y1 = Torch.ones(100)            # label 1
x = Torch.cat([x0, x1], 0).type(:float32)   # shape (200, 2) FloatTensor = 32-bit floating
y = Torch.cat([y0, y1], ).type(:int64)      # shape (200,) LongTensor = 64-bit integer
# argument [x0, x1] instead of (x0, x1)
# type: Torch.FloatTensor => :float32, Torch.LongTensor => :int64


class MyNet < Torch::NN::Module
    def initialize(n_feature, n_hidden, n_output)
        super()
        @hidden = Torch::NN::Linear.new(n_feature, n_hidden)
        @out = Torch::NN::Linear.new(n_hidden, n_output)
    end

    def forward(x)
        x = @hidden.call(x)
        x = Torch::NN::F.relu(x)  # activation func
        x = @out.call(x)
        return x
    end
end

net = MyNet.new(2, 10, 2)  # output => softmax [0.1, 0.9] for example
p net

optimizer = Torch::Optim::SGD.new(net.parameters, lr: 0.01)
loss_func = Torch::NN::CrossEntropyLoss.new

plt = Matplotlib::Pyplot
plt.ion  # start real time drawing

epochs = 100

1.upto(epochs) do |epoch|
    out = net.call(x)
    loss = loss_func.call(out, y)   # must be (1. nn output, 2. target), the target label is NOT one-hotted

    optimizer.zero_grad # clear the last grads
    loss.backward       # complete the new grads
    optimizer.step      # update the weights

    if epochs%2==0
        plt.cla
        prediction = Torch.max(out, 1)[1]  # dim: 1 row, return tensor(max-value, max-index)
        pred_y = prediction  # 0 or 1
        target_y = y
        plt.scatter(x[0..(x.shape[0]-1), 0].to_a, x[0..(x.shape[0]-1), 1].to_a, c: pred_y.to_a, s: 100, lw: 0, cmap: 'RdYlGn')
        accuracy = pred_y.eq(target_y).type(:int).sum.to_f / target_y.size[0].to_f  # astype => type
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict: {'size': 20, 'color':  'red'})
        plt.pause(0.1)
    end
end

plt.ioff
plt.show

# supervised learning

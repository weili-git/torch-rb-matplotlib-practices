  require 'torch'
require 'torchvision'
require 'matplotlib/pyplot'

plt = Matplotlib::Pyplot

# Hyper Parameters
TIME_STEP = 10      # rnn time step
INPUT_SIZE = 1      # rnn input size
LR = 0.02           # learning rate
PI = 3.1415926


# show data
# steps = Torch.linspace(0, PI*2, 100).type(:float32)
# x = Torch.sin(steps)
# y = Torch.cos(steps)
# plt.plot(steps.to_a, y.to_a, 'r-', 'target(cos)')
# plt.plot(steps.to_a, x.to_a, 'b-', 'input(sin)')
# plt.legend(loc: 'best')
# plt.show

class RNN < Torch::NN::Module
    def initialize
        super()
        @rnn = Torch::NN::RNN.new(
            INPUT_SIZE,
            32, # hidden_size
            num_layers: 1,
            batch_first: true
        )
        @out = Torch::NN::Linear.new(32, 1)

    end

    def forward(x, h_state)
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = @rnn.call(x, hx: h_state)  # hx: is necessary!!!

        outs = []   # save all predictions
        r_out.size[1].times do |time_step|
            outs.append(@out.call( r_out[0..(r_out.size[0]-1), time_step, 0..(r_out.size[2]-1)] ))
        end
        
        return Torch.stack(outs, 1), h_state

        # instead, for simplicity, you can replace above codes by follows
        # r_out = r_out.view(-1, 32)
        # outs = self.out(r_out)
        # outs = outs.view(-1, TIME_STEP, 1)
        # return outs, h_state
        
        # or even simpler, since nn.Linear can accept inputs of any dimension 
        # and returns outputs with same dimension except for the last
        # outs = self.out(r_out)
        # return outs
    end
end

rnn = RNN.new
p rnn

optimizer = Torch::Optim::Adam.new(rnn.parameters, lr: LR)
loss_func = Torch::NN::MSELoss.new

h_state = nil   # for initial hidden state

plt.figure(1, figsize: [12, 5])
plt.ion

100.times do |step|
    start_, end_ = step*PI, (step+1)*PI # time range
    # use sin predicts cos
    steps = Torch.linspace(start_, end_, TIME_STEP, dtype: :float32)  # end_point: false
    x = Torch.sin(steps)
    y = Torch.cos(steps)

    x = x.view(1, -1, 1)     # shape (batch, time_step, input_size)
    y = y.view(1, -1, 1)

    prediction, h_state = rnn.call(x, h_state)
    h_state = h_state.data      # repack the hidden state, break the connection from last iteration

    loss = loss_func.call(prediction, y)
    optimizer.zero_grad
    loss.backward
    optimizer.step

    # plotting
    plt.plot(steps.to_a, y.view(-1).to_a, 'r-')
    plt.plot(steps.to_a, prediction.view(-1).to_a, 'b-')
    plt.draw
    plt.pause(0.05)
end

plt.ioff
plt.show




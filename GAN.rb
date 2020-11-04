require 'torch'
require 'matplotlib/pyplot'

plt = Matplotlib::Pyplot

# Hyper Parameters
BATCH_SIZE = 64
LR_G = 0.0001           # learning rate for generator
LR_D = 0.0001           # learning rate for discriminator
N_IDEAS = 5             # think of this as number of ideas for generating an art work (Generator)
ART_COMPONENTS = 15     # it could be total point G can draw in the canvas
PAINT_POINTS = Torch.stack( BATCH_SIZE.times.map {|i| Torch.linspace(-1, 1, ART_COMPONENTS)} )
# PAINT_POINTS = Torch.stack([Torch.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])

# show our beautiful painting range
# plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
# plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
# plt.legend(loc='upper right')
# plt.show()


def artist_works()     # painting from the famous artist (real target)
    a = (Torch.rand(BATCH_SIZE)+1).view(-1, 1)  # 1-2 uniform distribution
    paintings = a * (PAINT_POINTS ** 2) + (a-1)
    paintings = Torch.tensor(paintings).type(:float)
    return paintings
end

G = Torch::NN::Sequential.new(                      # Generator
    Torch::NN::Linear.new(N_IDEAS, 128),            # random ideas (could from normal distribution)
    Torch::NN::ReLU.new(),
    Torch::NN::Linear.new(128, ART_COMPONENTS),     # making a painting from these random ideas
)

D = Torch::NN::Sequential.new(                      # Discriminator
    Torch::NN::Linear.new(ART_COMPONENTS, 128),     # receive art work either from the famous artist or a newbie like G
    Torch::NN::ReLU.new(),
    Torch::NN::Linear.new(128, 1),
    Torch::NN::Sigmoid.new(),                       # tell the probability that the art work is made by artist
)

opt_D = Torch::Optim::Adam.new(D.parameters, lr: LR_D)
opt_G = Torch::Optim::Adam.new(G.parameters, lr: LR_G)

plt.ion

10000.times do |step|
    artist_paintings = artist_works                 # real painting from artist
    G_ideas = Torch.randn(BATCH_SIZE, N_IDEAS, requires_grad: true)  # random ideas
    G_paintings = G.call(G_ideas)                   # fake painting from G (random ideas)
    prob_artist1 = D.call(G_paintings)              # D try to reduce this prob
    G_loss = Torch.mean(Torch.log(- prob_artist1 + 1.0 ))  
    opt_G.zero_grad
    G_loss.backward
    opt_G.step
     
    prob_artist0 = D.call(artist_paintings)         # D try to increase this prob
    prob_artist1 = D.call(G_paintings.detach)       # D try to reduce this prob
    D_loss = - Torch.mean(Torch.log(prob_artist0) + Torch.log(- prob_artist1 + 1.0))
    opt_D.zero_grad
    D_loss.backward(retain_graph: true)      # reusing computational graph
    opt_D.step

    if step % 50 == 0   # plotting
        plt.cla
        plt.plot(PAINT_POINTS[0].to_a, G_paintings.to_a[0], c: '#4AD631', lw: 3, label: 'Generated painting',)
        plt.plot(PAINT_POINTS[0].to_a, (PAINT_POINTS[0]**2 * 2 + 1).to_a, c: '#74BCFF', lw: 3, label: 'upper bound')
        plt.plot(PAINT_POINTS[0].to_a, (PAINT_POINTS[0]**2 * 1 + 0).to_a, c: '#FF9359', lw: 3, label: 'lower bound')
        plt.text(-0.5, 2.3, "D accuracy=#{prob_artist0.mean.to_a[0]} (0.5 for D to converge)", fontdict: {'size': 13})
        plt.text(-0.5, 2, "D score= #{(-D_loss).to_a[0]} (-1.38 for G to converge)", fontdict: {'size': 13})
        plt.ylim([0, 3]);plt.legend(loc='upper right', fontsize: 10);plt.draw();plt.pause(0.01)
    end
end

plt.ioff
plt.show

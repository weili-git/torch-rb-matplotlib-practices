require 'torch'

BATCH_SIZE = 8

x = Torch.linspace(1, 10, 10)
y = Torch.linspace(10, 1, 10)


torch_dataset = Torch::Utils::Data::TensorDataset.new(x, y)
loader = Torch::Utils::Data::DataLoader.new(
    torch_dataset, batch_size: BATCH_SIZE, shuffle: false   # , num_workers: 2
)

3.times do |epoch|
    loader.each_with_index do |(batch_x, batch_y), step|
        # traning...
        puts "Epoch: #{epoch}| Step: #{step}| batch_x: #{batch_x.to_a}| batch_y: #{batch_y.to_a}"
    end
end

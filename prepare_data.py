import torchvision
import torch

mnist_trainset = torchvision.datasets.MNIST(
  root='./data',
  train=True, download=False, transform=torchvision.transforms.ToTensor()
)

dataset_len = len(mnist_trainset)

train_dataset_0, train_dataset_1, validate_dataset = torch.utils.data.random_split(mnist_trainset, [dataset_len//3, dataset_len//3, dataset_len - dataset_len//3*2])

torch.save(train_dataset_0, "./data/MNIST/splitted/train_dataset_0")
torch.save(train_dataset_1, "./data/MNIST/splitted/train_dataset_1")
torch.save(validate_dataset, "./data/MNIST/splitted/validate_dataset")

# # dataset = torch.load("./data/MNIST/splitted/train_dataset_1")

# # torch.utils.data.DataLoader(
# #   dataset=dataset,
# #   batch_size=100,
# #   shuffle=False
# # )

# mnist_testset = torchvision.datasets.MNIST(
#   root='./data',
#   train=False, download=False, transform=None
# )

# idx0 = list(range(0, len(mnist_testset), 2))
# idx1 = list(range(1, len(mnist_testset), 2))

# test_dataset_0 = torch.utils.data.Subset(mnist_testset, idx0)
# test_dataset_1 = torch.utils.data.Subset(mnist_testset, idx1)

# torch.save(test_dataset_0, "./data/MNIST/splitted/test_dataset_0")
# torch.save(test_dataset_1, "./data/MNIST/splitted/test_dataset_1")
import torchvision
import torchvision.transforms as transforms

# 加载 CIFAR-100 数据集并对标签进行增加处理
def load_cifar100_with_labels_offset(offset):
    # 定义数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 加载 CIFAR-100 训练集
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

    # 增加标签编号的偏移量
    for i in range(len(trainset)):
        image, label = trainset[i]
        trainset.targets[i] = label + offset

    # 加载 CIFAR-100 测试集
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    # 增加标签编号的偏移量
    for i in range(len(testset)):
        image, label = testset[i]
        testset.targets[i] = label + offset

    return trainset, testset

# 使用偏移量为10加载数据集
trainset, testset = load_cifar100_with_labels_offset(10)

# 验证前10个样本的标签是否增加了10
print(trainset.targets[:10])
print(testset.targets[:10])

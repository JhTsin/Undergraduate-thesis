import torch.nn as nn
import torch
import torchquantum as tq
import torch.nn.functional as F
from tqdm import tqdm
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch.utils.data import DataLoader, WeightedRandomSampler

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#全体目光向我看齐，看我看我看我，我觉得这里qccn的卷积for循环也可以用张量，这样大大加速，but现在我没时间搞这玩意儿
def R_y(theta):
    """Generate the R_y rotation matrix for a given theta in degrees."""

    # 创建并返回R_y(theta)旋转矩阵
    return torch.tensor([
        [torch.cos(theta / 2), -torch.sin(theta / 2)],
        [torch.sin(theta / 2), torch.cos(theta / 2)]
    ],dtype=torch.complex64).to(device)  # 或 torch.complex64 如需与复数张量一起使用

I = torch.tensor([[1, 0], [0, 1]], dtype=torch.complex64).to(device)

CNOT = torch.tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=torch.complex64).to(device)
Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64).to(device)
Z1 = torch.kron(torch.kron(torch.kron(Z, I), I), I).to(device)
Z2 = torch.kron(torch.kron(torch.kron(I, Z), I), I).to(device)
Z3 = torch.kron(torch.kron(torch.kron(I, I), Z), I).to(device)
Z4 = torch.kron(torch.kron(torch.kron(I, I), I), Z).to(device)
CNOT01 = torch.kron(torch.kron(CNOT, I), I).to(device)
CNOT12 = torch.kron(torch.kron(I, CNOT), I).to(device)
CNOT23 = torch.kron(torch.kron(I, I), CNOT).to(device)

class QuanvolutionFilter(tq.QuantumModule):#
    class Qonvlayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.ry0 = tq.RY(has_params=True, trainable=True)#不要加初始值
            self.ry1 = tq.RY(has_params=True, trainable=True)
            self.ry2 = tq.RY(has_params=True, trainable=True)
            self.ry3 = tq.RY(has_params=True, trainable=True)
            self.ry0_parameter = self.ry0.params
            self.ry1_parameter = self.ry1.params
            self.ry2_parameter = self.ry2.params
            self.ry3_parameter = self.ry3.params


        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            init_state = self.q_device.get_states_1d()
            # state = []
            # density = []
            state = torch.stack([s.unsqueeze(0).T for s in init_state])
            #print(states)
            density = torch.matmul(state, state.conj().transpose(-2, -1))
            # for i in range(q_device.bsz):
            #     state.append(init_state[i].unsqueeze(0).T)
            #     density.append(torch.outer(state[i].view(-1), state[i].view(-1).conj()))#初始密度矩阵建立
            #print(state)
            #print("________________")

            #这里可以再再ry0左右cnot[3,0]可能效果会好丢丢
            self.ry0(self.q_device, wires=0) #cancel
            RY0 = torch.kron(torch.kron(torch.kron(R_y(self.ry0_parameter), I), I), I)
            #print(self.ry0_parameter)
            # for i in range(q_device.bsz):
            #     density[i] = RY0 @ density[i] @ RY0.conj().T
            density = torch.matmul(torch.matmul(RY0, density), RY0.conj().transpose(-2, -1))
            #print(density.shape)

            tq.cnot(self.q_device, wires=[0, 1]) #cancel
            # for i in range(q_device.bsz):
            #     density[i] = CNOT01 @ density[i] @ CNOT01.conj().T
            density = torch.matmul(torch.matmul(CNOT01, density), CNOT01.conj().transpose(-2, -1))

            self.ry1(self.q_device, wires=1) #cancel
            RY1 = torch.kron(torch.kron(torch.kron(I, R_y(self.ry1_parameter)), I), I)
            # for i in range(q_device.bsz):
            #     density[i] = RY1 @ density[i] @ RY1.conj().T
            density = torch.matmul(torch.matmul(RY1, density), RY1.conj().transpose(-2, -1))

            tq.cnot(self.q_device, wires=[0, 1]) #cancel
            # for i in range(q_device.bsz):
            #     density[i] = CNOT01 @ density[i] @ CNOT01.conj().T
            density = torch.matmul(torch.matmul(CNOT01, density), CNOT01.conj().transpose(-2, -1))

            tq.cnot(self.q_device, wires=[1, 2]) #cancel
            # for i in range(q_device.bsz):
            #     density[i] = CNOT12 @ density[i] @ CNOT12.conj().T
            density = torch.matmul(torch.matmul(CNOT12, density), CNOT12.conj().transpose(-2, -1))

            self.ry2(self.q_device, wires=2) #cancel
            RY2 = torch.kron(torch.kron(torch.kron(I, I), R_y(self.ry2_parameter)), I)
            # for i in range(q_device.bsz):
            #     density[i] = RY2 @ density[i] @ RY2.conj().T
            density = torch.matmul(torch.matmul(RY2, density), RY2.conj().transpose(-2, -1))

            tq.cnot(self.q_device, wires=[1, 2]) #cancel
            # for i in range(q_device.bsz):
            #     density[i] = CNOT12 @ density[i] @ CNOT12.conj().T
            density = torch.matmul(torch.matmul(CNOT12, density), CNOT12.conj().transpose(-2, -1))

            tq.cnot(self.q_device, wires=[2, 3]) #cancel
            # for i in range(q_device.bsz):
            #     density[i] = CNOT23 @ density[i] @ CNOT23.conj().T
            density = torch.matmul(torch.matmul(CNOT23, density), CNOT23.conj().transpose(-2, -1))

            self.ry3(self.q_device, wires=3) #cancel
            RY3 = torch.kron(torch.kron(torch.kron(I, I), I), R_y(self.ry3_parameter))
            # for i in range(q_device.bsz):
            #     density[i] = RY3 @ density[i] @ RY3.conj().T
            density = torch.matmul(torch.matmul(RY3, density), RY3.conj().transpose(-2, -1))

            tq.cnot(self.q_device, wires=[2, 3]) #cancel
            # for i in range(q_device.bsz):
            #     density[i] = CNOT23 @ density[i] @ CNOT23.conj().T
            density = torch.matmul(torch.matmul(CNOT23, density), CNOT23.conj().transpose(-2, -1))

            return density






    def __init__(self):
        super().__init__()
        self.n_wires = 4

        self.encoder = tq.GeneralEncoder(
        [   {'input_idx': [0], 'func': 'ry', 'wires': [0]},
            {'input_idx': [1], 'func': 'ry', 'wires': [1]},
            {'input_idx': [2], 'func': 'ry', 'wires': [2]},
            {'input_idx': [3], 'func': 'ry', 'wires': [3]},])
        self.q_layer = self.Qonvlayer()   #tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.q_layer1 = self.Qonvlayer()
        self.q_layer2 = self.Qonvlayer()
        self.q_layer3 = self.Qonvlayer()
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x):
        bsz = x.shape[0]
        size = 8
        #print(type(256))
        #x = x.view(bsz, size, size)
        #print(x.size())
        x = torch.squeeze(x)
        #print(x.size())
        #x = x.view(bsz, size, size)
        data_list = []
        # self.q_device3 = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, record_op=True)
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, record_op=True, device=device)
        for c in range(0, size-1, 1):
            for r in range(0, size-1, 1):
                data = torch.transpose(torch.cat((x[:, c, r], x[:, c, r+1], x[:, c+1, r], x[:, c+1, r+1])).view(4, bsz), 0, 1)
                # data = torch.cat((x[:, c, r], x[:, c, r+1], x[:, c+1, r], x[:, c+1, r+1])).view(bsz, 4)
                # data = data*pi
                data = torch.asin(torch.sqrt(data))*2

                # print(len(self.q_device0.op_history))
                self.q_device.reset_op_history()
                self.q_device.reset_states(bsz=bsz)
                self.encoder(self.q_device, data)
                density = self.q_layer(self.q_device)
                out = self.measure(self.q_device)  #cancel
                # expectations = []
                # for i in range(bsz):
                #     expectation_Z1 = torch.trace(Z1 @ density[i]).real
                #     expectation_Z2 = torch.trace(Z2 @ density[i]).real
                #     expectation_Z3 = torch.trace(Z3 @ density[i]).real
                #     expectation_Z4 = torch.trace(Z4 @ density[i]).real
                #     expectations.append(torch.tensor([expectation_Z1, expectation_Z2, expectation_Z3, expectation_Z4]))
                # expectations_tensor = torch.stack(expectations)
                expectation_Z1 = torch.einsum('bij,ji->b', density, Z1).real  # 计算所有批次的迹
                expectation_Z2 = torch.einsum('bij,ji->b', density, Z2).real
                expectation_Z3 = torch.einsum('bij,ji->b', density, Z3).real
                expectation_Z4 = torch.einsum('bij,ji->b', density, Z4).real

                # 将所有期望值合并为一个新的张量
                expectations_tensor = torch.stack([expectation_Z1, expectation_Z2, expectation_Z3, expectation_Z4],dim=1)


                out = out.mean(dim=1).view(bsz, 1).float()*4  # 用mean不用sum,sum下不去
                expectations_tensor = expectations_tensor.mean(dim=1).view(bsz, 1).float() * 4
                # print(expectations_tensor)
                # print(out)
                # print("-------")
                # out = expectations_tensor.mean(dim=1).view(bsz, 1).float() * 4
                # for _ in range(bsz): #我真是天才
                #     tmp = out[_][0] - expectations_tensor[_][0]
                #     out[_][0] = tmp + out[_][0]
                correction = out[:, 0] - expectations_tensor[:, 0]
                out[:, 0] = out[:, 0] - correction

                data_list.append(out) # 97.3 相当于sum了


                # print(data_list)
        # print(data_list)

        result = torch.cat(data_list, dim=1).float()
        # print(result)
        return result

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Net(nn.Module):
    # define nn
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 2, stride=2)
        self.ql1 = QuanvolutionFilter()
        self.fc1 = nn.Linear(7*7, 10)
        self.fc2 = nn.Linear(10*10, 10)

    def forward(self, X):
            bs = X.shape[0]
            X = X.view(bs, 1, 8, 8)
            #X = self.conv1(X)
            #X = F.relu(X)   #问题在这呀relu后太小了
            X = self.ql1(X).to(device)
            #print(X[7])
            X = F.relu(X)
            #print(X.shape)
            X = X.view(bs, -1)

            X = self.fc1(X)
            # X = F.relu(X)
            # X = self.fc2(X)
            return F.log_softmax(X,dim=1)


train_acuracys = []
test_acuracys = []

train_final_acuracys = []
test_final_acuracys = []

train_ensemble_accuracys = []
test_ensemble_accuracys = []

print('start!')
# px = 0.1
# print("噪声水平为",px)
batch_size = 64
lr = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
prev_loss = float('inf')

transform_train = torchvision.transforms.Compose([transforms.Resize((8, 8))
                                            #, transforms.Grayscale(num_output_channels=1)
                                            , torchvision.transforms.ToTensor()
                                            #,transforms.Normalize(0.5, 0.5)
                                            ])
transform_test = torchvision.transforms.Compose([transforms.Resize((8, 8))
                                            #,transforms.Grayscale(num_output_channels=1)
                                            , torchvision.transforms.ToTensor()
                                            #,transforms.Normalize(0.5, 0.5)
                                            ])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform_train)
#train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform_test)

selected_classes = [1,3,5,7]#,2,3,4,5,6,7,8,9
train_images_per_class = 700
test_images_per_class = 300


# 过滤函数，用于选择每个类别的指定数量的图像
def filter_by_class_fixed_number(dataset, classes, num_per_class):
    class_counts = {class_: 0 for class_ in classes}
    indices = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        if class_counts.get(label, 0) < num_per_class and label in classes:
            indices.append(i)
            class_counts[label] += 1
            if all(count == num_per_class for count in class_counts.values()):
                break
    return Subset(dataset, indices)
train_dataset = filter_by_class_fixed_number(train_dataset, selected_classes, train_images_per_class )
test_dataset = filter_by_class_fixed_number(test_dataset, selected_classes, test_images_per_class )
#batch_size = 64  # 或者你选择的其他批次大小
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# AdaBoost 实现
num_epochs = 5
n_classifiers = 10
classifiers = [Net() for _ in range(n_classifiers)]
classifier_weights = torch.zeros(n_classifiers, dtype=torch.float)
data_weights = torch.ones(len(train_dataset), dtype=torch.float) / len(train_dataset)
data_weights.to(device)
def update_data_weights(data_weights, classifier, data_loader, classifier_weight):
    clf.eval()
    data_weights = data_weights.to(device)
    classifier_weight = classifier_weight.to(device)
    with torch.no_grad():
        for i, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            output = classifier(data)
            pred = output.max(1, keepdim=True)[1]
            incorrect = pred.ne(target.view_as(pred)).view(-1).float().to(device)
            incorrect[incorrect == 0] = -1
            data_weights[i*len(data):(i+1)*len(data)] *= torch.exp(classifier_weight * incorrect)
    data_weights /= data_weights.sum()  # 归一化
def calculate_error(clf, train_loader, data_weights, device):
    clf.eval()
    total_error = 0.0
    data_weights = data_weights.to(device) 
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = clf(inputs)
            predicted= outputs.max(1, keepdim=True)[1]
            incorrect = predicted.ne(labels.view_as(predicted)).view(-1)
            weighted_error = torch.dot(data_weights[i*len(inputs):(i+1)*len(inputs)], incorrect.float()) / data_weights.sum()
            total_error += weighted_error
    return total_error.item()

def train(clf, device):
    clf.eval()        
    correct = 0
    train_loss = 0
    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = clf(inputs)
            train_loss += torch.nn.functional.cross_entropy(outputs, labels, reduction='sum').item()
            predicted= outputs.argmax(dim=1, keepdim=True)  # 获取最大概率的索引
            #_, predicted = torch.max(outputs, 1)
            correct += predicted.eq(labels.view_as(predicted)).sum().item()
    train_loss /= len(train_loader.dataset)
    train_acuracy = 100. * correct / len(train_loader.dataset)
    train_acuracys.append(train_acuracy)
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        train_loss, correct, len(train_loader.dataset),
        train_acuracy))
def test(clf, device):
    clf.eval()        
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = clf(inputs)
            outputs = outputs.to(device)
            test_loss += torch.nn.functional.cross_entropy(outputs, labels, reduction='sum').item()
            predicted= outputs.argmax(dim=1, keepdim=True)  # 获取最大概率的索引
            #_, predicted = torch.max(outputs, 1)
            correct += predicted.eq(labels.view_as(predicted)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_acuracy = 100. * correct / len(test_loader.dataset)
    test_acuracys.append(test_acuracy)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        test_acuracy))
    
def final_train(clf, device):
    clf.eval()        
    correct = 0
    train_loss = 0
    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = clf(inputs)
            train_loss += torch.nn.functional.cross_entropy(outputs, labels, reduction='sum').item()
            predicted= outputs.argmax(dim=1, keepdim=True)  # 获取最大概率的索引
            #_, predicted = torch.max(outputs, 1)
            correct += predicted.eq(labels.view_as(predicted)).sum().item()
    train_loss /= len(train_loader.dataset)
    train_acuracy = 100. * correct / len(train_loader.dataset)
    train_final_acuracys.append(train_acuracy)

def final_test(clf, device):
    clf.eval()        
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = clf(inputs)
            outputs = outputs.to(device)
            test_loss += torch.nn.functional.cross_entropy(outputs, labels, reduction='sum').item()
            predicted= outputs.argmax(dim=1, keepdim=True)  # 获取最大概率的索引
            #_, predicted = torch.max(outputs, 1)
            correct += predicted.eq(labels.view_as(predicted)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_acuracy = 100. * correct / len(test_loader.dataset)
    test_final_acuracys.append(test_acuracy)


for i, clf in enumerate(classifiers):
    sampler = WeightedRandomSampler(data_weights, len(data_weights))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    optimizer = torch.optim.Adam(clf.parameters())

    for epoch in range(num_epochs):
        clf.train()
        clf.to(device)
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)#, outputs.to(device)
            outputs = clf(inputs)
            loss = F.cross_entropy(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # 打印训练集上的平均损失
        print(f"第{i}个分类器，epoch为{epoch+1}/{num_epochs}，Loss: {running_loss / len(train_loader)}")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        train(clf, device)
        test(clf, device)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        torch.save(train_acuracys, 'train_acuracys.pt')
        torch.save(test_acuracys, 'test_acuracys.pt')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)    
    final_train(clf, device)
    final_test(clf, device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    torch.save(train_final_acuracys, 'train_final_acuracys.pt')
    torch.save(test_final_acuracys, 'test_final_acuracys.pt')

    # 计算错误率和分类器权重
    error = calculate_error(clf, train_loader, data_weights, device)
    classifier_weight = 0.5*torch.log(torch.tensor((1 - error) / (error+0.00000001)))#+ torch.log(torch.tensor(4-1))
    classifier_weights[i] = classifier_weight

    # 更新数据权重
    update_data_weights(data_weights, clf, train_loader, classifier_weight)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # 定义强分类器函数
    def strong_classifier(data, classifiers, classifier_weights):
        final_output = torch.zeros((len(data), 10)).to(device)
        for weight, clf in zip(classifier_weights, classifiers):
            data, clf = data.to(device), clf.to(device)
            weight = weight.to(device)
            output = clf(data)
            # 使用对数softmax作为弱分类器的输出
            final_output += weight * output
            #print(weight)
        return final_output
    # 计算集成模型的准确率
    def train_ensemble_accuracy(train_loader, classifiers, classifier_weights):
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                # 使用 strong_classifier 函数获取集成模型的输出
                ensemble_output = strong_classifier(data, classifiers, classifier_weights)
                # 获取最大预测值的索引作为预测结果
                pred = ensemble_output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                #print(total)
        return correct / total

    def test_ensemble_accuracy(test_loader, classifiers, classifier_weights):
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                # 使用 strong_classifier 函数获取集成模型的输出
                ensemble_output = strong_classifier(data, classifiers, classifier_weights)
                # 获取最大预测值的索引作为预测结果
                pred = ensemble_output.max(1, keepdim=True)[1].to(device)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                #print(total)
        return correct / total

    # 计算准确率
    #test_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    train_ada_accuracy = train_ensemble_accuracy(train_loader, classifiers, classifier_weights)
    train_ensemble_accuracys.append(train_ada_accuracy)
    print("Ensemble Train Accuracy: {:.2f}%".format(100 * train_ada_accuracy))
    test_ada_accuracy = test_ensemble_accuracy(test_loader, classifiers, classifier_weights)
    test_ensemble_accuracys.append(test_ada_accuracy)
    print("Ensemble Test Accuracy: {:.2f}%".format(100 * test_ada_accuracy))
    print('__________________________')
    torch.save(train_ensemble_accuracys, 'train_ensemble_accuracys.pt')
    torch.save(test_ensemble_accuracys, 'test_ensemble_accuracys.pt')
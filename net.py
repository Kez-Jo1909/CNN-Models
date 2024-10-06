from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import torch.nn as nn
import torch.onnx

net=nn.Sequential(
	nn.Conv2d(1,6,kernel_size=3,padding=1),nn.ReLU(),
	nn.AvgPool2d(kernel_size=2, stride=2),
	nn.Conv2d(6,16,kernel_size=3,padding=1),nn.ReLU(),
	#nn.AvgPool2d(kernel_size=2, stride=2),
	nn.Flatten(),
	nn.Linear(16*14*10,120),
	nn.Linear(120,84),nn.ReLU(),
	nn.Linear(84,5)
)

batch_size=64

transform = transforms.Compose([
	transforms.Resize((28, 20)),
	transforms.Grayscale(num_output_channels=1),
	transforms.ToTensor(),
])

def target_transform(target):
	return target

train_dataset = datasets.ImageFolder(root='/home/kezjo/项目/pytorchtry/data（1）/train/', transform=transform,target_transform=target_transform)
test_dataset = datasets.ImageFolder(root='/home/kezjo/项目/pytorchtry/data（1）/test/', transform=transform,target_transform=target_transform)

train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)



def train_model(net, train_loader, test_loader, num_epochs=150, learning_rate=0.01):
	# 设置设备（使用 GPU 如果可用）
	device = torch.device('cpu')
	net.to(device)

	best_accuracy=0

	# 定义损失函数和优化器
	criterion = nn.CrossEntropyLoss()  # 对于分类问题，通常使用交叉熵损失
	optimizer = optim.Adam(net.parameters(), lr=learning_rate)

	for epoch in range(num_epochs):
		net.train()  # 设置网络为训练模式
		running_loss = 0.0

		for batch_idx, (data, targets) in enumerate(train_loader):
			data, targets = data.to(device), targets.to(device)

			# 前向传播
			outputs = net(data)

			# 计算损失
			loss = criterion(outputs, targets)

			# 清零梯度
			optimizer.zero_grad()

			# 反向传播
			loss.backward()

			# 更新参数
			optimizer.step()

			running_loss += loss.item()
			if (batch_idx + 1) % 100 == 0:  # 每 100 个 batch 打印一次
				print(
					f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {running_loss / (batch_idx + 1):.4f}')

		# 在每个 epoch 结束时进行评估
		net.eval()  # 设置网络为评估模式
		correct = 0
		total = 0

		with torch.no_grad():  # 不需要计算梯度
			for data, targets in test_loader:
				data, targets = data.to(device), targets.to(device)
				outputs = net(data)
				_, predicted = torch.max(outputs.data, 1)
				total += targets.size(0)
				correct += (predicted == targets).sum().item()

		accuracy = 100 * correct / total
		print(f'Epoch [{epoch + 1}/{num_epochs}], Test Accuracy: {accuracy:.2f}%')
		if(accuracy>best_accuracy):
			dummy_input = torch.randn(1, 1, 20, 28)  # 创建一个示例输入张量
			torch.onnx.export(net, dummy_input, '/home/kezjo/models/CNN4-20(m).onnx', opset_version=11,
							  input_names=['input'], output_names=['output'])
			print("ONNX 模型已保存为 CNN4-20(m).onnx")
			best_accuracy = accuracy
		# if(best_accuracy>90):
		#     print('Accuracy>90%,Training process end')
		#     return
	dummy_input = torch.randn(1, 1, 20, 28)
	torch.onnx.export(net,dummy_input,'/home/kezjo/models/CNN4-20(f).onnx', opset_version=11,
							  input_names=['input'], output_names=['output'])

	print('Finished Training')


from datetime import datetime
# 获取当前日期和时间
now = datetime.now()

# 获取当前时间（仅时间部分）
start_time = datetime.now().time()

# 调用训练函数
train_model(net, train_loader, test_loader, num_epochs=20, learning_rate=0.001)


# 获取当前日期和时间
end = datetime.now()

print(start_time)
# 获取当前时间（仅时间部分）
end_time = datetime.now().time()
print(end_time)

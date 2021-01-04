import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import AbSASADataset
from models import NaiveResNet1D, NaiveResNetBlock

test_data = AbSASADataset("data/total_area_data/test_data.npz")
testloader = DataLoader(test_data, batch_size=4, shuffle=True, collate_fn=AbSASADataset.collate_fn)

model = NaiveResNet1D(21, NaiveResNetBlock, 3, 26)
model.load_state_dict(torch.load("models/20210103_net.pth"))
model.eval()

correct = 0
total = 0
with torch.no_grad():
	for data in testloader:
		inputs, labels = data
		outputs = model(inputs)
		_, predicted = torch.max(outputs, 1)

		total += labels.shape[0] * labels.shape[1]
		correct += (predicted == labels).sum().item()
print('Accuracy of the network on the test set: %d %%' % (
	100 * correct / total))

class_correct = np.zeros(test_data.num_bins)
class_distance = np.zeros(test_data.num_bins)
class_total = np.zeros(test_data.num_bins)
with torch.no_grad():
	for data in testloader:
		inputs, labels = data
		outputs = model(inputs)
		_, predicted = torch.max(outputs, 1)
		c = (predicted == labels).squeeze()
		d = torch.abs(torch.sub(predicted, labels)).squeeze()
		for i, label in enumerate(labels):
			for j, l in enumerate(label):
				class_correct[int(l)] += c[i][j].item()
				class_distance[int(l)] += d[i][j].item()
				class_total[int(l)] += 1
for num in range(26):
	print(class_total[num])
	if class_total[num] == 0:
		print('No examples of %5s in test dataset' % (num))
	else:
		print('Accuracy of %5s : %2d %%' % (num, 100 * class_correct[num] / class_total[num]))
		print('Avg Distance of %5s : %2f' % (num, class_distance[num] / class_total[num]))


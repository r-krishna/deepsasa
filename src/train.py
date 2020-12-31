from datasets import AbSASADataset
from models import NaiveResNetBlock, NaiveResNet1D
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# HYPERPARAMETERS
train_val_split = .95
learning_rate = 0.01

def train(path, train_loader, validation_loader, model, epochs, optimizer, criterion, lr_modifier):
	""" trains the model and saves it in the models directory """
	for epoch in range(epochs):
		train_loss = _train_epoch(train_loader, model, optimizer, criterion)
		lr_modifier.step(train_loss)
		avg_training_loss = train_loss/(len(train_loader))
		print('\nAverage training loss (epoch {}): {}'.format(
            epoch, avg_training_loss))

		validation_loss = _validate(validation_loader, model, criterion)
		avg_validation_loss = validation_loss/(len(validation_loader))
		print('\nAverage validation loss (epoch {}): {}'.format(
            epoch, avg_validation_loss))

	torch.save(model.state_dict(), path)
	print('Finished Training')

def _train_epoch(train_loader, model, optimizer, criterion):
	""" trains the model for one epoch"""
	model.train()
	running_loss = 0.0
	for i, data in enumerate(train_loader):
		inputs, labels = data
		optimizer.zero_grad()
		outputs = model(inputs)
		loss = criterion(outputs, labels.long())
		running_loss += loss.item()
		loss.backward()
		optimizer.step()
	return running_loss

def _validate(validation_loader, model, criterion):
	""" validates model every epoch """
	with torch.no_grad():
		model.eval()
		running_loss = 0.0
		for i, data in enumerate(validation_loader):
			inputs, labels = data
			outputs = model(inputs)
			loss = criterion(outputs, labels.long())
			running_loss += loss.item()
	return running_loss

# 21 channels for 20 residues and the chain delimiter 
model = NaiveResNet1D(21, NaiveResNetBlock, 3, 26)
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_type)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=-1)

dataset = AbSASADataset("data/training_data.npz")
train_split = int(len(dataset) * train_val_split)
train_dataset, validation_dataset = random_split(dataset, [train_split, len(dataset)-train_split])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=AbSASADataset.collate_fn)
validation_loader = DataLoader(validation_dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=AbSASADataset.collate_fn)

lr_modifier = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)

train("models/20201230_net.pth", train_loader, validation_loader, model, 30, optimizer, criterion, lr_modifier)

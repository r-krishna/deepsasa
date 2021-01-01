import argparse
from datetime import date
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

def _get_args():
	description = "Script for training a model to predict relative SASA of individual residues in Fabs. The model is trained with a series of 1D convolutions on all antibody structures with a 99 percent similarity cutoff from the SAbDab and using SASA calculations from freesasa."
	parser = argparse.ArgumentParser(description=description)
	
	# Network Architecture
	parser.add_argument("--num_blocks", type=int, default=25, help="number of 1D resnet blocks to use")
	parser.add_argument("--num_bins", type=int, default=26, help="number of bins to classify SASAs into")
	parser.add_argument("--dropout", type=float, default=0.2, help="the probability of entire channels being zerod out")

	# Training Hyperparameters
	parser.add_argument("--epochs", type=int, default=30, help="number of epochs to train")
	parser.add_argument("--batch_size", type=int, default=4, help="number of proteins per batch")
	parser.add_argument("--lr", type=float, default=0.01, help="learning rate for Adam")
	parser.add_argument("--train_val_split", type=float, default=0.95, help="percentage of dataset used for training")
	output_path = "models/{}_net.pth".format(date.today().strftime("%Y%m%d"))
	parser.add_argument("--output_path", type=str, default=output_path)	
	return parser.parse_args()

def main():
	args = _get_args()
	# 21 channels for 20 residues and the chain delimiter 
	model = NaiveResNet1D(21, NaiveResNetBlock, args.num_blocks, args.num_bins)
	device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
	device = torch.device(device_type)

	optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
	# padded values are -1 in labels so do not calculate loss 
	criterion = nn.CrossEntropyLoss(ignore_index=-1)

	dataset = AbSASADataset("data/training_data.npz")
	train_split = int(len(dataset) * args.train_val_split)
	train_dataset, validation_dataset = random_split(dataset, [train_split, len(dataset)-train_split])

	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=AbSASADataset.collate_fn)
	validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=AbSASADataset.collate_fn)

	lr_modifier = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)

	train(args.output_path, train_loader, validation_loader, model, args.epochs, optimizer, criterion, lr_modifier)

if __name__ == '__main__':
    main()
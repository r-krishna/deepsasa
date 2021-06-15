import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from datasets import AbSASADataset
from models import DeepSASAResNet


def evaluate(model, loader, name):
	"""calculates the classification metrics and the confusion matrix for a dataset"""
	model.eval()
	total_predicted = []
	total_labels = []
	output_logits = {}
	with torch.no_grad():
		for i, data in enumerate(loader):
			names, inputs, labels = data
			outputs = model(inputs)
			output_logits.update({name:outputs[i, :, :] for i, name in enumerate(names)})
			_, predicted = torch.max(outputs, 1)
			total_predicted.append(torch.flatten(predicted))
			total_labels.append(torch.flatten(labels))
		y_true = torch.cat(total_labels)
		y_predicted = torch.cat(total_predicted)
		report = pd.DataFrame(data=classification_report(y_true, y_predicted, output_dict=True))
		conf_matrix = pd.DataFrame(data=confusion_matrix(y_true, y_predicted)) 
		report.to_csv("{}_report.csv".format(name))
		conf_matrix.to_csv("{}_confusion.csv".format(name))
		torch.save(output_logits, "{}_output_logits.pth".format(name))


def _get_args():
	description = "This script tests model predictions on an user provided test set"
	parser = argparse.ArgumentParser(description=description)
	parser.add_argument("--model", type=str, help="the state dictionary of the pretrained model that is being evaluated")
	parser.add_argument("--trainset", type=str, help="NPZ file storing the data for the train set")
	parser.add_argument("--testset", type=str, help="NPZ file storing the data for the test set")
	parser.add_argument("--epoch", type=str, help="the epoch number of the model that is being evaluated")
	parser.add_argument("--jobname", type=str, help="name to use for storing output files")
	return parser.parse_args()

def main():
	args = _get_args()
	train_data = AbSASADataset(args.trainset)
	trainloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=AbSASADataset.collate_fn)

	test_data = AbSASADataset(args.testset)
	testloader = DataLoader(test_data, batch_size=4, shuffle=True, collate_fn=AbSASADataset.collate_fn)

	model = DeepSASAResNet(21)
	model_checkpoint = torch.load(args.model)[args.epoch]
	model.load_state_dict(model_checkpoint)

	evaluate(model, trainloader, "{}_train".format(args.jobname))
	evaluate(model, testloader, "{}_test".format(args.jobname))

if __name__ == '__main__':
    main()








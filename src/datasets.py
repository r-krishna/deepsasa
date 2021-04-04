import numpy as np
import torch
from torch.utils.data import Dataset

from data import constants


class AbSASADataset(Dataset):

	def __init__(self, out_file, num_bins=26):
		data = np.load(out_file, allow_pickle=True)
		self.out_file = out_file
		self.files = data.files
		# Calculated the minimum and maximum in the train dataset
		bins = np.linspace(0, 1, num_bins-1)
		self.num_bins = num_bins
		self.bins = np.append(bins, np.inf)

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):

		data = np.load(self.out_file, allow_pickle=True)
		sequence = data[self.files[idx]].item()['sequence']
		sasa = data[self.files[idx]].item()['sasa']
		sequence_encoded = self.one_hot_encode(sequence)
		sasa_encoded = self.flatten_sasa(sasa)
		sample = {"sequence":sequence_encoded, "sasa":sasa_encoded}
		return sample

	@staticmethod
	def one_hot_encode(sequence, alphabet=constants.amino_acids):
		"""One hot encode the sequence: resulting dimension Lx21"""
		sequence_lengths = [len(sequence[chain]) for chain in sequence.keys()]
		encoding = np.zeros((np.sum(sequence_lengths), len(alphabet)+1), dtype=int)
		for chain in sequence.keys():
			for i, res in enumerate(sequence[chain]):
				idx = alphabet.index(res)
				if chain == "H":
					encoding[i][idx] = 1
				elif chain == "L":
					encoding[i+sequence_lengths[0]][idx] = 1
			if chain == "H":
				encoding[sequence_lengths[0]-1][len(alphabet)] = 1
		return encoding

	def flatten_sasa(self, sasa):
		"""
		create a featurized matrix with dimensions Lx1 from the dictionary of sasa values for each residue
		"""
		sequence_lengths = [len(sasa[chain].keys()) for chain in sasa.keys()]
		sasa_matrix = np.zeros(np.sum(sequence_lengths))

		for chain in sasa.keys():
			for i, residue in enumerate(sasa[chain].keys()):
				# check to see what bin each residue sasa falls into
				for idx in range(len(self.bins)-1):
					if sasa[chain][residue] >= self.bins[idx] and sasa[chain][residue] < self.bins[idx+1]:
						if chain == "H":
							sasa_matrix[i] = idx
						elif chain == "L":
							sasa_matrix[i+sequence_lengths[0]] = idx
		return sasa_matrix

	@classmethod
	def collate_fn(cls, batch):
		""" pad tensors in the same batch and convert them into float tensors """
		return cls.pad_data([item['sequence'] for item in batch]).transpose(1,2), cls.pad_data([item['sasa'] for item in batch], pad_value=-1) 

	@staticmethod
	def pad_data(batch, pad_value=0):
		"""takes batched data and converts them into a single stacked tensor"""
		shapes = [_.shape for _ in batch]
		padded_shape = np.max([shape for shape in shapes], 0).tolist()
		padded_tensors = []
		for item in batch:
			# subtract the padded shape from the tensor shape
			to_pad = tuple(map(lambda i, j: i - j, padded_shape, item.shape))
			padded_item = np.pad(item, [(0, to_pad[i]) for i in range(len(to_pad))], mode="constant", constant_values=pad_value)
			padded_tensor = torch.as_tensor(padded_item, dtype=torch.float)
			padded_tensors.append(padded_tensor)
		return torch.stack(padded_tensors)

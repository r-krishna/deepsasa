import os
import numpy as np

from Bio.PDB.PDBParser import PDBParser
import freesasa

import utils, constants


def make_dataset(pdb_dir, out_file):
	""" 
	parse pdb files in pdb_dir and save one hot encoded sequences and sasa values for each residue
	and save to a npz file
	"""
	files = [file for file in os.listdir(pdb_dir)]
	parser = PDBParser()
	with utils.IncrementalNPZ(out_file) as npzfile:
		for file in files:
			name = file.split("/")[-1].split(".")[0]
			structure = parser.get_structure(name, os.path.join(pdb_dir, file))
			sequence = {}
			for model in structure:
				for chain in model:
					sequence[chain.id] = np.array([residue.get_resname() for residue in chain])
			try:
				result, _ = freesasa.calcBioPDB(structure)
			except AssertionError:
				print("{} failed".format(name))
				result = {}

			if result:
				residue_areas = result.residueAreas()
				for chain_key in residue_areas.keys():
					for residue_key in residue_areas[chain_key]:
						residue_areas[chain_key][residue_key] = residue_areas[chain_key][residue_key].total

				data = {"sequence": sequence, "sasa":residue_areas}
				to_save = {name: data}
				npzfile.savez(**to_save)

def load_dataset(out_file):
	"""
	Loads data from npz file 
	"""
	data = np.load(out_file, allow_pickle=True)
	return data
pdb_dir = "test_set/"
out_file = "total_area_data/test_data.npz"
make_dataset(pdb_dir, out_file)


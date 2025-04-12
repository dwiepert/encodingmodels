"""
Helper functions

from https://github.com/HuthLab/encoding-models
"""
import functools 
import warnings
from pathlib import Path
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import cottoncandy as cc

def recursive_add_filetype(dict_or_path):
	if isinstance(dict_or_path,Path):
		return Path(str(dict_or_path) + '.npz')
	elif isinstance(dict_or_path, dict):
		return {k: recursive_add_filetype(v) for k, v in dict_or_path.items()}

def recursive_dict_map(dict_or_str, func):
	if isinstance(dict_or_str, Path):
		return dict_or_str.exists()
	elif isinstance(dict_or_str, str):
		return func(dict_or_str)
	elif isinstance(dict_or_str, dict):
		return {k: recursive_dict_map(v, func) for k, v in dict_or_str.items()}

	raise ValueError(f"type {type(dict_or_str)} unsupported")

def recursive_dict_all(dict_or_bool):
	if isinstance(dict_or_bool, bool):
		return dict_or_bool
	elif isinstance(dict_or_bool, dict):
		return all(recursive_dict_all(v) for v in dict_or_bool.values())

	raise ValueError(f"type {type(dict_or_bool)} unsupported")

def str_is_type(val, test_type):
	try:
		test_type(val)
	except ValueError:
		return False

	return True

# From "banded-ridge.py"
def int_tuple(strings):
	strings = strings.replace("(", "").replace(")", "")
	mapped_int = map(int, strings.split(","))
	return tuple(mapped_int)

# Use this decorator to cache `get_stories_in_sessions`.
# From: https://stackoverflow.com/a/60980685
def listToTuple(function):
	"""
	TODO
	"""
	def wrapper(*args):
		args = [tuple(x) if type(x) == list else x for x in args]
		result = function(*args)
		result = tuple(result) if type(result) == list else result
		return result
	return wrapper

# Cache the responses so we don't keep spamming Corral with small requests.
@listToTuple
@functools.lru_cache(maxsize=128)
def get_stories_in_sessions(sessions, stim_bucket='stimulidb'):
	"""
	TODO
	"""
	cci_stim = cc.get_interface(stim_bucket, verbose=False)
	sess_to_story = cci_stim.download_json('sess_to_story')
	allstories, Pstories = [], []
	for sess in sessions:
		stories, test_story = sess_to_story[sess][0], sess_to_story[sess][1]
		allstories.extend(stories)
		if test_story not in Pstories:
			Pstories.append(test_story)
		if test_story not in allstories:
			allstories.append(test_story)

	Rstories = list(set(allstories) - set(Pstories))
	return allstories, Rstories, Pstories
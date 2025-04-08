"""
Encoding Model class

Author(s): Aditya Vaidya, Daniela Wiepert
Last modified: 02/15/2025
"""
#IMPORTS
##built-in
import functools
import os
import json
from pathlib import Path
from typing import Optional, List, Union
import warnings 

##third-party
import numpy as np
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import cottoncandy as cc

##local
from encodingmodels.utils import *
from database_utils.functions import *

def get_save_location(subject, feature_type, nuisance=None, chunk_sz: float = 0.010, context_sz: float = 64.0):
	# These are obsolete parameters, but are still in the object paths, so we
	# should keep them until we move to a different object store namespace.
	num_sel_frames = 1
	frame_skip = 5

	if nuisance is None:
		nuisance_suffix = ''
	else:
		nuisance_suffix = '_' + nuisance

	return os.path.join("subjects", f"{subject}{nuisance_suffix}",
					f"cnk{chunk_sz:0.1f}_ctx{context_sz:0.1f}_pick_{num_sel_frames}_skip{frame_skip}_{feature_type}")

def config_equals(config_a, config_b) -> bool:
	"""
	Check if two configurations are equal. This function is needed because
	configs may be missing entries, but those entries should have assumed defaults.

	`config_a` and `config_b` are dicts.

	TODO: make a EncodingModelConfiguration class that has a custom comparator.
	"""
	for key in (set(config_a.keys()) | set(config_b.keys())):
		if key == 'alphas':
			if (key not in config_a) or (key not in config_b):
				# Even though np.logspace(1,4,10) is a good default, some older
				# versions of this code used (1,3,10) as a default.
				return False
			# `config[key]` may be a list or array
			if len(config_a[key]) != len(config_b[key]): return False
			# Don't do strict == comparisons on floats!!
			if not np.allclose(config_a[key], config_b[key]): return False

		elif key == 'delays':
			# configs need to have one of 'ndelays' and 'delays'
			if ('delays' not in config_a) and ('ndelays' not in config_a):
				return False
			if ('delays' not in config_b) and ('ndelays' not in config_b):
				return False

			# recompute the delays from `ndelays`, if needed.
			# Don't try to simplify this logic with .get() b/c it'll try to
			# eagerly compute config['ndelays']+1
			if 'delays' in config_a:
				delays_a = config_a['delays']
			else:
				delays_a = list(range(1, config_a['ndelays']+1))
			if 'delays' in config_b:
				delays_b = config_b['delays']
			else:
				delays_b = list(range(1, config_b['ndelays']+1))

			if delays_a != delays_b: return False

		elif key == 'ndelays':
			# configs need to have one of 'ndelays' and 'delays'
			if ('ndelays' not in config_a) and ('delays' not in config_a):
				return False
			if ('ndelays' not in config_b) and ('delays' not in config_b):
				return False

			if 'ndelays' in config_a:
				ndelays_a = config_a['ndelays']
			else:
				# Check if config['delays'] is the same as [1, 2, ...,
				# len(config['delays'])], in which case we've found `ndelays`
				tmp_delays_a = list(range(1, len(config_a['delays'])))
				if tmp_delays_a != config_a['delays']:
					ndelays_a = None
				else:
					ndelays_a = len(config_a['delays'])

			if 'ndelays' in config_b:
				ndelays_b = config_b['ndelays']
			else:
				tmp_delays_b = list(range(1, len(config_b['delays'])))
				if tmp_delays_b != config_b['delays']:
					ndelays_b = None
				else:
					ndelays_b = len(config_b['delays'])

			if ndelays_a != ndelays_b: return False

		elif key == 'Pstory_trim':
			if config_a.get(key, 0) != config_b.get(key, 0): return False

		elif key == 'ignore_dialogue':
			if config_a.get(key, False) != config_b.get(key, False): return False

		elif key == 'scaling_story_splits':
			if config_a.get(key, False) != config_b.get(key, False): return False

		elif key in ['sessions', 'Rstories', 'Pstories']:
			# configs must have sessions OR Rstories+Pstories
			if ('sessions' not in config_a) and (('Rstories' not in config_a) or ('Pstories' not in config_a)):
				return False
			if ('sessions' not in config_b) and (('Rstories' not in config_b) or ('Pstories' not in config_b)):
				return False

			_, _, Rstories_a, Pstories_a = EncodingModel.get_config_stories(
				sessions=config_a.get('sessions', []), Rstories=config_a.get('Rstories', []),
				Pstories=config_a.get('Pstories', []), ignore_dialogue=config_a.get('ignore_dialogue', False),
				scaling_story_splits=config_a.get('scaling_story_splits', False))

			_, _, Rstories_b, Pstories_b = EncodingModel.get_config_stories(
				sessions=config_b.get('sessions', []), Rstories=config_b.get('Rstories', []),
				Pstories=config_b.get('Pstories', []), ignore_dialogue=config_b.get('ignore_dialogue', False),
				scaling_story_splits=config_b.get('scaling_story_splits', False))

			# Need to sort because we're comparing ordered lists, not sets
			if (sorted(Rstories_a) != sorted(Rstories_b)) or (sorted(Pstories_a) != sorted(Pstories_b)):
				return False

		else: # These options have no defaults
			if (key not in config_a) or (key not in config_b):
				return False
			else: # both configs have 'key'
				if config_a[key] != config_b[key]:
					return False

	return True

class EncodingModel(object):
	"""
	TODO
	"""
	def __init__(
			self, subject, out_bucket, feats:dict, feature_type:str, Rstories: List[str],
			Pstories: List[str], sessions:List[str], save_dir:Union[Path,str]=None,
			trim=5, extra_trim=15, ndelays=4, nboots=10, singcutoff=1e-10,
			chunklen=40, nchunks=125, chunk_sz: float = 0.010, context_sz:float=64.0, 
			use_corr=False, single_alpha=False, save_weights=True,
			alphas_logspace=(1,4,10), nuisance=None, overwrite: bool = False, save_crossval=True,
			save_pred=True, delays: Optional[List[int]] = None, ignore_dialogue: bool = False,
			scaling_story_splits: bool = False, Pstory_trim: int = 0):

		self.subject = subject
		self.bucket= out_bucket
		self.downsampled_feat = feats

		if (self.bucket == '') or (self.bucket is None):
			self.cci = None
			self.save_dir = Path(save_dir)
		else:
			self.cci = cc.get_interface(self.bucket, verbose=False)
			self.save_dir = None

		self.Rstories = Rstories
		self.Pstories = Pstories
		self.feature_type = feature_type
		# Let a different function handle the different cases for selecting
		# stories.
		self.ignore_dialogue = ignore_dialogue
		self.scaling_story_splits = scaling_story_splits
		
		assert (len(self.Rstories) > 0) and (len(self.Pstories) > 0), \
			f"insufficient train or test stories ({len(self.Rstories)} and {len(self.Pstories)})"

		# These settings don't actually affect the encoding model (i.e. ridge)
		# computation, and only affect which results are saved. So don't store
		# them in the config
		self.save_weights = save_weights
		self.save_crossval = save_crossval
		self.save_pred = save_pred

		#feature context parameters
		self.chunk_sz = chunk_sz
		self.context_sz = context_sz

		alphas_log_min, alphas_log_max, num_alphas = alphas_logspace
		self.alphas = np.logspace(alphas_log_min, alphas_log_max, num_alphas)

		self.nuisance = nuisance

		self.overwrite = overwrite # Do we overwrite existing results?

		self.config = {
			'sessions': sessions, 'trim': trim, 'extra_trim': extra_trim,'chunklen': chunklen,
			'nchunks': nchunks,
			'ndelays': ndelays, 'nboots': nboots, 'singcutoff': singcutoff, 'use_corr': use_corr,
			'single_alpha': single_alpha, 'ntest': len(self.Pstories),
			'alphas': self.alphas.tolist(), 'ignore_dialogue': ignore_dialogue,
			'scaling_story_splits': scaling_story_splits, 'Rstories': self.Rstories,
			'Pstories': self.Pstories, 'Pstory_trim': Pstory_trim}
		
		if (delays is not None) and len(delays) > 0: # if using `delays`, override any `ndelays` setting
			del self.config['ndelays']
			self.config['delays'] = delays

		self.initialize_config()

        ## GET STORY MRI DATA
		self.cci_resp = cc.get_interface('story-mri-data', verbose=False)

		# Store the expected paths for the results. We can check if they
		# already exist before re-running the regression.
		self.result_paths = {}
		results = ['corrs', 'valphas']
		if self.save_weights: results.append('weights')
		if self.save_crossval: results += ['bscorrs', 'valinds']
		if self.save_pred: results.append('pred')
		self.result_paths = dict(zip(results, results)) # mapping of 'x' --> 'x', so we can define more complicated mappings later

		self.result_paths['test_story_corrs'] = {}
		if self.save_pred: self.result_paths['test_story_pred'] = {}
		if len(self.Pstories) > 1:
			for Pstory in self.Pstories:
				self.result_paths['test_story_corrs'][Pstory] = os.path.join('test_story_corrs', Pstory)
				if self.save_pred:
					self.result_paths['test_story_pred'][Pstory] = os.path.join('test_story_pred', Pstory)

		self.save_location.mkdir(parents=True, exist_ok=True)
		self.result_paths = self._prepend_save_location(self.result_paths)

	@classmethod
	def get_config_stories(cls, sessions: List[str]=[], Rstories: List[str]=[],
						Pstories: List[str]=[], ignore_dialogue: bool=False,
						scaling_story_splits: bool=False):
		"""
		Return the Rstories and Pstories used for the given config
		"""

		allstories = list(set(Rstories + Pstories))

		# If the user specified sessions, then load stories from those.
		# Otherwise, use the specific test & train stories given by user.
		if len(sessions) > 0:
			sessions = np.sort(list(map(int, sessions))) # sort them numerically first
			sessions = list(map(str, sessions))
			allstories, Rstories, Pstories = get_stories_in_sessions(sessions)

		if ignore_dialogue:
			# dialogue{1..6} are not narrative stories like the rest, so we
			# might want to exclude them
			#print('Ignoring "dialogue*" stories. Starting with:', len(allstories), 'stories')
			allstories = list(filter(lambda x: not x.startswith('dialogue'), allstories))
			Rstories = list(filter(lambda x: not x.startswith('dialogue'), Rstories))
			Pstories = list(filter(lambda x: not x.startswith('dialogue'), Pstories))
			#print('After filtering:', len(allstories), 'stories')

		# Save more test story outputs & ignore a few more stories
		if scaling_story_splits:
			# Save outputs for (at least) the 3 main test stories
			Pstories = list(set(Pstories + ['wheretheressmoke', 'fromboyhoodtofatherhood', 'onapproachtopluto']))
			Rstories = list(set(Rstories) - set(Pstories))
			# RJ excludes these stories for other reasons (something to do with
			# OPT tokenization)
			Rstories = list(set(Rstories) - set(['myfirstdaywiththeyankees', 'onlyonewaytofindout']))
			allstories = list(set(Rstories + Pstories))
			#print('After more filtering:', len(allstories), 'stories')

		assert all([(x not in Rstories) for x in Pstories]), "train-test overlap found!"

		return sessions, allstories, Rstories, Pstories

	def _prepend_save_location(self, dict_or_str):
		""" Use this function to recursively prepend the save location to
		everything in the results dictionary.
		TODO 
		"""
		if isinstance(dict_or_str, str):
			if isinstance(self.save_location, Path):
				return self.save_location / dict_or_str
			return os.path.join(self.save_location, dict_or_str)
		elif isinstance(dict_or_str, dict):
			return {k: self._prepend_save_location(v) for k, v in dict_or_str.items()}

	def initialize_config(self):
		"""
		TODO
		"""
		
		save_location = get_save_location(self.subject, self.feature_type, nuisance=self.nuisance,
									chunk_sz=self.chunk_sz, context_sz=self.context_sz)
		# Does an encoding model with this config already exist?
		self.new_config = True
		### CCI
		if self.cci is not None:
			for f in self.cci.lsdir(save_location):
				config_path = os.path.join(f, 'config')
				if not self.cci.exists_object(config_path):
					continue
				elif config_equals(self.config, self.cci.download_json(config_path)):
					# If it exists, populate replace our config with the existing one
					self.new_config = False
					self.save_location = f
					print('Loading config from %s' % self.save_location)
					break

			# If not, save our config into the next available slot (index)
			if self.new_config:
				used_config_num = [0]
				for f in self.cci.lsdir(save_location):
					if not self.cci.exists_object(os.path.join(f, 'config')):
						continue
					used_config_num.append(int(f.replace(save_location+'/', '')))
				config_num  = str(np.amax(used_config_num) + 1)
				self.save_location = os.path.join(save_location, config_num)
				config_path = os.path.join(self.save_location, 'config')
				self.cci.upload_json(config_path, self.config)
				print('Saving config to %s' % self.save_location)

		else:
			save_location = self.save_dir / Path(save_location)
			save_location.mkdir(parents=True, exist_ok=True)
			#list directories in save loca
			for f in [x for x in self.save_dir.iterdir() if x.is_dir()]:
				config_path = f / 'config.json'
				if not config_path.exists():
					continue
				else:
					with open(str(config_path), 'r') as file:
						prev_config = json.load(file)
					if config_equals(self.config, prev_config):
						self.new_config = False
						self.save_location = f
						print('Loading config from %s' % self.save_location)
						break

			if self.new_config:
				used_config_num = [0]
			for f in [x for x in save_location.iterdir() if x.is_dir()]:
				config_path = f / 'config.json'
				if not config_path.exists():
					continue
				used_config_num.append(int(f.name))
				
			config_num  = str(np.amax(used_config_num) + 1)
			self.save_location = save_location / config_num
			self.save_location.mkdir(parents=True, exist_ok=True)
			config_path = self.save_location / 'config.json'
			with open(str(config_path), 'w') as file:
				json.dump(self.config, file, indent=4)
			print('Saving config to %s' % self.save_location)

		# Populate other settings from config
		self.set_config_vars(self.config)
		return

	def set_config_vars(self, config):
		self.sessions = config['sessions']
		if ('Rstories' in config) or ('Pstories' in config):
			self.Rstories = config['Rstories']
			self.Pstories = config['Pstories']
		self.trim = config['trim']
		self.extra_trim = config['extra_trim']
		# Compute delays from ndelays
		if 'ndelays' in config: self.ndelays = config['ndelays']
		if 'delays' in config:
			self.delays = config['delays']
		else:
			self.delays = list(range(1, config['ndelays']+1))
		self.nboots = config['nboots']
		self.chunklen = config['chunklen']
		self.nchunks = config['nchunks']
		self.singcutoff = config['singcutoff']
		self.use_corr = config['use_corr']
		self.single_alpha = config['single_alpha']
		self.alphas = np.array(config['alphas'])
		self.ignore_dialogue = config['ignore_dialogue']
		self.scaling_story_splits = config['scaling_story_splits']
		self.start_tr = 5 + self.trim + self.extra_trim
		self.end_tr = -self.trim
		# Sometimes we *only* want to trim the test stories (e.g. scaling paper)
		self.Pstory_trim = config['Pstory_trim']
		return

	def apply_zscore_and_hrf(self, stories, is_Pstories=False):
		"""Get (z-scored and delayed) stimulus for train and test stories.

		Args:
			stories: List of stimuli stories.

		Variables:
			downsampled_feat (dict): Downsampled feature vectors for all stories.
			trim: Trim downsampled stimulus matrix.
			extra_trim: Extra trim applied to downsampled stimulus matrix to account
						for words at the beginning of the story with insufficient
						context length in neural LMs.
			delays: List of delays to model the hemodynamic response function.

		Returns:
			delstim: <float32>[TRs, features * ndelays]
		"""
		start_tr, end_tr = self.start_tr, self.end_tr
		if is_Pstories: start_tr += self.Pstory_trim

		stim = [zscore_encoding(self.downsampled_feat[s][start_tr:end_tr]) for s in stories]
		stim = np.vstack(stim)
		delays = self.delays
		delstim = make_delayed(stim, delays)
		return delstim

	def preprocess_stimulus(self, features: np.array, do_zscore: bool=True, do_trim=True):
		"""Apply the necessary preprocessing for the given downsampled stimulus features.

		In this case, it (optionally) truncates, then (optionally) z-scores,
		and finally adds delays.
		"""
		if do_trim:
			tr_slice = slice(self.start_tr, self.end_tr)
		else:
			tr_slice = slice(None) # equivalent to [:]

		if do_zscore:
			stim = zscore(features[tr_slice, :])
		else:
			stim = features[tr_slice, :] # TODO: does this copy? avoid if so

		delays = self.delays
		delstim = make_delayed(stim, delays)
		return delstim

	def get_response(self, stories, try_cached=True, is_Pstories=False):
		"""Get the subject's fMRI response for stories."""
		resp = []
		start_trim_trs = self.extra_trim
		if is_Pstories:
			start_trim_trs += self.Pstory_trim

		if self.nuisance is None:
			cci_resp = self.cci_resp # TODO: should be more explicit
			base_path = f"{self.subject}/"
		elif self.nuisance == 'eye-motion':
			cci_resp = cc.get_interface('transformers', verbose=False)
			base_path = f"{self.subject}/{self.nuisance}/1/corrected_resp"
		elif self.nuisance == 'nuisance':
			cci_resp = cc.get_interface('phrase-level-models', verbose=False)
			base_path = f"{self.subject}/{self.nuisance}/1/corrected_resp"
		else:
			raise ValueError(f"nuisance regressor '{self.nuisance}' not valid")

		CACHE_BASE_PATH = Path('cc-responses-cache')
		for story in stories:
			resp_path = os.path.join(base_path, story)
			cached_file_path = Path(CACHE_BASE_PATH / cci_resp.bucket_name / (resp_path + '.npz'))
			if cached_file_path.is_file():
				resp.extend(np.load(cached_file_path)['arr_0'][start_trim_trs:])
			else:
				resp.extend(cci_resp.download_raw_array(resp_path)[start_trim_trs:])
		return np.array(resp)

	def get_prediction_corrs(self, delayed_stim, weights, response, corrs_path, story_name):
		#prediction = np.dot(delayed_stim.astype(delayed_stim), weights)
		prediction = delayed_stim.astype(weights.dtype) @ weights
		corrs = np.zeros((response.shape[1],), dtype=prediction.dtype)
		for vi in range(response.shape[1]):
			corrs[vi] = np.corrcoef(response[:,vi],
									prediction[:,vi].astype(response.dtype))[0,1]
		corrs = np.nan_to_num(corrs)
		
		if self.cci is None:
			np.savez(self.result_paths['test_story_pred'][story_name], corrs)
			if self.save_pred:
				np.savez(self.result_paths['test_story_pred'][story_name], prediction)
		else:
			self.cci.upload_raw_array(self.result_paths['test_story_corrs'][story_name], corrs)
			if self.save_pred:
				self.cci.upload_raw_array(self.result_paths['test_story_pred'][story_name], prediction)
		return
	
	def _check_valphas_local(self, prev_valphas):
		recursive_add_filetype(self.result_paths)
		weights = self.save_location / 'weights.npz'
	
		if (self.save_pred and not self.result_paths['pred'].exists()) and all([v.exists() for v in self.result_paths['test_story_pred'].values()]) and weights.exists():
			print('WARNING: previous weights found, but we are discarding and recomputing them to get the predicted response!!!')
	
		# If ridge parameters have alreadby been found, don't do
		# cross-validation. Ignore this if we want to overwrite previous
		# results.
		# At the time of writing, this should only be run if overwrite=False
		# and weights are not found.
		if (prev_valphas is None) and (not self.overwrite) and \
				self.result_paths['valphas'].exists():
				
			if ('bscorrs' in self.result_paths) and not self.result_paths['bscorrs'].exists():
				print("WARNING: user requested 'bscorrs' but none is found. Proceeding anyway!!!")
			print("Previous valphas found. Reusing and skipping cross-validation...")
			valphas = np.load(self.result_paths['valphas'])
			return self.run_regression(prev_valphas=valphas)

	def _check_valphas_cci(self, prev_valphas):
		# can't use result_paths['weights'] here because it might not be set
		if (self.save_pred and not (self.cci.exists_object(self.result_paths['pred']) and \
			all(map(self.cci.exists_object, self.result_paths['test_story_pred'])))) and \
				self.cci.exists_object(os.path.join(self.save_location, 'weights')):
			print('WARNING: previous weights found, but we are discarding and recomputing them to get the predicted response!!!')

		# If ridge parameters have alreadby been found, don't do
		# cross-validation. Ignore this if we want to overwrite previous
		# results.
		# At the time of writing, this should only be run if overwrite=False
		# and weights are not found.
		if (prev_valphas is None) and (not self.overwrite) and \
				self.cci.exists_object(self.result_paths['valphas']):
			if ('bscorrs' in self.result_paths) and not self.cci.exists_object(self.result_paths['bscorrs']):
				print("WARNING: user requested 'bscorrs' but none is found. Proceeding anyway!!!")
			print("Previous valphas found. Reusing and skipping cross-validation...")
			valphas = self.cci.download_raw_array(self.result_paths['valphas'])
			return self.run_regression(prev_valphas=valphas)
	
	def _save_local(self, wt, corrs, valphas, prev_valphas, bscorrs, valinds, delPstim, pred):
		results = {}
		results['weights'] = wt
		if self.save_weights:
			try:
				np.savez(self.result_paths['weights'], wt)
			except:
				np.savez(self.save_location /'weights.npz', wt)
		
		np.savez(self.result_paths['corrs'] , corrs)
		np.savez(self.result_paths['valphas'], valphas)
		results['corrs'] = corrs
		results['valphas'] = valphas

		# Save cross-validation results only if we actually ran cross-val,
		# and if user wants those results.
		if (prev_valphas is None) and self.save_crossval:
			np.savez(self.result_paths['bscorrs'], bscorrs)
			np.savez(self.result_paths['valinds'], np.array(valinds))
		if self.save_pred:
			if pred is None: pred = delPstim @ wt
			np.savez(self.result_paths['pred'], pred)
			results['pred'] = pred
		print('r2: %f' % np.nansum(corrs * np.abs(corrs)))
		return results
	
	def _save_cci(self, wt, corrs, valphas, prev_valphas, bscorrs, valinds, delPstim, pred):
		results = {}
		base_path = self.save_location
		results['weights'] = wt
		if self.save_weights:
			try:
				self.cci.upload_raw_array(self.result_paths['weights'], wt, compression='Zstd') # zstd is much faster & comparable size to gzip
			except: # cottoncandy ValueError, or other connection error
				if not os.path.exists(base_path):
					os.makedirs(base_path)
				np.savez('%s/weights' % base_path, wt)
		self.cci.upload_raw_array(self.result_paths['corrs'], corrs)
		self.cci.upload_raw_array(self.result_paths['valphas'], valphas)
		results['corrs'] = corrs
		results['valphas'] = valphas

		# Save cross-validation results only if we actually ran cross-val,
		# and if user wants those results.
		if (prev_valphas is None) and self.save_crossval:
			self.cci.upload_raw_array(self.result_paths['bscorrs'], bscorrs)
			self.cci.upload_raw_array(self.result_paths['valinds'], np.array(valinds))
			results['bscorrs'] = bscorrs
			results['valinds'] = np.array(valinds)
		if self.save_pred:
			if pred is None: pred = delPstim @ wt
			self.cci.upload_raw_array(self.result_paths['pred'], pred)
			results['pred'] = pred
		print('r2: %f' % np.nansum(corrs * np.abs(corrs)))

		return results
			
	def run_regression(self, prev_valphas=None):
		"""Run ridge regression for given stimulus and response matrices.

		Variables:
			cci: Cottoncandy bucket interface.
			base_path: Base path where encoding results are stored on corral.
			nboots: Number of validation bootstraps.
			alphas: Ridge penalties to try for regression.
			chunklen: Length of TR chunk held out for validation.
			nchunks: Number of TR chunks held out for validation in each bootstrap.
			singcutoff: Remove singluar values <= this number in SVD for efficiency.
			use_corr : If True, use correlation as the metric of model fit.
					   Else, use variance explained (R-squared).
			downsampled_feat (dict): Downsampled feature vectors for all stories.
			trim: Trim downsampled stimulus matrix.
			extra_trim: Extra trim applied to downsampled stimulus matrix to account
						for words at the beginning of the story with insufficient
						context length in neural LMs.
			delays: List of delays to model the hemodynamic response function.

		NOTE: Some variable descriptions are adapted from the original ridge regression
		code written by Alex Huth! (git repo: speechmodeltutorial)
		"""
		# TODO(shailee): Add more detail/fix variables' description.

		if self.cci is None:
			self._check_valphas_local(prev_valphas)
		else:
			self._check_valphas_cci(prev_valphas)

		print('Stimulus & Response parameters:')
		print('trim: %d, extra_trim: %d, delays: %s' % (self.trim, self.extra_trim, self.delays))
		# Delayed stimulus
		delRstim = self.apply_zscore_and_hrf(self.Rstories)
		print('delRstim: ', delRstim.shape)
		delPstim = self.apply_zscore_and_hrf(self.Pstories, is_Pstories=True)
		print('delPstim: ', delPstim.shape)

		# Response
		zRresp = self.get_response(self.Rstories)
		print('zRresp: ', zRresp.shape)
		zPresp = self.get_response(self.Pstories, is_Pstories=True)
		print('zPresp: ', zPresp.shape)

		# Ridge
		print('Ridge parameters:')
		if isinstance(self.nchunks, int):
			nchunks = self.nchunks
		else:
			# If this is a float, then `self.nchunks` is a _proportion_ of the
			# training TRs
			nchunks = int(zRresp.shape[0] * self.nchunks / self.chunklen)
		print('nboots: %d, chunklen: %d, nchunks: %d, single_alpha: %s, use_corr: %s' % (
			self.nboots, self.chunklen, nchunks, self.single_alpha, self.use_corr))

		pred = None
		if prev_valphas is not None:
			valphas = prev_valphas
			wt = ridge(delRstim, zRresp, valphas, singcutoff=self.singcutoff, normalpha=False, solver_dtype=np.float32)

			# Predict responses on prediction set
			#logger.info("Predicting responses for predictions set..")
			pred = delPstim @ wt

			# Find prediction correlations
			nnpred = np.nan_to_num(pred)
			if self.use_corr:
				corrs = np.nan_to_num(np.array([np.corrcoef(zPresp[:,ii], nnpred[:,ii].ravel())[0,1]
												for ii in range(zPresp.shape[1])]))
			else:
				resvar = (zPresp-pred).var(0)
				Rsqs = 1 - (resvar / zPresp.var(0))
				corrs = np.sqrt(np.abs(Rsqs)) * np.sign(Rsqs)
				del resvar, Rsqs
		else:
			# TODO: Use the `return_wt` argument with `self.save_weights`.
			wt, corrs, valphas, bscorrs, valinds = bootstrap_ridge(
				delRstim, zRresp, delPstim, zPresp, self.alphas, self.nboots, self.chunklen,
				nchunks, singcutoff=self.singcutoff, single_alpha=self.single_alpha,
				use_corr=self.use_corr, solver_dtype=np.float32)

		if self.cci is None:
			results = self._save_local(wt, corrs, valphas, prev_valphas, bscorrs, valinds, delPstim, pred)
		else:
			results = self._save_cci(wt, corrs, valphas, prev_valphas, bscorrs, valinds, delPstim, pred)

		# Evaluate on individual test stories.
		if len(self.Pstories) > 1:
			for story in self.Pstories:
				delPstim = self.apply_zscore_and_hrf([story])
				# NOTE: we're NOT doing Pstory_trim here. This was originally a
				# bug. But will probably be kept for compatibility.
				zPresp = self.get_response([story])
				corrs_path = os.path.join(self.save_location, 'corrs_%s'%story)
				self.get_prediction_corrs(delPstim, wt, zPresp, corrs_path, story_name=story)

		return results


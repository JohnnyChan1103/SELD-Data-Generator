import numpy as np
import librosa
from pathlib import Path
from multiprocessing import Manager
from tqdm.contrib.concurrent import process_map
import functools
import pandas as pd
from tqdm import tqdm


class DBConfig(object):
    def __init__(self, params):
        self._db_path = params['db_path'] # Path to the selected sound event database
        self._db_name = params['db_name']
        self._se_root_path = params['database_dir']
        self._min_samples_per_class = params['min_samples_per_class']
        self._rnd_generator = np.random.default_rng()
        self._classes = set()
        self._nb_classes = 0
        self._sample_list = self._load_db_fileinfo() 
    
    def load_file(self, file_list, sample_list, start, cls, ncl, ns):
        audio_path = file_list[ns][0] 
        if 'FSD50K' in str(audio_path):
            start_time, end_time = file_list[ns][1]
            duration = end_time - start_time
            audio, sr = librosa.load(audio_path, sr=None, 
                                     offset=start_time, duration=duration)
            audio, onoffset = librosa.effects.trim(audio, top_db=30)
            onoffset = onoffset / sr + start_time
            duration = onoffset[1] - onoffset[0]
            onoffset = [onoffset[0], onoffset[1]]
            if duration < 0.25:
                return
            mid = file_list[ns][2]
            timestamps = onoffset
            active_duration = duration
        
        pow_per_sec = np.sum(audio**2) / active_duration
        if pow_per_sec < 1e-4:
            return
        sample_list['class'][start+ns] = ncl
        sample_list['mid'][start+ns] = mid
        sample_list['audiofile'][start+ns] = audio_path
        sample_list['onoffset'][start+ns] = onoffset
        sample_list['duration'][start+ns] = duration
        sample_list['energy_per_sec'][start+ns] = pow_per_sec
        sample_list['timestamps'][start+ns] = timestamps
            

    def _load_db_fileinfo(self):
        file_dict = self._make_selected_filedict()
        manager = Manager()

        print('Preparing sample list...')
        sample_list = manager.dict()
        sample_list.update({
            'class': manager.list(), 'audiofile': manager.list(), 'duration': manager.list(),
            'mid': manager.list(), 'onoffset': manager.list(), 'nSamplesPerClass': np.array([]),
            'meanStdDurationPerClass': np.array([]), 'minMaxDurationPerClass': np.array([]),
            'energy_per_sec': manager.list(), 'energy_quartile': np.array([]), 'timestamps': manager.list()
            })
        start_idx = 0
        for ncl, cls in enumerate(self._classes):
            counter = 0
            file_list = file_dict[cls]
            nb_samples_per_class = len(file_list)
            sample_list['class'].extend([None]*nb_samples_per_class)
            sample_list['mid'].extend([None]*nb_samples_per_class)
            sample_list['audiofile'].extend([None]*nb_samples_per_class)
            sample_list['duration'].extend([None]*nb_samples_per_class)
            sample_list['onoffset'].extend([None]*nb_samples_per_class)
            sample_list['energy_per_sec'].extend([None]*nb_samples_per_class)
            sample_list['timestamps'].extend([None]*nb_samples_per_class)
            print('Loading {} samples from class {}...'.format(nb_samples_per_class, cls))
            process_map(
                functools.partial(
                    self.load_file,
                    file_list, sample_list, start_idx, cls, ncl),
                    range(nb_samples_per_class),
                    max_workers=1,
                    chunksize=1,
                    desc='class {}/{}'.format(ncl+1, self._nb_classes),
                )
            
            while None in sample_list['class']:
                counter += 1
                sample_list['class'].remove(None)
                sample_list['mid'].remove(None)
                sample_list['audiofile'].remove(None)
                sample_list['duration'].remove(None)
                sample_list['onoffset'].remove(None)
                sample_list['energy_per_sec'].remove(None)
                sample_list['timestamps'].remove(None)
            start_idx += nb_samples_per_class - counter 

        sample_list['class'] = np.array(sample_list['class'])
        sample_list['mid'] = np.array(sample_list['mid'])
        sample_list['audiofile'] = np.array(sample_list['audiofile'])
        sample_list['duration'] = np.array(sample_list['duration'])
        sample_list['onoffset'] = np.squeeze(np.array(sample_list['onoffset'], dtype=object))
        sample_list['energy_per_sec'] = np.array(sample_list['energy_per_sec'])
        sample_list['timestamps'] = np.squeeze(np.array(sample_list['timestamps'], dtype=object))
        
        for n_class in range(self._nb_classes):
            class_idx = (sample_list['class'] == n_class)
            sample_list['nSamplesPerClass'] = np.append(sample_list['nSamplesPerClass'], np.sum(class_idx))
            energy_per_sec = np.array(sample_list['energy_per_sec'])
            if n_class == 0:
                sample_list['meanStdDurationPerClass'] = \
                    np.array([[np.mean(sample_list['duration'][class_idx]), 
                               np.std(sample_list['duration'][class_idx])]])
                sample_list['minMaxDurationPerClass'] =  \
                    np.array([[np.min(sample_list['duration'][class_idx]), 
                               np.max(sample_list['duration'][class_idx])]])
                sample_list['energy_quartile'] = \
                    np.array([np.min(energy_per_sec), np.quantile(energy_per_sec, 0.25),
                              np.median(energy_per_sec), np.quantile(energy_per_sec, 0.75), 
                              np.max(energy_per_sec)])
            else:
                sample_list['meanStdDurationPerClass'] = \
                    np.vstack((sample_list['meanStdDurationPerClass'], 
                               np.array([np.mean(sample_list['duration'][class_idx]), 
                                         np.std(sample_list['duration'][class_idx])])))
                sample_list['minMaxDurationPerClass'] = \
                    np.vstack((sample_list['minMaxDurationPerClass'],
                               np.array([np.min(sample_list['duration'][class_idx]), 
                                         np.max(sample_list['duration'][class_idx])])))
                sample_list['energy_quartile'] = \
                    np.vstack((sample_list['energy_quartile'], 
                               np.array([np.min(energy_per_sec), np.quantile(energy_per_sec, 0.25), 
                                         np.median(energy_per_sec), np.quantile(energy_per_sec, 0.75), 
                                         np.max(energy_per_sec)])))
                
        return dict(sample_list)
    

    def _make_selected_filedict(self):
        file_dict = {}
        classwise_file_list = []
        if self._db_name == 'FSD50K':
            db_path = Path(self._db_path) / 'FSD50K'
            classwise_file_list += sorted(db_path.glob('*.tsv'))
        else: 
            raise NotImplementedError('Database {} is not supported.'.format(self._db_name))
        for cls_ix, cls in enumerate(classwise_file_list):
            try:
                _cls_file_list = pd.read_csv(cls, sep='\t', header=None).values
                _cls_file_list = tqdm(_cls_file_list, desc=f'Loading {cls.stem} samples, {cls_ix+1}/{len(classwise_file_list)}...')
            except:
                continue
            _filtered_cls_file_list = []
            if 'FSD50K' in str(cls):
                for _item in _cls_file_list:
                    fname, duration, mids, split = _item

                    audiofile = str(fname) + '.wav'
                    if split == 'train':
                        se_path = self._se_root_path / 'FSD50K/FSD50K.dev_audio'
                    elif split == 'test':
                        se_path = self._se_root_path / 'FSD50K/FSD50K.eval_audio'
                    audiofile = str(se_path/audiofile)
                    duration = librosa.get_duration(path=audiofile)

                    if duration <= 10 and duration >= 0.3:
                        _filtered_cls_file_list.append([audiofile, [0, duration], mids])
                    elif duration > 10:
                        audio, sr = librosa.load(audiofile, sr=None)
                        audio, onoffset = librosa.effects.trim(audio, top_db=30)
                        onoffset = onoffset / sr
                        duration = onoffset[1] - onoffset[0]
                        if duration <= 10:
                            _filtered_cls_file_list.append(
                                [audiofile, onoffset, mids])
                        elif duration > 10 and duration <= 10.5:
                            _filtered_cls_file_list.append(
                                [audiofile, [onoffset[0], onoffset[0]+10], mids])
                        else:
                            nSegments = np.ceil(duration/10).astype(int)
                            for idx in range(nSegments):
                                if idx == nSegments-1:
                                    _filtered_cls_file_list.append(
                                        [audiofile, [onoffset[0]+idx*10, onoffset[1]], mids])
                                else:
                                    _filtered_cls_file_list.append(
                                        [audiofile, [onoffset[0]+idx*10, onoffset[0]+(idx+1)*10], mids])
            
            if cls.stem not in file_dict.keys():
                file_dict[cls.stem] = []
            file_dict[cls.stem] += _filtered_cls_file_list

        filtered_file_dict = {}
        for cls in file_dict.keys():      
            if len(file_dict[cls]) >= self._min_samples_per_class:
                filtered_file_dict[cls] = file_dict[cls]
                self._classes.add(cls)
            else:
                print('Class {} has {} samples, which is less than the minimum number of samples per class ({}).'.format(
                    cls, len(file_dict[cls]), self._min_samples_per_class))  

        self._nb_classes = len(self._classes)
        self._classes = sorted(list(self._classes))
        
        return filtered_file_dict
    

import functools
import os
import json
from pathlib import Path
import pandas as pd
import pickle
from multiprocessing import Manager

import librosa
import numpy as np
import pyroomacoustics as pra
import soundfile as sf
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from sklearn.model_selection import train_test_split

import utils
from srir.ambisonics import Ambisonics as Amb
from srir.srir import GenerateSRIR as SRIR

def get_materials_absorption_database(root_path, surface):
    """ Get materials absorption database.
    """
    assert surface in ['ceiling', 'floor', 'wall'], 'Unknown surface type.'
    files = [file for file in os.listdir(root_path) if surface in file]
    materials = []
    for file in files:
        df = pd.read_csv(os.path.join(root_path, file)).values
        for item in df:
            material = {'description': item[0], 'coeffs': item[1:],
                        'center_freqs': [125, 250, 500, 1000, 2000, 4000]}
            materials.append(material)
    return materials


class DataSynthesizer(object):
    def __init__(self, db_config, params):
        self._db_config = db_config
        self.params = params
        self.max_samples_per_cls = params['nb_events_per_classes']
        self.max_polyphony = {
            'target_classes': params['max_polyphony_target'],
            'interf_classes': params['max_polyphony_interf'],
        }
        self._metadata_path = params['mixturepath'] / 'metadata'
        self._mixture_path = {
            'mic': params['mixturepath'] / 'mic',
            'foa': params['mixturepath'] / 'foa',
            'sum': params['mixturepath'] / 'sum',
        }
        self._classnames = db_config._classes

        self.ontology = json.load(open(params['ontology_path']))
        self.class_labels_indices = {}
        for item in self.ontology:
            self.class_labels_indices[item['id']] = item['name']

        params['target_classes'] = params['target_classes'] \
            if params['target_classes'] != 'all' else list(range(len(self._classnames)))

        params['interf_classes'] = params['interf_classes'] \
            if params['interf_classes'] != 'all' else list(range(len(self._classnames)))

        self._active_classes = {
            'target_classes': np.sort(params['target_classes']),
            'interf_classes': np.sort(params['interf_classes'])
        }
        self._nb_active_classes = {
            'target_classes': len(self._active_classes['target_classes']),
            'interf_classes': len(self._active_classes['interf_classes'])
        }

        # print('Active target classes ({}): {}'.format(self._nb_active_classes['target_classes'], self._active_classes['target_classes']))

        self._mixture_setup = {}
        self._mixture_setup['classnames'] = []
        for cl in self._classnames:
            self._mixture_setup['classnames'].append(cl)
        self._apply_gains = True
        self._class_gains = db_config._sample_list['energy_quartile']
        self._mixture_setup['fs_mix'] = params['fs'] #fs of RIRs
        self._mixture_setup['mixture_duration'] = params['mixture_duration']
        self._mixture_setup['mixture_points'] = int(self._mixture_setup['fs_mix'] * params['mixture_duration'])
        self._nb_mixtures = params['nb_mixtures']
        self._mixture_setup['total_duration'] = self._nb_mixtures * self._mixture_setup['mixture_duration']
        self._mixture_setup['snr_set'] = np.arange(*params['snr_set'])
        self._update_interval = params.get('rir_update_interval', 0.1)  # 默认 0.1s 
        # self._mixture_setup['time_idx_step'] = np.arange(0., self._mixture_setup['mixture_duration'], self._update_interval)
        self._mixture_setup['time_idx_100ms'] = np.arange(0.,self._mixture_setup['mixture_duration'],0.1)
        self._mixture_setup['start_delay'] = np.arange(0.1, params['start_delay'], 0.1)
        #### SRIR setup #####
        self._mixture_setup['room_size_range'] = np.array(params['room_size_range'])
        self._mixture_setup['temperature_range'] = np.arange(*params['temperature_range'])
        self._mixture_setup['humidity_range'] = np.arange(*params['humidity_range'])
        self._mixture_setup['RT60_range'] = params['RT60_range']
        # print(params['RT60_range'])
        self._mixture_setup['mic_pos_range_percentage'] = params['mic_pos_range_percentage']
        self._mixture_setup['src_pos_from_walls'] = params['src_pos_from_walls']

        self._nb_frames = len(self._mixture_setup['time_idx_100ms'])
        # self._nb_frames = len(self._mixture_setup['time_idx_step'])
        self._rnd_generator = np.random.default_rng(seed=params['seed'])
        self._nb_snrs = len(self._mixture_setup['snr_set'])
        self._nb_dealys = len(self._mixture_setup['start_delay'])

        self._trim_threshold = 2. #in seconds, minimum length under which a trimmed event at end is discarded
        
        self._mixtures = {
            'target_classes': [],
            'interf_classes': [],
        }
        self._metadata = {
            'target_classes': [],
            'interf_classes': [],
        }
        self._srir_setup = {
            'target_classes': [],
            'interf_classes': [],
        }

        absortpion_table = pra.materials_data['absorption']
        ceilings, floors, walls = [], [], []
        ceilings += list(absortpion_table['Ceiling absorbers'].keys())
        floors += list(absortpion_table['Floor coverings'].keys()) + \
            ['concrete_floor', 'marble_floor'] * 8 + ['audience_floor', 'stage_floor'] * 3
        walls += list(absortpion_table['Wall absorbers'].keys()) + \
            ['hard_surface', 'brickwork', 'brick_wall_rough', 'limestone_wall']
        ceilings += get_materials_absorption_database(params['materials_path'], 'ceiling')
        floors += get_materials_absorption_database(params['materials_path'], 'floor')
        walls += get_materials_absorption_database(params['materials_path'], 'wall')
        self.materials = {
            'ceilings': ceilings,
            'floors': floors, 
            'walls': walls,
        }

        manager = Manager()
        self.rt60 = manager.list()
        self.rt60.extend([None] * self._nb_mixtures)
        self.mixture_params_file = os.path.join(self.params['mixturepath'], 'mixture_params.csv')
    
    def create_mixtures(self, scenes='target_classes'):
        """ Create mixtures for the target and interf class index.
        """
        
        foldlist = {}

        print('\nGenerating mixtures...\n')

        idx_active1 = np.array([])
        idx_active2 = np.array([])
        for na in range(self._nb_active_classes[scenes]):
            idx_active1 = np.append(idx_active1, \
                np.nonzero(self._db_config._sample_list['class'] == self._active_classes[scenes][na]))
        
        path_dict = dict() # {class: [idx]}
        for idx, path in enumerate(self._db_config._sample_list['audiofile']):
            cls_idx = self._db_config._sample_list['class'][idx]
            cls = self._classnames[cls_idx]
            if cls not in path_dict.keys():
                # path_dict[cls] = np.array([])
                path_dict[cls] = []
            # path_dict[cls] = np.append(path_dict[cls], idx)
            path_dict[cls].append([idx, path])
        
        if self.max_samples_per_cls > 0:
            for cls in path_dict.keys():
                cls_sampleperm = self._rnd_generator.permutation(len(path_dict[cls]))[:self.max_samples_per_cls]
                path_dict[cls] = np.array(path_dict[cls])[cls_sampleperm]
                # path_dict[cls] = path_dict[cls][:self.max_samples_per_cls]

        path_dict_selected = dict()
        rnd = np.random.default_rng(seed=2024)
        for cls in path_dict.keys():
            cls_wise_segments = np.array(path_dict[cls])
            if scenes == 'target_classes':
                train_segments = np.array(
                    [_segment[0] for _segment in cls_wise_segments if 'eval_' not in str(_segment[1])]
                    ).astype('int')
                test_segments = np.array(
                    [_segment[0] for _segment in cls_wise_segments if 'eval_' in str(_segment[1])]
                    ).astype('int')

                if len(train_segments) == 0: segments = test_segments
                elif len(test_segments) == 0: segments = train_segments
                else: segments = np.append(train_segments, test_segments)
                train_segments, test_segments = train_test_split(
                    segments, shuffle=False, test_size=0.1)
                if len(test_segments) < 10:
                    train_segments, test_segments = train_test_split(
                        segments, shuffle=False, test_size=10)

                if self.params['dataset_type'] == 'train':
                    # NOTE: Clips of a few classes may be not enough for training.
                    cls_path_idx = train_segments
                elif self.params['dataset_type'] == 'test':
                    cls_path_idx = test_segments
                else:
                    cls_path_idx = np.array([_segment[0] for _segment in cls_wise_segments]).astype('int')
                idx_active2 = np.append(idx_active2, cls_path_idx)
                path_dict_selected[cls] = cls_path_idx

            elif scenes == 'interf_classes':
                # cls_path_idx = np.array([_segment[0] for _segment in cls_wise_segments]).astype('int')
                # idx_active2 = np.append(idx_active2, cls_path_idx)
                # path_dict_selected[cls] = cls_path_idx
                train_segments = np.array(
                    [_segment[0] for _segment in cls_wise_segments if 'eval_' not in str(_segment[1])]
                    ).astype('int')
                test_segments = np.array(
                    [_segment[0] for _segment in cls_wise_segments if 'eval_' in str(_segment[1])]
                    ).astype('int')

                if len(train_segments) == 0: segments = test_segments
                elif len(test_segments) == 0: segments = train_segments
                else: segments = np.append(train_segments, test_segments)
                train_segments, test_segments = train_test_split(
                    segments, shuffle=False, test_size=0.1)
                if len(test_segments) < 10:
                    train_segments, test_segments = train_test_split(
                        segments, shuffle=False, test_size=10)

                if self.params['dataset_type'] == 'train':
                    # NOTE: Clips of a few classes may be not enough for training.
                    cls_path_idx = train_segments
                elif self.params['dataset_type'] == 'test':
                    cls_path_idx = test_segments
                else:
                    cls_path_idx = np.array([_segment[0] for _segment in cls_wise_segments]).astype('int')
                idx_active2 = np.append(idx_active2, cls_path_idx)
                path_dict_selected[cls] = cls_path_idx

        idx_active1 = idx_active1.astype('int')
        idx_active2 = idx_active2.astype('int')
        # intersection set
        idx_active = np.intersect1d(idx_active1, idx_active2)

        foldlist['class'] = self._db_config._sample_list['class'][idx_active]
        foldlist['mid'] = self._db_config._sample_list['mid'][idx_active]
        foldlist['audiofile'] = self._db_config._sample_list['audiofile'][idx_active]
        foldlist['duration'] = self._db_config._sample_list['duration'][idx_active]
        foldlist['onoffset'] = self._db_config._sample_list['onoffset'][idx_active]
        foldlist['timestamps'] = self._db_config._sample_list['timestamps'][idx_active]

        cls_indices_path = self.params['mixturepath'] / 'cls_indices.tsv'
        cls_indices = np.unique(foldlist['class'])
        num_clips, sum_duration = 0, 0
        f = open(cls_indices_path, 'w')
        for cls_idx in cls_indices:
            cls_mid = self._classnames[cls_idx]
            indices = path_dict_selected[cls_mid]
            duration = np.sum(self._db_config._sample_list['duration'][indices])
            cls_mid = cls_mid[:3].replace('_', '/') + cls_mid[3:]
            label = self.class_labels_indices[cls_mid]
            f.write('{}\t{}\t{}\t{}\t{:.1f}\n'.format(
                cls_idx, cls_mid, label, 
                len(indices), duration))
            num_clips += len(indices)
            sum_duration += duration
        f.close()
        mids_path = self.params['mixturepath'] / 'mids.tsv'
        mids = set()
        for _mids in foldlist['mid']:
            # if isinstance(_mids, str):
            mids.update(_mids.split(','))
        # mids = np.unique(mids)
        f = open(mids_path, 'w')
        for mid in mids:
            label = self.class_labels_indices[mid]
            f.write('{}\t{}\n'.format(mid, label))
        f.close()

        stats_path = self.params['mixturepath'] / 'stats.txt'
        with open(stats_path, 'w') as f:
            f.write('Dataset: {}, number of clips: {}, total duration: {:.1f} hours.\n'.format(
                self.params['dataset_type'], num_clips, sum_duration/3600))
        
        # expand samples that are not enough
        if self.params['dataset_type'] == '???':
            cls_indices = np.unique(foldlist['class'])
            for cls_idx in cls_indices:
                indices = np.where(foldlist['class'] == cls_idx)[0]
                num_tile = 100 // len(indices)
                if num_tile > 0:
                    foldlist['class'] = np.append(foldlist['class'], np.tile(foldlist['class'][indices], num_tile))
                    foldlist['mid'] = np.append(foldlist['mid'], np.tile(foldlist['mid'][indices], num_tile))
                    foldlist['audiofile'] = np.append(foldlist['audiofile'], np.tile(foldlist['audiofile'][indices], num_tile))
                    foldlist['duration'] = np.append(foldlist['duration'], np.tile(foldlist['duration'][indices], num_tile))
                    foldlist['onoffset'] = np.append(
                        foldlist['onoffset'], 
                        np.tile(foldlist['onoffset'][indices], (num_tile, 1)),
                        axis=0)
                    foldlist['timestamps'] = np.array(
                        list(foldlist['timestamps']) + list(foldlist['timestamps'][indices]) * num_tile,
                        dtype=object)
                    print('Expand class {} to {} samples.'.format(cls_idx, len(indices)*(num_tile+1)))

        nb_samples = len(foldlist['duration'])
        sampleperm = self._rnd_generator.permutation(nb_samples)
        foldlist['class'] = foldlist['class'][sampleperm]
        foldlist['mid'] = foldlist['mid'][sampleperm]
        foldlist['audiofile'] = foldlist['audiofile'][sampleperm]
        foldlist['duration'] = foldlist['duration'][sampleperm]
        foldlist['onoffset'] = foldlist['onoffset'][sampleperm]
        foldlist['timestamps'] = foldlist['timestamps'][sampleperm]
        
        iterator = tqdm(range(self._nb_mixtures), total=self._nb_mixtures, desc='Creating mixtures')
        sample_idx = 0
        for nmix in iterator:
            mixture = {}
            mixture['class'] = []
            mixture['mid'] = []
            mixture['audiofile'] = []
            mixture['duration'] = []
            mixture['onoffset'] = []
            mixture['start_time'] = []
            mixture['timestamps'] = []

            if self.params['add_noise']:
                mixture['snr'] = self._rnd_generator.choice(self._mixture_setup['snr_set'])
                mixture['noise'] = '*Gaussian*'
            else:
                mixture['snr'] = None
                mixture['noise'] = None

            for nlayer in range(self.max_polyphony[scenes]):
                # print(f'Create Mixtures: mixture {nmix+1}, layer {nlayer+1}')
                
                #fetch event samples till they add up to the target event time per layer
                event_start_time_in_layer = []
                start_time_in_layer = 0.
                event_idx_in_layer = []
                ev_duration = 0.

                start_time = self._rnd_generator.choice(self._mixture_setup['start_delay'])
                start_time_in_layer = start_time_in_layer + start_time
                while start_time_in_layer < self._mixture_setup['mixture_duration']:
                    event_start_time_in_layer.append(start_time_in_layer)

                    # get event duration
                    ev_duration = foldlist['duration'][sample_idx]
                    event_idx_in_layer.append(sample_idx)

                    start_time = self._rnd_generator.choice(self._mixture_setup['start_delay'])
                    start_time_in_layer = start_time_in_layer + ev_duration + start_time

                    sample_idx += 1
                    if sample_idx == nb_samples:
                        sample_idx = 0
                    
                
                # trim the last event if it is too long
                trimmed_event_length = self._mixture_setup['mixture_duration'] - (start_time_in_layer - ev_duration)

                if trimmed_event_length > self._trim_threshold:
                    TRIMMED_SAMPLE_AT_END = True
                else:
                    TRIMMED_SAMPLE_AT_END = False
                    event_idx_in_layer.pop()
                    event_start_time_in_layer.pop()
                    if sample_idx == 0:
                        sample_idx = nb_samples - 1
                    else:
                        sample_idx -= 1
                
                nb_samples_in_layer = len(event_idx_in_layer)

                for nSample in range(nb_samples_in_layer):
                    event_idx = event_idx_in_layer[nSample]
                    start_time = event_start_time_in_layer[nSample]

                    mixture['class'].append(foldlist['class'][event_idx])
                    mixture['mid'].append(foldlist['mid'][event_idx])
                    mixture['audiofile'].append(foldlist['audiofile'][event_idx])
                    mixture['timestamps'].append(foldlist['timestamps'][event_idx])
                    mixture['start_time'].append(start_time)

                    if nSample == nb_samples_in_layer - 1 and TRIMMED_SAMPLE_AT_END:
                        max_duration = trimmed_event_length
                        onset, offset = foldlist['onoffset'][event_idx]
                        duration = offset - onset
                        onset = onset if duration <= max_duration else \
                            self._rnd_generator.choice(np.arange(onset, offset-max_duration, 0.1))
                        offset = onset + max_duration
                        mixture['duration'].append(max_duration)
                        mixture['onoffset'].append([onset, offset])

                    else:
                        mixture['duration'].append(foldlist['duration'][event_idx])
                        mixture['onoffset'].append(foldlist['onoffset'][event_idx])
            
            self._mixtures[scenes].append(mixture)

        iterator.close()

        # save self._mixtures
        mixtures_path = self.params['mixturepath'] / 'mixtures.obj'
        # print(list(self._mixtures["interf_classes"]))
        with open(mixtures_path, 'wb') as f:
            pickle.dump(self._mixtures, f)
        # with open(mixtures_path, 'rb') as f:
        #     mixtures = pickle.load(f)
        #     print(mixtures_path)
        #     print(f"{len(mixtures['interf_classes'])=}")


    def create_metadata(self, add_interf=True):
        """ Create metadata for the mixture.
        """
        # NOTE: it only supports static sources.

        print('\n Preparing metadata...\n')

        mic_pos_percentage = self._rnd_generator.uniform(
            low=self._mixture_setup['mic_pos_range_percentage'][0],
            high=self._mixture_setup['mic_pos_range_percentage'][1], 
            size=self._nb_mixtures)

        if self._mixture_setup['RT60_range'] is None:
            rt60 = [None] * self._nb_mixtures
        else:
            rt60 = self._rnd_generator.uniform(
                low=self._mixture_setup['RT60_range'][0], 
                high=self._mixture_setup['RT60_range'][1], 
                size=self._nb_mixtures)
        # print(rt60)
        iterator = tqdm(range(self._nb_mixtures), total=self._nb_mixtures, desc='Creating metadata')
        for nmix in iterator:
            nmix_metadata = {
                'classid': [None] * self._nb_frames, 
                'mid': [None] * self._nb_frames,
                'trackid': [None] * self._nb_frames, 
                'eventtimetracks': [None] * self._nb_frames, 
                'eventdoatimetracks': [None] * self._nb_frames
            }
            nmix_setup = {
                'room_size': None, 
                'mic_pos_center': None, 
                'src_pos': [], 
                'rt60':None
            }

            nmix_rt60 = rt60[nmix]

            # Generate appropriate room size
            while True:
                nmix_room_size = self._rnd_generator.uniform(
                    low=self._mixture_setup['room_size_range'][:, 0], 
                    high=self._mixture_setup['room_size_range'][:, 1])
                if nmix_rt60 is None:
                    break
                try:
                    pra.inverse_sabine(nmix_rt60, nmix_room_size)
                    break
                except ValueError:
                    print('ValueError: rt60[{}] = {} for room_size {}'\
                        .format(nmix, nmix_rt60, nmix_room_size))

            nmix_mic_pos_center = mic_pos_percentage[nmix] * nmix_room_size

            nmix_setup['room_size'] = nmix_room_size
            nmix_setup['mic_pos_center'] = nmix_mic_pos_center
            nmix_setup['mic_pos'] = self.params['mic_pos'] + nmix_mic_pos_center
            nmix_setup['rt60'] = nmix_rt60

            num_events_in_mix = len(self._mixtures['target_classes'][nmix]['class'])
            for nEvent in range(num_events_in_mix):
                # Generate appropriate mic position, and make sure it is not too close to the source
                while True:
                    src_pos = self._rnd_generator.uniform(
                        low=self._mixture_setup['src_pos_from_walls'],
                        high=nmix_room_size-self._mixture_setup['src_pos_from_walls'])
                    if np.linalg.norm(src_pos - nmix_mic_pos_center)\
                            > self.params['src_pos_from_listener']:
                        break
                x, y, z = src_pos - nmix_mic_pos_center
                azi, ele, r = np.squeeze(utils.cart2sph(x, y, z))
                # nmix_setup['src_pos'].append(src_pos)

                is_this_event_moving = self._rnd_generator.random() < self.params['moving_ratio']

                start_time = self._mixtures['target_classes'][nmix]['start_time'][nEvent]
                duration = self._mixtures['target_classes'][nmix]['duration'][nEvent]
                start_idx = np.floor(start_time / 0.1)
                end_idx = np.ceil((start_time + duration) / 0.1)
                end_idx = min(end_idx, self._nb_frames)
                active_frames = np.arange(start_idx, end_idx).astype(int)

                # if self.params['is_moving'] and is_this_event_moving:
                #     # 调用之前写的轨迹生成函数
                #     trajectory = self.generate_trajectory(
                #         room_dim=nmix_room_size,
                #         mic_pos_center=nmix_mic_pos_center,
                #         num_points=len(active_frames),
                #         speed_range=self.params['speed_range']
                #     )
                #     nmix_setup['src_pos'].append(trajectory) # 存入数组
                # else:
                #     # 静态声源：为了逻辑统一，存入一个形状为 (1, 3) 的数组
                #     nmix_setup['src_pos'].append(src_pos[np.newaxis, :])

                # 计算混合音频总帧数 (例如 10秒 / 0.1s = 100帧)
                num_total_frames = int(np.round(self._mixture_setup['mixture_duration'] / 0.1))

                if self.params['is_moving']:
                    # --- 移动模式开启：强制生成覆盖全长(10s)的轨迹 ---
                    # 默认让它全程静止在初始点 src_pos
                    full_trajectory = np.tile(src_pos, (num_total_frames, 1)) 
                    
                    if is_this_event_moving:
                        # 如果这个特定事件被标记为移动，则生成移动段并嵌入
                        moving_seg = self.generate_trajectory(
                            room_dim=nmix_room_size,
                            mic_pos_center=nmix_mic_pos_center,
                            num_points=len(active_frames),
                            speed_range=self.params['speed_range']
                        )
                        start_frame = int(start_time / 0.1)
                        end_frame = start_frame + len(moving_seg)
                        actual_end = min(end_frame, num_total_frames)
                        full_trajectory[start_frame:actual_end] = moving_seg[:actual_end-start_frame]
                    
                    nmix_setup['src_pos'].append(full_trajectory) # 存入 (100, 3) 数组

                else:
                    # --- 全静态模式：直接存入原始坐标点 ---
                    nmix_setup['src_pos'].append(src_pos) # 存入 (3,) 数组

                timestamps = self._mixtures['target_classes'][nmix]['timestamps'][nEvent]
                onoffset = self._mixtures['target_classes'][nmix]['onoffset'][nEvent]
                filename = self._mixtures['target_classes'][nmix]['audiofile'][nEvent]
                active_idx = np.ones(len(active_frames), dtype=bool)

                for idx, frame_idx in enumerate(active_frames):
                    # if self.params['is_moving'] and is_this_event_moving:
                    #     print(f"{idx=}, {frame_idx=}, {duration=}, {len(active_frames)=}, {trajectory.shape}")
                    if not active_idx[idx]:
                        continue
                    if nmix_metadata['classid'][frame_idx] is None:
                        nmix_metadata['classid'][frame_idx] = \
                            [self._mixtures['target_classes'][nmix]['class'][nEvent]]
                        nmix_metadata['mid'][frame_idx] = \
                            [self._mixtures['target_classes'][nmix]['mid'][nEvent]]
                        nmix_metadata['trackid'][frame_idx] = [nEvent]
                        nmix_metadata['eventtimetracks'][frame_idx] = \
                            [self._mixtures['target_classes'][nmix]['start_time'][nEvent]]
                        nmix_metadata['eventdoatimetracks'][frame_idx]= [[azi, ele, r]]
                    else:
                        nmix_metadata['classid'][frame_idx].append(
                            self._mixtures['target_classes'][nmix]['class'][nEvent])
                        nmix_metadata['mid'][frame_idx].append(
                            self._mixtures['target_classes'][nmix]['mid'][nEvent])
                        nmix_metadata['trackid'][frame_idx].append(nEvent)
                        nmix_metadata['eventtimetracks'][frame_idx].append(
                            self._mixtures['target_classes'][nmix]['start_time'][nEvent])
                        nmix_metadata['eventdoatimetracks'][frame_idx].append([azi, ele, r])        
    
            self._metadata['target_classes'].append(nmix_metadata)       
            self._srir_setup['target_classes'].append(nmix_setup)

            # Add interference source
            if add_interf:
                nmix_setup_interf = {'src_pos': []}
                num_events_in_mix = len(self._mixtures['interf_classes'][nmix]['class'])
                for nEvent in range(num_events_in_mix):
                    src_pos = self._rnd_generator.uniform(
                        low=self._mixture_setup['src_pos_from_walls'],
                        high=nmix_room_size-self._mixture_setup['src_pos_from_walls']
                    )
                    nmix_setup_interf['src_pos'].append(src_pos)
                self._srir_setup['interf_classes'].append(nmix_setup_interf)
            
        # print(list(self._metadata["interf_classes"]))
        # print(list(self._srir_setup["interf_classes"]))
        # save self._metadata and self._srir_setup
        metadata_path = self.params['mixturepath'] / 'metadata.obj'
        with open(metadata_path, 'wb') as f:
            pickle.dump(self._metadata, f)
        srir_setup_path = self.params['mixturepath'] / 'srir_setup.obj'
        with open(srir_setup_path, 'wb') as f: 
            pickle.dump(self._srir_setup, f)


    def write_metadata(self, scenes='target_classes'):
        r""" Write metadata for the mixture.
        """

        if scenes == 'interf_classes':
            return

        if not os.path.isdir(self._metadata_path):
            Path(self._metadata_path).mkdir(exist_ok=True, parents=True)
        
        print('\n Writing metadata...\n')

        iterator = tqdm(range(self._nb_mixtures), total=self._nb_mixtures, 
                        unit='mixtures', desc='Writing metadata')
        for nmix in iterator:
            mixture = self._metadata[scenes][nmix]
            nmix_per_room = self._nb_mixtures
            nr, nmix_in_room = divmod(nmix, nmix_per_room)
            mixture_name = 'fold0_room{}_mix{}.csv'.format(nr, nmix_in_room)
            # mixture_name = 'fold0_room0_mix{}.csv'.format(nmix)
            file_id = open(os.path.join(self._metadata_path, mixture_name), 'w')
            for frame_idx in range(self._nb_frames):
                if mixture['classid'][frame_idx] is None:
                    continue
                num_events = len(mixture['classid'][frame_idx])
                assert num_events <= self.max_polyphony[scenes], \
                    'Number of events in a frame exceeds the maximum polyphony.'
                for event_idx in range(num_events):
                    classid = mixture['classid'][frame_idx][event_idx]
                    classid = self.params[scenes].index(classid)
                    mid = mixture['mid'][frame_idx][event_idx]
                    # azi, ele, r = mixture['eventdoatimetracks'][frame_idx][event_idx]
                    # file_id.write('{},{},{},{},{},{:.2f},{}\n'.format(
                    #     frame_idx, classid, event_idx, int(azi), int(ele), r, '\"'+mid+'\"'))
                    x, y, z = mixture['eventdoatimetracks'][frame_idx][event_idx]   # 记录笛卡尔坐标而不是球坐标
                    file_id.write('{},{},{},{:.3f},{:.3f},{:.3f},{}\n'.format(
                        frame_idx, classid, event_idx, x, y, z, '\"'+mid+'\"'))
            file_id.close()
        iterator.close()
    

    def synthesize_mixtures(self, add_interf=True, audio_format='both', add_noise=True):
        r""" Synthesize mixtures.
        """
        assert audio_format in ['both', 'foa', 'mic'], \
            'audio_format must be either "both", "foa" or "mic".'
        
        for _subdir in self._mixture_path.keys():
            if not os.path.isdir(self._mixture_path[_subdir]):
                Path(self._mixture_path[_subdir]).mkdir(exist_ok=True, parents=True)
        
        amb_encoding = Amb(
            SH_order=self.params['SH_order'],
            array_type=self.params['array_type'],
            azi=self.params['mic_pos'][:, 0],
            ele=self.params['mic_pos'][:, 1],
            fs=self._mixture_setup['fs_mix'], 
            SH_type=self.params['SH_type'], 
            radius=self.params['radius'],)

        process_map(
            functools.partial(
                self.generate_mixture,
                self._mixtures,
                self._srir_setup,
                self.rt60,
                add_interf,
                add_noise,
                audio_format,
                amb_encoding,
            ),
                range(self._nb_mixtures),
                max_workers=self.params['max_workers'],
                chunksize=self.params['chunksize'],
        )   
        _mixture_params_f = open(self.mixture_params_file, 'a')
        for nmix in range(self._nb_mixtures):
            rt60 = self.rt60[nmix]
            word = 'mix: {}, rt60: {} \n'.format(nmix, rt60)
            _mixture_params_f.writelines(word)
        _mixture_params_f.close()

    def _generate_mixture(self, mixtures, srir_setups, computed_rt60,
                         add_interf, add_noise, audio_format, amb_encoding, nmix):
        """ Write mixture to disk.

        """
        nmix_per_room = self._nb_mixtures
        nr, nmix_in_room = divmod(nmix, nmix_per_room)
        mixture_name = 'fold0_room{}_mix{}.flac'.format(nr, nmix_in_room)
        # mixture_name = 'fold0_room0_mix{}.flac'.format(nmix)

        mixture = mixtures['target_classes'][nmix]
        srir_setup = srir_setups['target_classes'][nmix]

        room_size = srir_setup['room_size']
        target_audio = mixture['audiofile']
        src_pos = srir_setup['src_pos']
        mic_pos_center = srir_setup['mic_pos_center']
        rt60 = srir_setup['rt60']

        srir_generator = SRIR(
            SH_order=self.params['SH_order'],
            fs=self._mixture_setup['fs_mix'],
            mic_pos=self.params['mic_pos'],
            coord_type='cart',
            radius=self.params['radius'],
            array_type=self.params['array_type'],
            tools=self.params['tools'],
        )
        
        src_sig = []
        for event_id, file in enumerate(target_audio):
            onset, offset = mixture['onoffset'][event_id]
            duration = mixture['duration'][event_id]
            start_time = mixture['start_time'][event_id]
            timestamps = mixture['timestamps'][event_id]

            audio, fs = librosa.load(
                path=file, 
                sr=self._mixture_setup['fs_mix'], 
                offset=onset, 
                duration=duration)

            if abs(duration - len(audio) / fs) > 0.2:
                print('Audio length is less than the duration of the event {}: {}s, {}s'.format(
                    file, len(audio) / fs, duration))
                # raise ValueError('Audio length is less than the duration of the event.')
            
            audio = utils.segment_mixtures(
                signal=audio,
                fs=self._mixture_setup['fs_mix'], 
                start=start_time, 
                end=start_time+duration, 
                clip_length=self._mixture_setup['mixture_duration'])
            if self._apply_gains:
                audio = utils.apply_event_gains(
                    audio, duration, self._class_gains, mixture['class'][event_id])
            src_sig.append(audio)
        
        if add_interf:
            mixture_interf = mixtures['interf_classes'][nmix]
            srir_setup_interf = srir_setups['interf_classes'][nmix]
            interf_audio = mixture_interf['audiofile']
            src_pos.extend(srir_setup_interf['src_pos'])

            for event_id, file in enumerate(interf_audio):
                onset, offset = mixture_interf['onoffset'][event_id]
                duration = mixture_interf['duration'][event_id]
                start_time = mixture_interf['start_time'][event_id]
                audio, fs = librosa.load(
                    path=file, 
                    sr=self._mixture_setup['fs_mix'], 
                    offset=onset, 
                    duration=duration)
                audio = utils.segment_mixtures(
                    signal=audio, 
                    fs=fs, 
                    start=start_time, 
                    end=start_time+duration, 
                    clip_length=self._mixture_setup['mixture_duration'])
                if self._apply_gains:
                    audio = utils.apply_event_gains(
                        audio, duration, self._class_gains, mixture_interf['class'][event_id])
                src_sig.append(audio)

        if self.params['tools'] in ['pyroomacoustics', 'gpuRIR']:
            kwargs = {}
            if rt60 is None:
                assert self.params['tools'] == 'pyroomacoustics', 'Only pyroomacoustics supports None RT60.'
                temperature = self._rnd_generator.choice(self._mixture_setup['temperature_range'])
                humidity = self._rnd_generator.choice(self._mixture_setup['humidity_range'])
                ceilings = self._rnd_generator.choice(self.materials['ceilings'])
                floors = self._rnd_generator.choice(self.materials['floors'])
                walls = self._rnd_generator.choice(self.materials['walls'], size=4)
                materials = pra.make_materials(ceiling=ceilings, floor=floors, east=walls[0], 
                                         west=walls[1], north=walls[2], south=walls[3])
                kwargs['materials'] = materials
                kwargs['temperature'] = temperature
                kwargs['humidity'] = humidity
                kwargs['max_order'] = 100
            srir_generator.compute_srir(
                rt60=rt60, 
                room_dim=room_size, 
                src_pos=src_pos,
                method=self.params['method'],
                mic_pos_center=mic_pos_center,
                **kwargs)
        else:
            raise ValueError('Unknown tools for SRIR generation.')
        
        """ Measure RT60 """
        if self.params['tools'] != 'collectedRIR':
            _rt60 = pra.experimental.measure_rt60(
                srir_generator.rir[0][0], fs=self._mixture_setup['fs_mix'], decay_db=60)
            _rt20 = pra.experimental.measure_rt60(
                srir_generator.rir[0][0], fs=self._mixture_setup['fs_mix'], decay_db=20)
            _rt30 = pra.experimental.measure_rt60(
                srir_generator.rir[0][0], fs=self._mixture_setup['fs_mix'], decay_db=30)
            computed_rt60[nmix] = [_rt20, _rt30, _rt60]

        audio_mic = srir_generator.simulate(src_pos_mic=src_pos-mic_pos_center, src_signals=src_sig)
        audio_mic = audio_mic[:, :self._mixture_setup['mixture_points']]
        
        audio_sum = np.sum(src_sig, axis=0, keepdims=True)[:, :self._mixture_setup['mixture_points']]
        clip_path_sum = os.path.join(self._mixture_path['sum'], mixture_name)
        sf.write(file=clip_path_sum, data=0.1*audio_sum.T, samplerate=self._mixture_setup['fs_mix'])

        if add_noise:
            ambience = np.random.randn(4, self._mixture_setup['mixture_points'])
            ambience = ambience / np.max(np.abs(ambience), axis=1, keepdims=True)
            Warning('Gaussian noise is added to the mixture.')
            if ambience.shape[0] < self._mixture_setup['mixture_points']:
                ambience = np.tile(ambience, (1, self._mixture_setup['mixture_points']//ambience.shape[1]+1))[:, :self._mixture_setup['mixture_points']]

            audio_energy = np.sum(np.mean(audio_mic, axis=0)**2)
            ambience_energy = np.sum(np.mean(ambience, axis=0)**2)
            snr = self._rnd_generator.choice(self._mixture_setup['snr_set'])
            ambi_norm = np.sqrt(audio_energy * (10.**(-snr/10.)) / ambience_energy)
            audio_mic += ambi_norm * ambience
            
        clip_path_mic = os.path.join(self._mixture_path['mic'], mixture_name)
        if audio_format in ['mic', 'both']:
            sf.write(file=clip_path_mic, data=audio_mic.T, samplerate=self._mixture_setup['fs_mix'])
        if audio_format in ['foa', 'both']:
            clip_path_foa = os.path.join(self._mixture_path['foa'], mixture_name)
            audio_foa = amb_encoding.encoding(signal=audio_mic)
            audio_foa = audio_foa[:, :self._mixture_setup['mixture_points']]
            # audio_foa /= np.max(audio_foa)
            sf.write(file=clip_path_foa, data=audio_foa.T, samplerate=self._mixture_setup['fs_mix'])
        tqdm.write(mixture_name)

    def __generate_mixture(self, mixtures, srir_setups, computed_rt60,
                         add_interf, add_noise, audio_format, amb_encoding, nmix):
        """ Write mixture to disk.

        """
        nmix_per_room = self._nb_mixtures
        nr, nmix_in_room = divmod(nmix, nmix_per_room)
        mixture_name = 'fold0_room{}_mix{}.flac'.format(nr, nmix_in_room)
        # mixture_name = 'fold0_room0_mix{}.flac'.format(nmix)

        mixture = mixtures['target_classes'][nmix]
        srir_setup = srir_setups['target_classes'][nmix]

        room_size = srir_setup['room_size']
        target_audio = mixture['audiofile']
        src_pos = srir_setup['src_pos']
        mic_pos_center = srir_setup['mic_pos_center']
        rt60 = srir_setup['rt60']

        srir_generator = SRIR(
            SH_order=self.params['SH_order'],
            fs=self._mixture_setup['fs_mix'],
            mic_pos=self.params['mic_pos'],
            coord_type='cart',
            radius=self.params['radius'],
            array_type=self.params['array_type'],
            tools=self.params['tools'],
        )
        
        src_sig = []
        for event_id, file in enumerate(target_audio):
            onset, offset = mixture['onoffset'][event_id]
            duration = mixture['duration'][event_id]
            start_time = mixture['start_time'][event_id]
            timestamps = mixture['timestamps'][event_id]

            audio, fs = librosa.load(
                path=file, 
                sr=self._mixture_setup['fs_mix'], 
                offset=onset, 
                duration=duration)

            if abs(duration - len(audio) / fs) > 0.2:
                print('Audio length is less than the duration of the event {}: {}s, {}s'.format(
                    file, len(audio) / fs, duration))
                # raise ValueError('Audio length is less than the duration of the event.')
            
            audio = utils.segment_mixtures(
                signal=audio,
                fs=self._mixture_setup['fs_mix'], 
                start=start_time, 
                end=start_time+duration, 
                clip_length=self._mixture_setup['mixture_duration'])
            if self._apply_gains:
                audio = utils.apply_event_gains(
                    audio, duration, self._class_gains, mixture['class'][event_id])
            src_sig.append(audio)
        
        if add_interf:
            mixture_interf = mixtures['interf_classes'][nmix]
            srir_setup_interf = srir_setups['interf_classes'][nmix]
            interf_audio = mixture_interf['audiofile']
            src_pos.extend(srir_setup_interf['src_pos'])

            for event_id, file in enumerate(interf_audio):
                onset, offset = mixture_interf['onoffset'][event_id]
                duration = mixture_interf['duration'][event_id]
                start_time = mixture_interf['start_time'][event_id]
                audio, fs = librosa.load(
                    path=file, 
                    sr=self._mixture_setup['fs_mix'], 
                    offset=onset, 
                    duration=duration)
                audio = utils.segment_mixtures(
                    signal=audio, 
                    fs=fs, 
                    start=start_time, 
                    end=start_time+duration, 
                    clip_length=self._mixture_setup['mixture_duration'])
                if self._apply_gains:
                    audio = utils.apply_event_gains(
                        audio, duration, self._class_gains, mixture_interf['class'][event_id])
                src_sig.append(audio)

        if self.params['tools'] in ['pyroomacoustics', 'gpuRIR']:
            kwargs = {}
            if rt60 is None:
                assert self.params['tools'] == 'pyroomacoustics', 'Only pyroomacoustics supports None RT60.'
                temperature = self._rnd_generator.choice(self._mixture_setup['temperature_range'])
                humidity = self._rnd_generator.choice(self._mixture_setup['humidity_range'])
                ceilings = self._rnd_generator.choice(self.materials['ceilings'])
                floors = self._rnd_generator.choice(self.materials['floors'])
                walls = self._rnd_generator.choice(self.materials['walls'], size=4)
                materials = pra.make_materials(ceiling=ceilings, floor=floors, east=walls[0], 
                                         west=walls[1], north=walls[2], south=walls[3])
                kwargs['materials'] = materials
                kwargs['temperature'] = temperature
                kwargs['humidity'] = humidity
                kwargs['max_order'] = 100
            srir_generator.compute_srir(
                rt60=rt60, 
                room_dim=room_size, 
                src_pos=src_pos,
                method=self.params['method'],
                mic_pos_center=mic_pos_center,
                **kwargs)
        else:
            raise ValueError('Unknown tools for SRIR generation.')
        
        """ Measure RT60 """
        if self.params['tools'] != 'collectedRIR':
            _rt60 = pra.experimental.measure_rt60(
                srir_generator.rir[0][0], fs=self._mixture_setup['fs_mix'], decay_db=60)
            _rt20 = pra.experimental.measure_rt60(
                srir_generator.rir[0][0], fs=self._mixture_setup['fs_mix'], decay_db=20)
            _rt30 = pra.experimental.measure_rt60(
                srir_generator.rir[0][0], fs=self._mixture_setup['fs_mix'], decay_db=30)
            computed_rt60[nmix] = [_rt20, _rt30, _rt60]

        audio_mic = srir_generator.simulate(src_pos_mic=src_pos-mic_pos_center, src_signals=src_sig)
        audio_mic = audio_mic[:, :self._mixture_setup['mixture_points']]
        
        audio_sum = np.sum(src_sig, axis=0, keepdims=True)[:, :self._mixture_setup['mixture_points']]
        clip_path_sum = os.path.join(self._mixture_path['sum'], mixture_name)
        sf.write(file=clip_path_sum, data=0.1*audio_sum.T, samplerate=self._mixture_setup['fs_mix'])

        if add_noise:
            ambience = np.random.randn(audio_mic.shape[0], self._mixture_setup['mixture_points'])
            ambience = ambience / np.max(np.abs(ambience), axis=1, keepdims=True)
            Warning('Gaussian noise is added to the mixture.')
            if ambience.shape[0] < self._mixture_setup['mixture_points']:
                ambience = np.tile(ambience, (1, self._mixture_setup['mixture_points']//ambience.shape[1]+1))[:, :self._mixture_setup['mixture_points']]

            audio_energy = np.sum(np.mean(audio_mic, axis=0)**2)
            ambience_energy = np.sum(np.mean(ambience, axis=0)**2)
            # snr = self._rnd_generator.choice(self._mixture_setup['snr_set'])
            snr = mixture.get('snr', 20)
            ambi_norm = np.sqrt(audio_energy * (10.**(-snr/10.)) / ambience_energy)
            audio_mic += ambi_norm * ambience
            
        clip_path_mic = os.path.join(self._mixture_path['mic'], mixture_name)
        if audio_format in ['mic', 'both']:
            sf.write(file=clip_path_mic, data=audio_mic.T, samplerate=self._mixture_setup['fs_mix'])
        if audio_format in ['foa', 'both']:
            clip_path_foa = os.path.join(self._mixture_path['foa'], mixture_name)
            audio_foa = amb_encoding.encoding(signal=audio_mic)
            audio_foa = audio_foa[:, :self._mixture_setup['mixture_points']]
            sf.write(file=clip_path_foa, data=audio_foa.T, samplerate=self._mixture_setup['fs_mix'])
        tqdm.write(mixture_name)

    def ___generate_mixture(self, mixtures, srir_setups, computed_rt60,
                        add_interf, add_noise, audio_format, amb_encoding, nmix):
        """
        支持移动声源的混合音频生成函数
        """
        # --- 1. 基础参数准备 ---
        nmix_per_room = self._nb_mixtures
        nr, nmix_in_room = divmod(nmix, nmix_per_room)
        mixture_name = 'fold0_room{}_mix{}.flac'.format(nr, nmix_in_room)
        print('Writing room {}, mixture {}/{} '.format(nr+1, nmix+1,nmix_per_room ))

        mixture = mixtures['target_classes'][nmix]
        srir_setup = srir_setups['target_classes'][nmix]
        
        room_size = srir_setup['room_size']
        mic_pos_center = srir_setup['mic_pos_center']
        rt60 = srir_setup['rt60']
        
        # 间隔参数（从params读取）
        update_interval = self.params.get('rir_update_interval', 0.1)
        is_moving_mode = self.params.get('is_moving', False)

        # 初始化 SRIR 生成器
        srir_generator = SRIR(
            SH_order=self.params['SH_order'],
            fs=self._mixture_setup['fs_mix'],
            mic_pos=self.params['mic_pos'],
            coord_type='cart',
            radius=self.params['radius'],
            array_type=self.params['array_type'],
            tools=self.params['tools'],
        )

        # --- 2. 收集信号与轨迹 ---
        src_sigs = []
        all_trajectories = [] # 存储 List[np.ndarray(N, 3)]
        
        # 合并处理目标声源和干扰声源
        event_lists = [('target_classes', mixture)]
        if add_interf:
            event_lists.append(('interf_classes', mixtures['interf_classes'][nmix]))

        for scene_type, mix_data in event_lists:
            for event_id, file in enumerate(mix_data['audiofile']):
                onset, offset = mix_data['onoffset'][event_id]
                duration = mix_data['duration'][event_id]
                start_time = mix_data['start_time'][event_id]

                # A. 加载并预处理音频
                audio, _ = librosa.load(path=file, sr=self._mixture_setup['fs_mix'], 
                                    offset=onset, duration=duration)
                # 补齐/切割到混合物总长度
                audio = utils.segment_mixtures(
                    signal=audio, fs=self._mixture_setup['fs_mix'], 
                    start=start_time, end=start_time+duration, 
                    clip_length=self._mixture_setup['mixture_duration'])
                
                if self._apply_gains:
                    audio = utils.apply_event_gains(audio, duration, self._class_gains, mix_data['class'][event_id])
                src_sigs.append(audio)

                # B. 生成或获取轨迹
                # 在 create_metadata 阶段我们应该已经把轨迹存入了 srir_setup
                # 如果是静态旧脚本，srir_setup['src_pos'][event_id] 是单个点 [x,y,z]
                pos_data = srir_setups[scene_type][nmix]['src_pos'][event_id]
                all_trajectories.append(pos_data)
                
                # if is_moving_mode:
                #     # 确保是 (N, 3) 形状。如果是静态点，通过 np.tile 变成序列
                #     if pos_data.ndim == 1 or pos_data.shape[0] == 1: 
                #         # print(f"Event {event_id} is not moving, ")
                #         num_frames = int(np.ceil(self._mixture_setup['mixture_duration'] / update_interval))
                #         traj = np.tile(pos_data, (num_frames, 1))
                #     else:
                #         # print(f"Event {event_id} is moving, ")
                #         traj = pos_data
                #     all_trajectories.append(traj)
                # else:
                #     # 传统静态模式，只取第一个点
                #     static_pos = pos_data[0] if pos_data.ndim > 1 else pos_data
                #     all_trajectories.append(static_pos)
        # NOTE: 这里有bug， 当duration=10.时，有可能len(all_trajectories)=0，duration调大可以解决，暂不清楚原因
        # print(f"{nmix}: {len(event_lists)=}")
        # print(f"{nmix}: {len(all_trajectories)=}")
        # print(f"{nmix}: {all_trajectories[0].shape=}")
        # --- 3. 批量计算 RIR ---
        kwargs = {}
        # if is_moving_mode:
        #     # 调用我们之前写的批量移动 RIR 函数
        #     # 返回 List[np.ndarray(mic, N, rir_len)]
        #     moving_rirs = srir_generator.compute_moving_srir_gpuRIR(
        #         room_dim=room_size,
        #         trajectories=all_trajectories,
        #         rt60=rt60,
        #         mic_pos_center=mic_pos_center
        #     )
            
        #     # --- 4. 移动渲染 ---
        #     audio_mic = srir_generator.simulate_moving(
        #         src_signals=src_sigs,
        #         moving_rirs=moving_rirs,
        #         update_interval=update_interval,
        #         crossfade_len=self.params.get('crossfade_len', 0.02)
        #     )

        #     # audio_mic = srir_generator.simulate_moving_ltv(
        #     #     src_signals=src_sigs,
        #     #     moving_rirs=moving_rirs,
        #     #     update_interval=update_interval,
        #     #     win_size=1024
        #     # )
        # else:
        #     # 传统静态渲染逻辑
        #     srir_generator.compute_srir(rt60=rt60, room_dim=room_size, src_pos=all_trajectories, method=self.params['method'],
        #         mic_pos_center=mic_pos_center,
        #         **kwargs)
        #     audio_mic = srir_generator.simulate(src_pos_mic=all_trajectories-mic_pos_center, src_signals=src_sigs)
        # --- 3. 根据全局开关切换渲染引擎 ---
        if is_moving_mode:
            # --- 移动模式渲染 ---
            # 此时 all_trajectories 是一个包含多个 (100, 3) 数组的 List
            moving_rirs = srir_generator.compute_moving_srir_gpuRIR(
                room_dim=room_size,
                trajectories=all_trajectories,
                rt60=rt60,
                mic_pos_center=mic_pos_center
            )
            # audio_mic = srir_generator.simulate_moving(
            #     src_signals=src_sigs,
            #     moving_rirs=moving_rirs,
            #     update_interval=update_interval
            # )
            audio_mic = srir_generator.simulate_moving_ltv(
                src_signals=src_sigs,
                moving_rirs=moving_rirs,
                update_interval=update_interval
            )
        else:
            # --- 原始静态渲染逻辑 ---
            # 此时 all_trajectories 是一个包含多个 (3,) 坐标点的 List
            # 关键修复：将其转换为 Numpy 数组，避免 List 减法报错
            src_pos_array = np.array(all_trajectories) 
            
            srir_generator.compute_srir(
                rt60=rt60, 
                room_dim=room_size, 
                src_pos=src_pos_array, # 传入 (N_src, 3) 数组
                method=self.params['method'],
                mic_pos_center=mic_pos_center,
                **kwargs
            )
            # 计算相对位置并渲染
            audio_mic = srir_generator.simulate(
                src_pos_mic = src_pos_array - mic_pos_center, 
                src_signals = src_sigs
            )

        # --- 5. 后处理与保存 (与原脚本一致) ---
        audio_mic = audio_mic[:, :self._mixture_setup['mixture_points']]
        
        audio_sum = np.sum(src_sigs, axis=0, keepdims=True)[:, :self._mixture_setup['mixture_points']]
        clip_path_sum = os.path.join(self._mixture_path['sum'], mixture_name)
        sf.write(file=clip_path_sum, data=0.1*audio_sum.T, samplerate=self._mixture_setup['fs_mix'])

        if add_noise:
            ambience = np.random.randn(audio_mic.shape[0], self._mixture_setup['mixture_points'])
            ambience = ambience / np.max(np.abs(ambience), axis=1, keepdims=True)
            Warning('Gaussian noise is added to the mixture.')
            if ambience.shape[0] < self._mixture_setup['mixture_points']:
                ambience = np.tile(ambience, (1, self._mixture_setup['mixture_points']//ambience.shape[1]+1))[:, :self._mixture_setup['mixture_points']]

            audio_energy = np.sum(np.mean(audio_mic, axis=0)**2)
            ambience_energy = np.sum(np.mean(ambience, axis=0)**2)
            snr = self._rnd_generator.choice(self._mixture_setup['snr_set'])
            # snr = mixture.get('snr', 20)
            ambi_norm = np.sqrt(audio_energy * (10.**(-snr/10.)) / ambience_energy)
            audio_mic += ambi_norm * ambience
            
        clip_path_mic = os.path.join(self._mixture_path['mic'], mixture_name)
        if audio_format in ['mic', 'both']:
            sf.write(file=clip_path_mic, data=audio_mic.T, samplerate=self._mixture_setup['fs_mix'])
        if audio_format in ['foa', 'both']:
            clip_path_foa = os.path.join(self._mixture_path['foa'], mixture_name)
            audio_foa = amb_encoding.encoding(signal=audio_mic)
            audio_foa = audio_foa[:, :self._mixture_setup['mixture_points']]
            sf.write(file=clip_path_foa, data=audio_foa.T, samplerate=self._mixture_setup['fs_mix'])
        tqdm.write(mixture_name)

    def generate_mixture(self, mixtures, srir_setups, computed_rt60,
                            add_interf, add_noise, audio_format, amb_encoding, nmix):
        """
        严格对齐参考脚本逻辑的混合音频生成函数
        """
        import scipy.signal
        import os
        import soundfile as sf

        # --- 1. 基础参数准备 ---
        nmix_per_room = self._nb_mixtures
        nr, nmix_in_room = divmod(nmix, nmix_per_room)
        mixture_name = 'fold0_room{}_mix{}.flac'.format(nr, nmix_in_room)
        print('Writing room {}, mixture {}/{} '.format(nr+1, nmix+1, nmix_per_room))

        mixture = mixtures['target_classes'][nmix]
        srir_setup = srir_setups['target_classes'][nmix]
        
        room_size = srir_setup['room_size']
        mic_pos_center = srir_setup['mic_pos_center']
        rt60 = srir_setup['rt60']
        
        fs = self._mixture_setup['fs_mix']
        mixture_points = self._mixture_setup['mixture_points']
        update_interval = self.params.get('rir_update_interval', 0.1)
        is_moving_mode = self.params.get('is_moving', False)

        # 初始化全局输出容器
        mixsig = np.zeros((mixture_points, len(self.params['mic_pos']))) # 假设4通道输出
        sumsig = np.zeros((mixture_points))

        # 初始化 SRIR 生成器
        srir_generator = SRIR(
            SH_order=self.params['SH_order'],
            fs=fs,
            mic_pos=self.params['mic_pos'],
            coord_type='cart',
            radius=self.params['radius'],
            array_type=self.params['array_type'],
            tools=self.params['tools'],
        )

        # --- 2. 收集信号与轨迹 ---
        all_event_audio = []    # 存储未补齐的干燥信号片段
        all_trajectories = []   # 存储 RIR 轨迹序列
        event_metadata_list = [] # 存储事件元数据用于渲染

        event_lists = [('target_classes', mixture)]
        if add_interf:
            event_lists.append(('interf_classes', mixtures['interf_classes'][nmix]))

        for scene_type, mix_data in event_lists:
            for event_id, file in enumerate(mix_data['audiofile']):
                onset, offset = mix_data['onoffset'][event_id]
                duration = mix_data['duration'][event_id]
                start_time = mix_data['start_time'][event_id]
                class_idx = int(mix_data['class'][event_id])

                # A. 仅加载有效片段 (Strict Alignment)
                audio, _ = librosa.load(path=file, sr=fs, mono=True, offset=onset, duration=duration)
                
                # B. 准备轨迹
                pos_data = srir_setups[scene_type][nmix]['src_pos'][event_id]
                
                # 根据 duration 计算所需的 RIR 数量
                num_frames_event = int(np.ceil(duration / update_interval))
                
                if is_moving_mode:
                    if pos_data.ndim == 1: # 静态点
                        traj = np.tile(pos_data, (num_frames_event, 1))
                        moving_flag = False
                    else: # 移动轨迹
                        traj = pos_data[:num_frames_event]
                        moving_flag = True
                    all_trajectories.append(traj)
                else:
                    # 全静态模式逻辑
                    traj = pos_data[0] if pos_data.ndim > 1 else pos_data
                    all_trajectories.append(traj)
                    moving_flag = False

                all_event_audio.append(audio)
                event_metadata_list.append({
                    'class': class_idx,
                    'start_time': start_time,
                    'duration': duration,
                    'is_moving': moving_flag
                })

        # --- 3. 批量计算 RIR ---
        if is_moving_mode:
            moving_rirs_batch = srir_generator.compute_moving_srir_gpuRIR(
                room_dim=room_size,
                trajectories=all_trajectories,
                rt60=rt60,
                mic_pos_center=mic_pos_center
            )
        else:
            src_pos_array = np.array(all_trajectories) 
            srir_generator.compute_srir(
                rt60=rt60, room_dim=room_size, src_pos=src_pos_array, 
                method=self.params['method'], mic_pos_center=mic_pos_center
            )

        # --- 4. 空间化渲染与能量对齐 (严格对齐参考脚本逻辑) ---
        for i, eventsig in enumerate(all_event_audio):
            meta = event_metadata_list[i]
            
            # A. 渲染
            if is_moving_mode:
                rirs = moving_rirs_batch[i]
                if meta['is_moving']:
                    # LTV 渲染：转换形状为 (rir_len, mic, frames)
                    irs_ltv = np.transpose(rirs, (2, 0, 1))
                    ir_times = np.arange(irs_ltv.shape[2]) * update_interval
                    # 应用补偿系数 481.6989 / len
                    mixeventsig = 481.6989 * utils.ctf_ltv_direct(
                        eventsig, irs_ltv, ir_times, fs, self.params.get('win_size', 1024)
                    ) / float(len(eventsig))
                else:
                    # 即使是移动模式下的静态声源，也取第一帧 RIR 卷积
                    chan_sigs = [scipy.signal.convolve(eventsig, rirs[m, 0, :], mode='full', method='fft') 
                                for m in range(rirs.shape[0])]
                    mixeventsig = np.stack(chan_sigs, axis=1)
            else:
                # 原始静态渲染
                chan_sigs = [scipy.signal.convolve(eventsig, srir_generator.rir[m][i], mode='full', method='fft') 
                            for m in range(len(srir_generator.rir))]
                mixeventsig = np.stack(chan_sigs, axis=1)

            # B. 能量四分位对齐 (Strict Alignment)
            if self._apply_gains:
                K = 1000
                class_gains = self._class_gains[meta['class']]
                rand_energies = utils.sample_from_quartiles(K, class_gains)
                # 取中间部分能量
                intr_quart_energies = rand_energies[K + np.arange(2*(K+1))]
                target_energy_per_sec = intr_quart_energies[self._rnd_generator.integers(len(intr_quart_energies))]
                target_total_energy = target_energy_per_sec * meta['duration']

                # 计算 Omni 能量
                if audio_format == 'mic':
                    event_omni_energy = np.sum(np.sum(mixeventsig, axis=1)**2)
                else: # foa
                    event_omni_energy = np.sum(mixeventsig[:, 0]**2)

                norm_gain = np.sqrt(target_total_energy / (event_omni_energy + 1e-9))
                mixeventsig *= norm_gain
                eventsig *= norm_gain

            # C. 叠加到混合物 (处理 start_time)
            start_sample = int(np.round(meta['start_time'] * fs))
            
            # 空间化信号叠加
            l_mix_ev = mixeventsig.shape[0]
            end_mix = min(start_sample + l_mix_ev, mixture_points)
            if end_mix > start_sample:
                mixsig[start_sample:end_mix, :] += mixeventsig[:end_mix-start_sample, :]

            # 干燥信号叠加 (sumsig)
            l_dry_ev = len(eventsig)
            end_dry = min(start_sample + l_dry_ev, mixture_points)
            if end_dry > start_sample:
                sumsig[start_sample:end_dry] += eventsig[:end_dry-start_sample]

        # --- 5. 全局归一化 (对齐参考脚本的 gnorm = 0.5) ---
        max_val = np.max(np.abs(mixsig))
        if max_val > 0:
            gnorm = 0.5 / max_val
            mixsig *= gnorm
            sumsig *= gnorm

        # --- 6. 注入背景噪声 (对齐参考脚本的 SNR 逻辑) ---
        if add_noise:
            if hasattr(self, 'ambience') and self.ambience is not None:
                l_ambience = self.ambience.shape[1]
                start_noise_idx = self._rnd_generator.integers(0, l_ambience - mixture_points)
                noise_seg = self.ambience[:, start_noise_idx : start_noise_idx + mixture_points].T
                
                if audio_format == 'mic':
                    target_omni = np.sum(np.mean(mixsig, axis=1)**2)
                    noise_omni = np.sum(np.mean(noise_seg, axis=1)**2)
                else:
                    target_omni = np.sum(mixsig[:, 0]**2)
                    noise_omni = np.sum(noise_seg[:, 0]**2)
                
                snr = self._rnd_generator.choice(self._mixture_setup['snr_set'])
                ambi_norm = np.sqrt(target_omni * (10.**(-snr/10.)) / (noise_omni + 1e-9))
                mixsig += ambi_norm * noise_seg
                
        # --- 7. 保存文件 ---
        sf.write(os.path.join(self._mixture_path['mic'], mixture_name), mixsig, fs)
        sf.write(os.path.join(self._mixture_path['sum'], mixture_name), sumsig[:, None], fs)

        if audio_format in ['foa', 'both']:
            audio_foa = amb_encoding.encoding(signal=mixsig.T)
            sf.write(os.path.join(self._mixture_path['foa'], mixture_name), audio_foa.T, fs)

        tqdm.write(mixture_name)

    def generate_trajectory(self, room_dim, mic_pos_center, num_points, speed_range=(0.5, 2.0)):
        """
        根据自定义间隔生成随机游走轨迹
        """
        # 计算需要的轨迹点数
        # num_points = int(np.ceil(duration / self._update_interval))
        
        # 初始位置约束检查
        while True:
            start_pos = self._rnd_generator.uniform(
                low=self._mixture_setup['src_pos_from_walls'],
                high=room_dim - self._mixture_setup['src_pos_from_walls']
            )
            if np.linalg.norm(start_pos - mic_pos_center) > self.params['src_pos_from_listener']:
                break
        
        trajectory = [start_pos]
        direction = self._rnd_generator.standard_normal(3)
        direction /= np.linalg.norm(direction)

        for _ in range(1, num_points):
            curr_pos = trajectory[-1]
            # 根据更新间隔计算当前帧的位移量 (速度 * 时间)
            step_size = self._rnd_generator.uniform(speed_range[0], speed_range[1]) * self._update_interval
            
            # 80% 概率保持原方向，20% 概率微调方向
            if self._rnd_generator.random() > 0.8:
                new_dir = self._rnd_generator.standard_normal(3)
                direction = (direction + new_dir * 0.3)
                direction /= np.linalg.norm(direction)
                
            next_pos = curr_pos + direction * step_size
            
            # 边界与听者距离约束
            in_walls = np.all(next_pos > self._mixture_setup['src_pos_from_walls']) and \
                    np.all(next_pos < room_dim - self._mixture_setup['src_pos_from_walls'])
            far_listener = np.linalg.norm(next_pos - mic_pos_center) > self.params['src_pos_from_listener']
            
            if in_walls and far_listener:
                trajectory.append(next_pos)
            else:
                # 撞墙或离人太近，反向弹回
                direction = -direction
                trajectory.append(curr_pos)
                
        return np.array(trajectory)
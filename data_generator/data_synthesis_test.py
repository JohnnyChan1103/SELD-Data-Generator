import numpy as np
import scipy.io
import utils
import os
import mat73
import json
import pandas as pd
import soundfile
from tqdm.contrib.concurrent import process_map
import functools
import librosa
from sklearn.model_selection import train_test_split


def cart2sph(xyz):
    return_list = False
    if len(np.shape(xyz)) == 2:
        return_list = True
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]
    else:
        x = xyz[0]
        y = xyz[1]
        z = xyz[2]
    
    azimuth = np.arctan2(y, x) * 180. / np.pi
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2)) * 180. / np.pi
    if return_list:
        return np.stack((azimuth,elevation),axis=0)
    else:
        return np.array([azimuth, elevation])

class MetadataSynthesizer(object):
    def __init__(self, db_config, params):
        self.max_samples_per_cls = params['nb_events_per_classes']
        self.params = params
        self._db_config = db_config
        self._mixture_path = params['mixturepath']
        self._classnames = db_config._classes
        params['target_classes'] = params['target_classes'] \
            if params['target_classes'] != 'all' else list(range(len(self._classnames)))
        
        self._active_classes = params['target_classes']
        self._nb_active_classes = len(self._active_classes)
        self._class_mobility = [2] * len(self._classnames) # MOVING or not

        self.ontology = json.load(open(params['ontology_path']))
        self.class_labels_indices = {}
        for item in self.ontology:
            self.class_labels_indices[item['id']] = item['name']
        
        self._mixture_setup = {}
        self._mixture_setup['rooms'] = params['rooms']
        self._mixture_setup['classnames'] = []
        for cl in self._classnames:
            self._mixture_setup['classnames'].append(cl)
        self._mixture_setup['nb_classes'] = len(self._active_classes)
        self._mixture_setup['fs_mix'] = 24000 #fs of RIRs
        self._mixture_setup['mixture_duration'] = params['mixture_duration']
        self._nb_mixtures = params['nb_mixtures']
        self._mixture_setup['total_duration'] = self._nb_mixtures * self._mixture_setup['mixture_duration']
        self._mixture_setup['speed_set'] =  [10., 20., 40.]
        self._mixture_setup['snr_set'] = np.arange(*params['snr_set'])
        self._mixture_setup['time_idx_100ms'] = np.arange(0.,self._mixture_setup['mixture_duration'],0.1)
        self._mixture_setup['nOverlap'] = params['max_polyphony_target']
        self._nb_frames = len(self._mixture_setup['time_idx_100ms'])
        self._rnd_generator = np.random.default_rng(seed=params['seed']+params['max_polyphony_target'])
        
        self._rirpath = params['rirpath']
        self._rirdata = scipy.io.loadmat(self._rirpath / 'rirdata.mat')
        self._rirdata = self._rirdata['rirdata']['room'][0][0]
        self._nb_classes = len(self._classnames)
        self._nb_speeds = len(self._mixture_setup['speed_set'])
        self._nb_snrs = len(self._mixture_setup['snr_set'])
        self._total_event_time_per_layer = params['event_time_per_layer']
        self._total_silence_time_per_layer = self._mixture_setup['mixture_duration'] \
            - self._total_event_time_per_layer
        self._min_gap_len = 0.5 # in seconds, minimum length of gaps between samples
        self._trim_threshold = 3. #in seconds, minimum length under which a trimmed event at end is discarded
        self._move_threshold = 3. #in seconds, minimum length over which events can be moving

        self._mixtures = []
        self._metadata = []
        self._srir_setup = []

    def create_mixtures(self):
        rirdata2room_idx = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 6, 9: 7, 10: 8} # room numbers in the rirdata array
            
        samplelist = {}
        rooms = np.array(self._mixture_setup['rooms'])
        rooms = rooms[rooms>0]
        nb_rooms = len(rooms)
        
        idx_active1 = np.array([])
        idx_active2 = np.array([])
        path_dict = dict()
        for na in range(self._nb_active_classes):
            idx_active1 = np.append(
                idx_active1, 
                np.nonzero(self._db_config._sample_list['class'] == self._active_classes[na]))
        
        for idx, path in enumerate(self._db_config._sample_list['audiofile']):
            cls_idx = self._db_config._sample_list['class'][idx]
            cls = self._classnames[cls_idx]
            if cls not in path_dict.keys():
                path_dict[cls] = []
            path_dict[cls].append([idx, path])
        
        if self.max_samples_per_cls > 0 and 'STARSS23' in self.params['db_name']:
            for cls in path_dict.keys():
                cls_sampleperm = self._rnd_generator.permutation(len(path_dict[cls]))[:self.max_samples_per_cls]
                path_dict[cls] = np.array(path_dict[cls])[cls_sampleperm]
        
        path_dict_selected = dict()
        rnd = np.random.default_rng(seed=2024)
        for cls in path_dict.keys():
            cls_wise_segments = np.array(path_dict[cls])
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

        samplelist['class'] = self._db_config._sample_list['class'][idx_active]
        samplelist['audiofile'] = self._db_config._sample_list['audiofile'][idx_active]
        samplelist['duration'] = self._db_config._sample_list['duration'][idx_active]
        samplelist['mid'] = self._db_config._sample_list['mid'][idx_active]
        samplelist['onoffset'] = self._db_config._sample_list['onoffset'][idx_active]
        samplelist['timestamps'] = self._db_config._sample_list['timestamps'][idx_active]

        # print all classes
        cls_indices_path = self.params['mixturepath'] / 'cls_indices.tsv'
        cls_indices = np.unique(samplelist['class'])
        f = open(cls_indices_path, 'w')
        num_clips, sum_duration = 0, 0
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

        # print all mids
        mids_path = self.params['mixturepath'] / 'mids.tsv'
        mids = set()
        for _mids in samplelist['mid']:
            # if isinstance(_mids, str):
            mids.update(_mids.split(','))
        # mids = np.unique(mids)
        f = open(mids_path, 'w')
        for mid in mids:
            label = self.class_labels_indices[mid]
            f.write('{}\t{}\n'.format(mid, label))
        f.close()

        # statistics of the dataset
        stats_path = self.params['mixturepath'] / 'stats.txt'
        with open(stats_path, 'w') as f:
            f.write('Dataset: {}, number of clips: {}, total duration: {:.1f} hours.\n'.format(
                self.params['dataset_type'], num_clips, sum_duration/3600))
                
        if self.params['dataset_type'] == 'all' and 'STARSS23' in self.params['db_name']:
            cls_indices = np.unique(samplelist['class'])
            for cls_idx in cls_indices:
                indices = np.where(samplelist['class'] == cls_idx)[0]
                num_tile = self.max_samples_per_cls // len(indices) - 1
                if num_tile > 0:
                    samplelist['class'] = np.append(samplelist['class'], np.tile(samplelist['class'][indices], num_tile))
                    samplelist['mid'] = np.append(samplelist['mid'], np.tile(samplelist['mid'][indices], num_tile))
                    samplelist['audiofile'] = np.append(samplelist['audiofile'], np.tile(samplelist['audiofile'][indices], num_tile))
                    samplelist['duration'] = np.append(samplelist['duration'], np.tile(samplelist['duration'][indices], num_tile))
                    samplelist['onoffset'] = np.append(
                        samplelist['onoffset'], 
                        np.tile(samplelist['onoffset'][indices], (num_tile, 1)),
                        axis=0)
                    samplelist['timestamps'] = np.array(
                        list(samplelist['timestamps']) + list(samplelist['timestamps'][indices]) * num_tile,
                        dtype=object)
                    print('Expand class {} to {} samples.'.format(cls_idx, len(indices)*(num_tile+1)))

        # shuffle randomly the samples in the target list to avoid samples of the same class coming consecutively
        nb_samples = len(samplelist['duration'])
        sampleperm = self._rnd_generator.permutation(nb_samples)
        samplelist['class'] = samplelist['class'][sampleperm]
        samplelist['audiofile'] = samplelist['audiofile'][sampleperm]
        samplelist['duration'] = samplelist['duration'][sampleperm]
        samplelist['onoffset'] = samplelist['onoffset'][sampleperm]
        samplelist['mid'] = samplelist['mid'][sampleperm]
        samplelist['timestamps'] = samplelist['timestamps'][sampleperm]
        room_mixtures = []
        
        sample_counter = 0
        for nr in range(nb_rooms):
            per_room_mixtures = {'mixture': []}
            per_room_mixtures['roomidx'] = rooms
            nroom = rooms[nr]     
            n_traj = np.shape(self._rirdata[rirdata2room_idx[nroom]][0][2])[0] #number of trajectories
            traj_doas = []
            
            for ntraj in range(n_traj):
                n_rirs = np.sum(self._rirdata[rirdata2room_idx[nroom]][0][3][ntraj,:])
                n_heights = np.sum(self._rirdata[rirdata2room_idx[nroom]][0][3][ntraj,:]>0)
                all_doas = np.zeros((n_rirs, 3))
                n_rirs_accum = 0
                flip = 0
                
                for nheight in range(n_heights):
                    n_rirs_nh = self._rirdata[rirdata2room_idx[nroom]][0][3][ntraj,nheight]
                    doa_xyz = self._rirdata[rirdata2room_idx[nroom]][0][2][ntraj,nheight][0]
                    #   stack all doas of trajectory together
                    #   flip the direction of each second height, so that a
                    #   movement can jump from the lower to the higher smoothly and
                    #   continue moving the opposite direction
                    if flip:
                        nb_doas = np.shape(doa_xyz)[0]
                        all_doas[n_rirs_accum + np.arange(n_rirs_nh), :] = doa_xyz[np.flip(np.arange(nb_doas)), :]
                    else:
                        all_doas[n_rirs_accum + np.arange(n_rirs_nh), :] = doa_xyz
                    
                    n_rirs_accum += n_rirs_nh
                    flip = not flip
                    
                traj_doas.append(all_doas)
        
            # start layering the mixtures for the specific room
            # sample_counter = self._rnd_generator.integers(0, nb_samples)
            nb_mixtures_per_room = int(np.round(self._nb_mixtures / float(nb_rooms)))
            
            for nmix in range(nb_mixtures_per_room):

                event_counter = 0
                nth_mixture = {'files': np.array([]), 'class': np.array([]), 'event_onoffsets': np.array([]), 'audio_onset': np.array([]),
                               'sample_onoffsets': np.array([]), 'trajectory': np.array([]), 'isMoving': np.array([]), 
                               'isFlippedMoving': np.array([]),'speed': np.array([]), 'rirs': [], 'timestamps': [],
                               'doa_azel': np.array([],dtype=object), 'mid': np.array([])}
                nth_mixture['room'] = nroom
                nth_mixture['snr'] = self._mixture_setup['snr_set'][self._rnd_generator.integers(0,self._nb_snrs)]
                
                for layer in range(self._mixture_setup['nOverlap']):
                    print(f'Create Mixtures: Room {nroom}, mixture {nmix+1}, layer {layer}')                     
                    #zero this flag (explained later)
                    TRIMMED_SAMPLE_AT_END = 0
                    
                    #fetch event samples till they add up to the target event time per layer
                    event_time_in_layer = 0
                    event_idx_in_layer = []
                    
                    while event_time_in_layer < self._total_event_time_per_layer:
                        #get event duration
                        ev_duration = np.ceil(samplelist['duration'][sample_counter]*10.)/10.
                        event_time_in_layer += ev_duration
                        event_idx_in_layer.append(sample_counter)
                        
                        event_counter += 1
                        sample_counter += 1

                        if sample_counter == nb_samples:
                            sample_counter = 0

                    # the last sample is going to be trimmed to fit the desired
                    # time, or omit if it is less than X sec, and occurs later than that time
                    trimmed_event_length = self._total_event_time_per_layer - (event_time_in_layer - ev_duration)
                    #Temporary workaround - for some reason for interference classes the dict is packed with an additional dimension - check it
                    # ons = samplelist['onoffset'][event_idx_in_layer[-1]][0]
                    
                    if (trimmed_event_length > self._trim_threshold) and (trimmed_event_length > np.floor(ev_duration*10.)/10.):
                        TRIMMED_SAMPLE_AT_END = 1
                    else:
                        if len(event_idx_in_layer) == 1:
                            print(sample_counter)
                            print(samplelist['duration'][event_idx_in_layer[0]])
                            print(samplelist['audiofile'][event_idx_in_layer[0]])
                            raise ValueError("STOP, we will get stuck here forever")

                        #remove from sample list
                        event_idx_in_layer = event_idx_in_layer[:-1]
                        # reduce sample count and events-in-recording by 1
                        event_counter -= 1
                        if sample_counter != 0:
                            sample_counter -= 1
                        else:
                            # move sample counter to end of the list to re-use sample
                            sample_counter = nb_samples-1
                        
                    nb_samples_in_layer = len(event_idx_in_layer)
                    # split silences between events
                    # randomize N split points uniformly for N events (in
                    # steps of 100msec)
                    mult_silence = np.round(self._total_silence_time_per_layer*10.)
                    
                    mult_min_gap_len = np.round(self._min_gap_len*10.)
                    if nb_samples_in_layer > 1:
                        
                        silence_splits = np.sort(
                            self._rnd_generator.integers(1, mult_silence, nb_samples_in_layer-1))
                        #force gaps smaller then _min_gap_len to it
                        gaps = np.diff(np.concatenate(([0],silence_splits,[mult_silence])))
                        smallgaps_idx = np.argwhere(gaps[:(nb_samples_in_layer-1)] < mult_min_gap_len)
                        while np.any(smallgaps_idx):
                            temp = np.concatenate(([0], silence_splits))
                            silence_splits[smallgaps_idx] = temp[smallgaps_idx] + mult_min_gap_len
                            gaps = np.diff(np.concatenate(([0],silence_splits,[mult_silence])))
                            smallgaps_idx = np.argwhere(gaps[:(nb_samples_in_layer-1)] < mult_min_gap_len)
                        if np.any(gaps < mult_min_gap_len):
                            min_idx = np.argwhere(gaps < mult_min_gap_len)
                            gaps[min_idx] = mult_min_gap_len
                        # if gaps[nb_samples_in_layer-1] < mult_min_gap_len:
                        #     gaps[nb_samples_in_layer-1] = mult_min_gap_len
                        
                    else:
                        gaps = np.array([mult_silence])
                    while np.sum(gaps) > self._total_silence_time_per_layer*10.:
                        silence_diff = np.sum(gaps) - self._total_silence_time_per_layer*10.
                        picked_gaps = np.argwhere(gaps > np.mean(gaps))
                        eq_subtract = silence_diff / len(picked_gaps)
                        picked_gaps = np.argwhere((gaps - eq_subtract) > mult_min_gap_len)
                        gaps[picked_gaps] -= eq_subtract
                        
                    # distribute events in timeline
                    time_idx = 0
                    for nl in range(nb_samples_in_layer):
                        #print('Sample {} in layer {}'.format(nl, layer))
                        # event offset (quantized to 100ms)
                        gap_nl = gaps[nl]
                        time_idx += gap_nl
                        event_nl = event_idx_in_layer[nl]
                        event_duration_nl = np.ceil(samplelist['duration'][event_nl]*10.)
                        event_class_nl = int(samplelist['class'][event_nl])
                        audiofile = samplelist['audiofile'][event_nl]
                        onoffsets = samplelist['onoffset'][event_nl]
                        timestamps = samplelist['timestamps'][event_nl]

                        # sample_onoffsets = np.floor(onoffsets*10.)/10.
                        sample_onoffsets = np.array(timestamps) - timestamps[0]
                        if nl == nb_samples_in_layer - 1 and TRIMMED_SAMPLE_AT_END:
                            event_duration_nl = len(self._mixture_setup['time_idx_100ms']) - time_idx - 1
                            if sample_onoffsets[1] - sample_onoffsets[0] > event_duration_nl/10.:
                                sample_onoffsets[1] = sample_onoffsets[0] + event_duration_nl/10.

                        # trajectory
                        ev_traj = self._rnd_generator.integers(0, n_traj)
                        nRirs = np.sum(self._rirdata[rirdata2room_idx[nroom]][0][3][ev_traj,:])
                        
                        #if event is less than move_threshold long, make it static by default
                        if event_duration_nl <= self._move_threshold*10:
                            is_moving = 0 
                        else:
                            if self._class_mobility[event_class_nl] == 2:
                                # randomly moving or static
                                is_moving = self._rnd_generator.integers(0,2)
                            else:
                                # only static or moving depending on class
                                is_moving = self._class_mobility[event_class_nl]

                        if is_moving:
                            ev_nspeed = self._rnd_generator.integers(0,self._nb_speeds)
                            ev_speed = self._mixture_setup['speed_set'][ev_nspeed]
                            # check if with the current speed there are enough
                            # RIRs in the trajectory to move through the full
                            # duration of the event, otherwise, lower speed
                            while len(np.arange(0,nRirs,ev_speed/10)) <= event_duration_nl:
                                ev_nspeed = ev_nspeed-1
                                if ev_nspeed == -1:
                                    break

                                ev_speed = self._mixture_setup['speed_set'][ev_nspeed]
                            
                            is_flipped_moving = self._rnd_generator.integers(0,2)
                            event_span_nl = event_duration_nl * ev_speed / 10.

                            if is_flipped_moving:
                                # sample length is shorter than all the RIRs
                                # in the moving trajectory
                                if ev_nspeed+1:
                                    end_idx = event_span_nl + self._rnd_generator.integers(0, nRirs-event_span_nl+1)
                                    start_idx = end_idx - event_span_nl
                                    riridx = start_idx + np.arange(0, event_span_nl, dtype=int)
                                    riridx = riridx[np.arange(0,len(riridx),ev_speed/10,dtype=int)] #pick every nth RIR based on speed
                                    riridx = np.flip(riridx)
                                else:
                                    riridx = np.arange(event_span_nl,0,-1)-1
                                    riridx = riridx - (event_span_nl-nRirs)
                                    riridx = riridx[np.arange(0, len(riridx), ev_speed/10, dtype=int)]
                                    riridx[riridx<0] = 0
                            else:
                                if ev_nspeed+1:
                                    start_idx = self._rnd_generator.integers(0, nRirs-event_span_nl+1)
                                    riridx = start_idx + np.arange(0,event_span_nl,dtype=int) - 1
                                    riridx = riridx[np.arange(0,len(riridx),ev_speed/10,dtype=int)]
                                else:
                                    riridx = np.arange(0,event_span_nl)
                                    riridx = riridx[np.arange(0,len(riridx),ev_speed/10,dtype=int)]
                                    riridx[riridx>nRirs-1] = nRirs-1
                        else:
                            is_flipped_moving = 0
                            ev_speed = 0
                            riridx = np.array([self._rnd_generator.integers(0,nRirs)])
                        riridx = riridx.astype('int')

                        if nl == 0 and layer == 0:
                            nth_mixture['event_onoffsets'] = np.array([[time_idx/10., (time_idx+event_duration_nl)/10.]])
                            nth_mixture['doa_azel'] = [cart2sph(traj_doas[ev_traj][riridx,:])]
                            nth_mixture['sample_onoffsets'] = [sample_onoffsets]
                        else:
                            nth_mixture['event_onoffsets'] = np.vstack(
                                (nth_mixture['event_onoffsets'], 
                                 np.array([time_idx/10., (time_idx+event_duration_nl)/10.])))
                            nth_mixture['doa_azel'].append(cart2sph(traj_doas[ev_traj][riridx,:]))
                            nth_mixture['sample_onoffsets'].append(sample_onoffsets)
                                        
                        nth_mixture['files'] = np.append(nth_mixture['files'], samplelist['audiofile'][event_nl])
                        nth_mixture['audio_onset'] = np.append(nth_mixture['audio_onset'], onoffsets[0])
                        nth_mixture['mid'] = np.append(nth_mixture['mid'], samplelist['mid'][event_nl])
                        nth_mixture['class'] = np.append(nth_mixture['class'], 
                                                         samplelist['class'][event_nl])
                                                        #  self._class2activeClassmap[int(samplelist['class'][event_nl])])
                        nth_mixture['trajectory'] = np.append(nth_mixture['trajectory'], ev_traj)
                        nth_mixture['isMoving'] = np.append(nth_mixture['isMoving'], is_moving)
                        nth_mixture['isFlippedMoving'] = np.append(nth_mixture['isFlippedMoving'], is_flipped_moving)
                        nth_mixture['speed'] = np.append(nth_mixture['speed'], ev_speed)
                        nth_mixture['timestamps'].append(samplelist['timestamps'][event_nl])
                        nth_mixture['rirs'].append(riridx)
                        
                        time_idx += event_duration_nl
                    
                    # sort overlapped events by temporal appearance
                sort_idx = np.argsort(nth_mixture['event_onoffsets'][:,0])
                nth_mixture['files'] = nth_mixture['files'][sort_idx]
                nth_mixture['audio_onset'] = nth_mixture['audio_onset'][sort_idx]
                nth_mixture['mid'] = nth_mixture['mid'][sort_idx]
                nth_mixture['class'] = nth_mixture['class'][sort_idx]
                nth_mixture['event_onoffsets'] = nth_mixture['event_onoffsets'][sort_idx]
                #nth_mixture['sample_onoffsets'] = nth_mixture['sample_onoffsets'][sort_idx]
                nth_mixture['trajectory'] = nth_mixture['trajectory'][sort_idx]
                nth_mixture['isMoving'] = nth_mixture['isMoving'][sort_idx]
                nth_mixture['isFlippedMoving'] = nth_mixture['isFlippedMoving'][sort_idx]
                nth_mixture['speed'] = nth_mixture['speed'][sort_idx]
                nth_mixture['rirs'] = np.array(nth_mixture['rirs'],dtype=object)
                nth_mixture['rirs'] = nth_mixture['rirs'][sort_idx]
                new_doas = np.zeros(len(sort_idx),dtype=object)
                new_sample_onoffsets = np.zeros(len(sort_idx),dtype=object)
                upd_idx = 0
                for idx in sort_idx:
                    new_doas[upd_idx] = nth_mixture['doa_azel'][idx].T
                    new_sample_onoffsets[upd_idx] = nth_mixture['sample_onoffsets'][idx]
                    upd_idx += 1
                nth_mixture['doa_azel'] = new_doas
                nth_mixture['sample_onoffsets'] = new_sample_onoffsets
            
                #accumulate mixtures for each room
                per_room_mixtures['mixture'].append(nth_mixture)
            #accumulate rooms
            room_mixtures.append(per_room_mixtures)
        #accumulate mixtures per fold
        self._mixtures = room_mixtures

    
    def prepare_metadata_and_stats(self):
        print('Calculate statistics and prepate metadata')
        stats = {}
        stats['nFrames_total'] = self._nb_mixtures * self._nb_frames
        stats['class_multi_instance'] = np.zeros(self._nb_classes)
        stats['class_instances'] = np.zeros(self._nb_classes)
        stats['class_nEvents'] = np.zeros(self._nb_classes)
        stats['class_presence'] = np.zeros(self._nb_classes)
        
        stats['polyphony'] = np.zeros(self._mixture_setup['nOverlap']+1)
        stats['event_presence'] = 0
        stats['nEvents_total'] = 0
        stats['nEvents_static'] = 0
        stats['nEvents_moving'] = 0
        
        rooms = self._mixtures[0]['roomidx']
        nb_rooms = len(rooms)
        room_mixtures=[]
        for nr in range(nb_rooms):
            nb_mixtures = len(self._mixtures[nr]['mixture'])
            per_room_mixtures = []
            for nmix in range(nb_mixtures):
                mixture = {'classid': np.array([]), 'trackid': np.array([]), 
                           'eventtimetracks': np.array([]), 'mid': np.array([]),
                           'eventdoatimetracks': np.array([])}
                mixture_nm = self._mixtures[nr]['mixture'][nmix]
                event_classes = mixture_nm['class']
                event_mid = mixture_nm['mid']
                event_states = mixture_nm['isMoving']
                
                #idx of events and interferers
                nb_events = len(event_classes)
                nb_events_moving = np.sum(event_states)
                stats['nEvents_total'] += nb_events
                stats['nEvents_static'] += nb_events - nb_events_moving
                stats['nEvents_moving'] += nb_events_moving

                # number of events per class
                for nc in range(self._mixture_setup['nb_classes']):
                    cls_idx = self._active_classes[nc]
                    nb_class_events = np.sum(event_classes == cls_idx)
                    stats['class_nEvents'][cls_idx] += nb_class_events
                
                # store a timeline for each event
                eventtimetracks = np.zeros((self._nb_frames, nb_events))
                eventdoatimetracks = np.nan*np.ones((self._nb_frames, 2, nb_events))

                #prepare metadata for synthesis
                for nev in range(nb_events):
                    print('Prepare metadata and stats: Room {}, mixture {}, event {}'.format(nr+1, nmix+1, nev+1))
                    event_onoffset = mixture_nm['event_onoffsets'][nev,:]*10
                    doa_azel = np.round(mixture_nm['doa_azel'][nev])
                    #zero the activity according to perceptual onsets/offsets
                    audiofile = mixture_nm['files'][nev]
                    sample_onoffsets = mixture_nm['sample_onoffsets'][nev]
                    ev_idx = np.arange(event_onoffset[0], event_onoffset[1]+0.1, dtype=int)
                    activity_mask = np.zeros(len(ev_idx),dtype=int)
                    sample_shape = np.shape(sample_onoffsets)

                    activity_mask = np.ones(len(ev_idx),dtype=int)                    
                    if len(activity_mask) > len(ev_idx):
                        activity_mask = activity_mask[0:len(ev_idx)]

                    if np.shape(doa_azel)[0] == 1:
                        # static event
                        try:
                            eventtimetracks[ev_idx, nev] = activity_mask
                            eventdoatimetracks[ev_idx[activity_mask.astype(bool)],0,nev] = np.ones(np.sum(activity_mask==1))*doa_azel[0,0]
                            eventdoatimetracks[ev_idx[activity_mask.astype(bool)],1,nev] = np.ones(np.sum(activity_mask==1))*doa_azel[0,1]
                        except IndexError:
                                excess_idx = len(np.argwhere(ev_idx >= self._nb_frames))
                                ev_idx = ev_idx[:-excess_idx]
                                if len(activity_mask) > len(ev_idx):
                                    activity_mask = activity_mask[0:len(ev_idx)]
                                eventtimetracks[ev_idx, nev] = activity_mask
                                eventdoatimetracks[ev_idx[activity_mask.astype(bool)],0,nev] = np.ones(np.sum(activity_mask==1))*doa_azel[0,0]
                                eventdoatimetracks[ev_idx[activity_mask.astype(bool)],1,nev] = np.ones(np.sum(activity_mask==1))*doa_azel[0,1]

                    else:
                        # moving event
                        nb_doas = np.shape(doa_azel)[0]
                        ev_idx = ev_idx[:nb_doas]
                        activity_mask = activity_mask[:nb_doas]
                        try:
                            eventtimetracks[ev_idx,nev] = activity_mask
                            eventdoatimetracks[ev_idx[activity_mask.astype(bool)],:,nev] = doa_azel[activity_mask.astype(bool),:]
                        except IndexError:
                            excess_idx = len(np.argwhere(ev_idx >= self._nb_frames))
                            ev_idx = ev_idx[:-excess_idx]
                            if len(activity_mask) > len(ev_idx):
                                activity_mask = activity_mask[0:len(ev_idx)]
                            eventtimetracks[ev_idx,nev] = activity_mask
                            eventdoatimetracks[ev_idx[activity_mask.astype(bool)],:,nev] = doa_azel[activity_mask.astype(bool),:]

                mixture['classid'] = event_classes
                mixture['mid'] = event_mid
                mixture['trackid'] = np.arange(0, nb_events)
                mixture['eventtimetracks'] = eventtimetracks
                mixture['eventdoatimetracks'] = eventdoatimetracks
                
                for nf in range(self._nb_frames):
                    # find active events
                    active_events = np.argwhere(eventtimetracks[nf,:] > 0)
                    # find the classes of the active events
                    active_classes = event_classes[active_events]
                    
                    if not active_classes.ndim and active_classes.size:
                        # add to zero polyphony
                        stats['polyphony'][0] += 1
                    else:
                        # add to general event presence
                        stats['event_presence'] += 1
                        # number of simultaneous events
                        nb_active = len(active_events)

                        # add to respective polyphony
                        try:
                            stats['polyphony'][nb_active] += 1
                        except IndexError:
                            pass #TODO: this is a workaround for less than 1% border cases, needs to be fixed although not very relevant
                        
                        # presence, instances and multi-instance for each class
                        
                        for nc in range(self._mixture_setup['nb_classes']):
                            cls_idx = self._active_classes[nc]
                            nb_instances = np.sum(active_classes == cls_idx)
                            if nb_instances > 0:
                                stats['class_presence'][cls_idx] += 1
                            if nb_instances > 1:
                                stats['class_multi_instance'][cls_idx] += 1
                            stats['class_instances'][cls_idx] += nb_instances
                per_room_mixtures.append(mixture)
            room_mixtures.append(per_room_mixtures)
            self._metadata = room_mixtures
         
        # compute average polyphony
        weighted_polyphony_sum = 0
        for nn in range(self._mixture_setup['nOverlap']):
            weighted_polyphony_sum += nn * stats['polyphony'][nn+1]
        
        stats['avg_polyphony'] = weighted_polyphony_sum / stats['event_presence']
        
        #event percentages
        stats['class_event_pc'] = np.round(stats['class_nEvents']*1000./stats['nEvents_total'])/10.
        stats['event_presence_pc'] = np.round(stats['event_presence']*1000./stats['nFrames_total'])/10.
        stats['class_presence_pc'] = np.round(stats['class_presence']*1000./stats['nFrames_total'])/10.
        # percentage of frames with same-class instances
        stats['multi_class_pc'] = np.round(np.sum(stats['class_multi_instance']*1000./stats['nFrames_total']))/10.

    
    def write_metadata(self):
        metadata_path = self._mixture_path / 'metadata'
        if not os.path.isdir(metadata_path):
            os.makedirs(metadata_path)
        
        nb_rooms = len(self._metadata)
        for nr in range(nb_rooms):
            nb_mixtures = len(self._metadata[nr])
            for nmix in range(nb_mixtures):
                print('Write Metadata: Room {} Mixture {}'.format(nr, nmix))
                metadata_nm = self._metadata[nr][nmix]
                
                # write to filename, omitting non-active frames
                mixture_filename = 'fold1_room{}_mix{:03}.csv'.format(nr+1, nmix+1)
                file_id = open(metadata_path / mixture_filename, 'w', newline="")
                # metadata_writer = csv.writer(file_id, delimiter=',',quoting=csv.QUOTE_NONE)
                metadata_writer = open(metadata_path / mixture_filename, 'w')
                for nf in range(self._nb_frames):
                    # find active events
                    active_events = np.argwhere(metadata_nm['eventtimetracks'][nf, :]>0)
                    nb_active = len(active_events)
                    
                    if nb_active > 0:
                        # find the classes of active events
                        active_classes = metadata_nm['classid'][active_events]
                        active_tracks = metadata_nm['trackid'][active_events]
                        active_mids = metadata_nm['mid'][active_events]
                        
                        # write to file
                        for na in range(nb_active):
                            classidx = int(active_classes[na][0]) # additional zero index since it's packed in an array
                            classidx = self._active_classes.index(classidx) # map to the original class index
                            trackidx = int(active_tracks[na][0])
                            mid = active_mids[na][0]
                            
                            azim = int(metadata_nm['eventdoatimetracks'][nf,0,active_events][na][0])
                            elev = int(metadata_nm['eventdoatimetracks'][nf,1,active_events][na][0])
                            # metadata_writer.writerow([nf,classidx,trackidx,azim,elev,0,'\"'+mid+'\"'])
                            metadata_writer.write(f'{nf},{classidx},{trackidx},{azim},{elev},0,\"{mid}\"\n')
                file_id.close()
                            

class AudioSynthesizer(object):
    def __init__(self, params, db_config, rirdata, mixtures, mixture_setup):
        self._mixtures = mixtures
        self._rirpath = params['rirpath']
        self._db_path = params['db_path']
        self._outpath = params['mixturepath']
        self._rirdata = rirdata
        self._nb_rooms = len(self._rirdata)
        self._room_names = []
        for nr in range(self._nb_rooms):
            self._room_names.append(self._rirdata[nr][0][0][0])
        self._classnames = mixture_setup['classnames']
        self._fs_mix = mixture_setup['fs_mix']
        self._t_mix = mixture_setup['mixture_duration']
        self._l_mix = int(np.round(self._fs_mix * self._t_mix))
        self._time_idx100 = np.arange(0., self._t_mix, 0.1)
        self._stft_winsize_moving = 0.1*self._fs_mix//2
        self._apply_event_gains = True
        self.params = params
        self._rnd_generator = np.random.default_rng(seed=params['seed'])
        self._audio_format = None
        if self._apply_event_gains:
            # self._class_gains = db_config._class_gains
            self._class_gains = db_config._sample_list['energy_quartile']
            # self._class_gains = [[0.1, 0.316, 1, 3.16, 10]] * db_config._nb_classes
        
        self.ambience = None
        self._noisepath = params['noisepath']
        self._rooms_paths = ['01_bomb_center', '02_gym', '03_pb132_paatalo_classroom2',
                             '04_pc226_paatalo_office', '05_sa203_sahkotalo_lecturehall',
                             '06_sc203_sahkotalo_classroom2', '07_se201_sahkotalo_classroom',
                             '08_se203_sahkotalo_classroom', '09_tb103_tietotalo_lecturehall',
                             '10_tc352_tietotalo_meetingroom']
        self._mic_format = None      
    
        
    def synthesize_mixtures(self, audio_format='foa'):
        rirdata2room_idx = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 6, 9: 7, 10: 8} # room numbers in the rirdata array
        # create path if doesn't exist

        os.makedirs(self._outpath/audio_format, exist_ok=True)
        os.makedirs(self._outpath/'sum', exist_ok=True)
        self._audio_format = audio_format
        if audio_format == 'foa':
            self._mic_format = 'foa_sn3d'
        elif audio_format == 'mic':
            self._mic_format = 'tetra'

        rooms = self._mixtures[0]['roomidx']
        nb_rooms = len(rooms)
        for nr in range(nb_rooms):

            nroom = rooms[nr]
            nb_mixtures = len(self._mixtures[nr]['mixture'])
            print('Loading RIRs for room {}'.format(nroom))
            
            room_idx = rirdata2room_idx[nroom]
            if nroom > 9:
                struct_name = 'rirs_{}_{}'.format(nroom,self._room_names[room_idx])
            else:
                struct_name = 'rirs_0{}_{}'.format(nroom,self._room_names[room_idx])
            path = self._rirpath / (struct_name + '.mat')
            rirs = mat73.loadmat(path)
            rirs = rirs['rirs'][self._audio_format]
            # stack all the RIRs for all heights to make one large trajectory
            print('Stacking same trajectory RIRs')
            lRir = len(rirs[0][0])
            nCh = len(rirs[0][0][0])
            
            n_traj = np.shape(self._rirdata[room_idx][0][2])[0]
            n_rirs_max = np.max(np.sum(self._rirdata[room_idx][0][3],axis=1))
            
            channel_rirs = np.zeros((lRir, nCh, n_rirs_max, n_traj))
            for ntraj in range(n_traj):
                nHeights = np.sum(self._rirdata[room_idx][0][3][ntraj,:]>0)
                
                nRirs_accum = 0
                
                # flip the direction of each second height, so that a
                # movement can jump from the lower to the higher smoothly and
                # continue moving the opposite direction
                flip = False
                for nheight in range(nHeights):
                    nRirs_nh = self._rirdata[room_idx][0][3][ntraj,nheight]
                    rir_l = len(rirs[ntraj][nheight][0,0,:])
                    if flip:
                        channel_rirs[:, :, nRirs_accum + np.arange(0,nRirs_nh),ntraj] = rirs[ntraj][nheight][:,:,np.arange(rir_l-1,-1,-1)]
                    else:
                        channel_rirs[:, :, nRirs_accum + np.arange(0,nRirs_nh),ntraj] = rirs[ntraj][nheight]
                        
                    nRirs_accum += nRirs_nh
                    flip = not flip
            
            del rirs #clear some memory

            if self.params['add_noise'] and not self.params['add_interf']:
                print('Loading ambient noise...')
                roompath = self._rooms_paths[nroom-1]
                ambience_path = self._noisepath / '{}/ambience_{}_24k_edited.wav'.format(roompath, self._mic_format)
                self.ambience, _ = librosa.load(ambience_path, sr=self._fs_mix, mono=False)
                print('Ambient noise is loaded.')

            process_map(
                functools.partial(
                    self.create_mixture,
                    nr, channel_rirs, nb_mixtures
                    ),
                    range(nb_mixtures),
                    max_workers=self.params['max_workers'],
                    chunksize=self.params['chunksize'],
            )   

                
    def create_mixture(self, nr, channel_rirs, nb_mixtures, nmix):
        print('{}: Writing room {}, mixture {}/{} '.format(self._audio_format, nr+1, nmix+1,nb_mixtures ))

        ### WRITE TARGETS EVENTS
        mixture_nm = self._mixtures[nr]['mixture'][nmix]
        try:
            nb_events = len(mixture_nm['class'])
        except TypeError:
            nb_events = 1
        
        mixsig = np.zeros((self._l_mix, 4))
        sumsig = np.zeros((self._l_mix))
        for nev in range(nb_events):
            classidx = int(mixture_nm['class'][nev])
            onoffset = mixture_nm['event_onoffsets'][nev,:]
            filename = mixture_nm['files'][nev]
            ntraj = int(mixture_nm['trajectory'][nev])
            audio_onset = mixture_nm['audio_onset'][nev]
            sample_active_time = onoffset[1] - onoffset[0]
            sample_onoffsets = mixture_nm['sample_onoffsets'][nev]
            # sample_active_time = sample_onoffsets[1] - sample_onoffsets[0]
            # load event audio and resample to match RIR sampling
            path = filename
            eventsig, _ = librosa.load(
                path, sr=self._fs_mix, mono=True, 
                offset=audio_onset, 
                duration=sample_active_time)
            
            #spatialize audio
            riridx = mixture_nm['rirs'][nev]  #if nb_events > 1 else mixture_nm['rirs']
            
            moving_condition = mixture_nm['isMoving'][nev]  #if nb_events > 1 else mixture_nm['isMoving']
            # if nb_events > 1 and not moving_condition:
            if moving_condition:
                nRirs_moving = len(riridx) if np.shape(riridx) else 1
                ir_times = self._time_idx100[np.arange(0,nRirs_moving)]
                riridx = np.array(riridx, dtype=np.int32)
            else:
                riridx = int(riridx[0]) if len(riridx)==1 else riridx.astype('int')
            
            rirs = channel_rirs[:, :, riridx, ntraj]
            
            if moving_condition:                
                mixeventsig = 481.6989*utils.ctf_ltv_direct(eventsig, rirs, ir_times, self._fs_mix, self._stft_winsize_moving) / float(len(eventsig))
            else:
                mixeventsig0 = scipy.signal.convolve(eventsig, np.squeeze(rirs[:, 0]), mode='full', method='fft')
                mixeventsig1 = scipy.signal.convolve(eventsig, np.squeeze(rirs[:, 1]), mode='full', method='fft')
                mixeventsig2 = scipy.signal.convolve(eventsig, np.squeeze(rirs[:, 2]), mode='full', method='fft')
                mixeventsig3 = scipy.signal.convolve(eventsig, np.squeeze(rirs[:, 3]), mode='full', method='fft')

                mixeventsig = np.stack((mixeventsig0,mixeventsig1,mixeventsig2,mixeventsig3),axis=1)
            if self._apply_event_gains:
                # apply random gain to each event based on class gain, distribution given externally
                K=1000
                rand_energies_per_spec = utils.sample_from_quartiles(K, self._class_gains[classidx])
                # intr_quart_energies_per_sec = rand_energies_per_spec[K + np.arange(3*(K+1))]
                intr_quart_energies_per_sec = rand_energies_per_spec[K + np.arange(2*(K+1))]
                rand_energy_per_spec = intr_quart_energies_per_sec[np.random.randint(len(intr_quart_energies_per_sec))]
                target_energy = rand_energy_per_spec*sample_active_time
                if self._audio_format == 'mic':
                    event_omni_energy = np.sum(np.sum(mixeventsig,axis=1)**2)
                elif self._audio_format == 'foa':
                    event_omni_energy = np.sum(mixeventsig[:,0]**2)
                    
                norm_gain = np.sqrt(target_energy / (event_omni_energy + 10e-6))
                mixeventsig = norm_gain * mixeventsig
                eventsig = norm_gain * eventsig

            lMixeventsig = np.shape(mixeventsig)[0]
            if np.round(onoffset[0]*self._fs_mix) + lMixeventsig <= self._t_mix * self._fs_mix:
                mixsig[int(np.round(onoffset[0]*self._fs_mix)) + np.arange(0,lMixeventsig,dtype=int), :] += mixeventsig
            else:
                lMixeventsig_trunc = int(self._t_mix * self._fs_mix - int(np.round(onoffset[0]*self._fs_mix)))
                mixsig[int(np.round(onoffset[0]*self._fs_mix)) + np.arange(0,lMixeventsig_trunc,dtype=int), :] += mixeventsig[np.arange(0,lMixeventsig_trunc,dtype=int), :]
            
            lSumsig = len(eventsig)
            if np.round(onoffset[0]*self._fs_mix) + lSumsig <= self._t_mix * self._fs_mix:
                sumsig[int(np.round(onoffset[0]*self._fs_mix)) + np.arange(0,lSumsig,dtype=int)] += eventsig
            else:
                lSumsig_trunc = int(self._t_mix * self._fs_mix - int(np.round(onoffset[0]*self._fs_mix)))
                sumsig[int(np.round(onoffset[0]*self._fs_mix)) + np.arange(0,lSumsig_trunc,dtype=int)] += eventsig[:lSumsig_trunc]

        # normalize
        gnorm = 0.5/np.max(np.max(np.abs(mixsig)))
        mixsig = gnorm*mixsig

        if self.params['add_noise'] and not self.params['add_interf']:
            snr = mixture_nm['snr']
            lSig = np.shape(self.ambience)[1]
            start_idx = self._rnd_generator.integers(0, lSig - self._l_mix)
            ambience = self.ambience[:, start_idx:start_idx + self._l_mix].T
            target_omni_energy = np.sum(np.mean(mixsig,axis=1)**2) if self._audio_format == 'mic' else np.sum(mixsig[:, 0]**2)
            ambi_energy = np.sum(np.mean(ambience,axis=1)**2) if self._audio_format == 'mic' else np.sum(ambience[:, 0]**2)
            ambi_norm = np.sqrt(target_omni_energy * 10.**(-snr/10.) / ambi_energy)
            mixsig += ambi_norm * ambience

        mixture_filename = 'fold1_room{}_mix{:03}.flac'.format(nr+1, nmix+1)
        soundfile.write(self._outpath / self._audio_format / mixture_filename, 
                        mixsig, self._fs_mix)
        
        sumsig_path = self._outpath / 'sum' / mixture_filename
        if not os.path.isfile(sumsig_path):
            soundfile.write(sumsig_path, sumsig[:,None], self._fs_mix)


class AudioMixer(object):
    def __init__(self, params):
        self.params = params
        self._noisepath = params['noisepath']
        self.ambience = None
        self._mic_format = None
        self._rooms_paths = ['01_bomb_center', '02_gym', '03_pb132_paatalo_classroom2',
                             '04_pc226_paatalo_office', '05_sa203_sahkotalo_lecturehall',
                             '06_sc203_sahkotalo_classroom2', '07_se201_sahkotalo_classroom',
                             '08_se203_sahkotalo_classroom', '09_tb103_tietotalo_lecturehall',
                             '10_tc352_tietotalo_meetingroom']
        self._mixturepath_interf = params['mixturepath']
        self._mixturepath = params['mixturepath'].parent
        self._rnd_generator = np.random.default_rng(seed=params['seed'])
        self._fs_mix = 24000
        self._nb_mixtures = params['nb_mixtures']
        self._t_mix = params['mixture_duration']
        self._l_mix = int(np.round(self._fs_mix * self._t_mix))
        self.snr = params['snr_set']
        self.sir = params['sir_set'] # [2, 5]
    
    def mix_audio(self, audio_format='foa'):
        self._audio_format = audio_format
        if audio_format == 'foa':
            self._mic_format = 'foa_sn3d'
        elif audio_format == 'mic':
            self._mic_format = 'tetra'
        
        rooms = self.params['rooms'][0]
        nb_rooms = len(rooms)
        nb_mixtures_per_room = self._nb_mixtures // nb_rooms
        for nr in range(nb_rooms):
            nroom = rooms[nr]
            roompath = self._rooms_paths[nroom-1]
            if self.params['add_noise']:
                print('Loading ambient noise...')
                ambience_path = self._noisepath / '{}/ambience_{}_24k_edited.wav'.format(roompath, self._mic_format)
                self.ambience, _ = librosa.load(ambience_path, sr=self._fs_mix, mono=False)
                print('Ambient noise is loaded.')
        
            process_map(
                functools.partial(
                    self.create_mixtures,
                    nr, nb_mixtures_per_room
                ),
                range(nb_mixtures_per_room),
                max_workers=self.params['max_workers'],
                chunksize=self.params['chunksize'],
            )

    def create_mixtures(self, nr, nb_mixtures, nmix):
        print('{}: Writing room {}, mixture {}/{} '.format(self._audio_format, nr+1, nmix+1, nb_mixtures))
        audiofile = 'fold1_room{}_mix{:03}.flac'.format(nr+1, nmix+1)

        target_sig, _ = soundfile.read(
            self._mixturepath / self._audio_format / audiofile)
        target_omni_energy = np.sum(np.mean(target_sig,axis=1)**2) \
            if self._audio_format == 'mic' else np.sum(target_sig[:,0]**2)
        interf_sig, _ = soundfile.read(
            self._mixturepath_interf / self._audio_format / audiofile)
        interf_omni_energy = np.sum(np.mean(interf_sig,axis=1)**2) \
            if self._audio_format == 'mic' else np.sum(interf_sig[:,0]**2)
        snr = self._rnd_generator.integers(self.snr[0], self.snr[1])
        sir = self._rnd_generator.integers(self.sir[0], self.sir[1])
        
        interf_norm = np.sqrt(target_omni_energy / (sir * interf_omni_energy))
        target_sig += interf_norm * interf_sig

        if self.params['add_noise']:
            lSig = np.shape(self.ambience)[1]
            start_idx = self._rnd_generator.integers(0, lSig - self._l_mix)
            ambience = self.ambience[:, start_idx:start_idx + self._l_mix].T
            ambi_energy = np.sum(np.mean(ambience,axis=1)**2) \
                if self._audio_format == 'mic' else np.sum(ambience[:,0]**2)
            ambi_norm = np.sqrt(target_omni_energy * 10.**(-snr/10.) / ambi_energy)
            target_sig += ambi_norm * ambience

        os.system(f'rm {self._mixturepath / self._audio_format / audiofile}')
        soundfile.write(self._mixturepath / self._audio_format / audiofile, 
                        target_sig, self._fs_mix)


        


from pathlib import Path
import warnings
import numpy as np


# Parameters used in the data generation process.
Database_dir = Path('database')

def get_params(arg=0):
    task_id = int(arg[1])
    params = dict(
        # path containing background noise recordings
        database_dir = Database_dir,
        mixturepath = Database_dir , # root path to the synthesized mixture files
        db_path = 'source_datasets/single_source_samples',
        materials_path = 'source_datasets/material_absorption',
        ontology_path =  'source_datasets/ontology.json',
        # noisepath = Database_dir / 'TAU-SRIR_DB/TAU-SNoise_DB', 
        min_samples_per_class = 30,
        metric_threshold = 10., # filter out the sound event class with lower F-scores
        mixture_duration = 60., # seconds
        start_delay = 10, # seconds
        audio_format = 'both', # 'foa' , 'mic' or 'both'
        db_name = 'seld', 
        seed = 2024, # fix the seed for reproducibility
        chunksize = 128,
        max_workers = 128,
        nb_mixtures = 10000, # number of mixtures 
        ################ Sound Event Parameters ################
        nb_events_per_classes = -1, # -1 means all events
        target_classes = 'all', # all classes are considered as target
        interf_classes = [], # no interference
        max_polyphony_target = 3,
        max_polyphony_interf = 1,
        ################ SRIR Parameters ################
        #### mic array parameters #####
        SH_order = 1, # spherical harmonic order
        array_type = 'rigid', # 'rigid' or 'open'
        SH_type = 'real', # shperical harmony type, 'real' or 'complex'
        radius = 0.042, # radius of the spherical array
        mic_pos = [[45,35],[-45,-35],[135,-35],[-135,35]],
        #### Room Parameters #####
        # [[lx_min, lx_max],[ly_min, ly_max],[lz_min, lz_max]]
        room_size_range = [[4., 20.], [4., 20.], [3., 10.]],
        # [value_min, value_max]
        temperature_range = [15, 35], # degree Celsius
        humidity_range = [0, 100],
        RT60_range = [0.2, 2], # in seconds. None for absorption of materials from database.
        mic_pos_range_percentage = [0.4, 0.6], # percentage of the room size
        src_pos_from_walls = 0.5,
        src_pos_from_listener = 1,
        method = 'hybrid', # 'hybrid' or 'ism'
        tools = 'pyroomacoustics', # 'gpuRIR' or 'pyroomacoustics' or 'smir'
        add_noise = False,
        snr_set = [6, 31], # dB
        add_interf = False,
        dataset_type = 'test', # NOTE: synthesizing training sets or test sets
    )

    if task_id == 1:
        ################################################################################
        #### Default for sound events from AudioSet or FSD50K (training set)
        ####   and SRIRs are computationally generated
        ####   (./data_generator/data_synthesis.py)
        ################################################################################
        params['db_name'] = 'FSD50K'
        params['dataset_type'] = 'train' # NOTE: synthesizing training sets or test sets
        params['nb_mixtures'] = 10
        params['max_polyphony_target'] = 1
        params['output_dir'] = 'seld_{}_{}_ov{}_{}'.format(
            params['db_name'], params['nb_mixtures'], 
            params['max_polyphony_target'], params['dataset_type'])
        params['mixturepath'] /= params['output_dir']
        params['audio_format'] = 'both'
        params['RT60_range'] = None # or [0.2, 2.]
        params['chunksize'] = 2
        params['max_workers'] = 16
    elif task_id == 2:
        ################################################################################
        #### Default for sound events from AudioSet and FSD50K (test set), 
        ####   and SRIRs from TAU-SRIR DB 
        ####   (./data_generator/data_synthesis_test.py)
        ################################################################################
        params['db_name'] = 'FSD50K'
        params['audio_format'] = 'both'
        params['dataset_type'] = 'test' # NOTE: synthesizing training sets or test sets
        params['nb_mixtures'] = 18 # TODO
        params['max_polyphony_target'] = 1
        params['output_dir'] = 'seld_{}_tau{}_ov{}_{}'.format(
            params['db_name'], params['nb_mixtures'], 
            params['max_polyphony_target'], params['dataset_type'])
        params['mixturepath'] /= params['output_dir']
        params['rooms'] = [[1, 2, 3, 4, 5, 6, 8, 9, 10]]
        params['mixture_duration'] = 60
        params['event_time_per_layer'] = 40
        params['chunksize'] = 22
        params['max_workers'] = 5
        TAU_SRIR_DB = Database_dir / 'TAU-SRIR_DB'
        params['rirpath'] = TAU_SRIR_DB / 'TAU-SRIR_DB'
        params['noisepath'] = TAU_SRIR_DB / 'TAU-SNoise_DB'
        params['add_noise'] = False
    
    params['mic_pos'] = np.array(params['mic_pos'])

    if params['tools'] == 'smir' and params['array_type'] == 'open':
        warnings.warn('SMIR does not support open array, change to rigid array!')
        params['array_type'] = 'rigid'
    if params['tools'] != 'pyroomacoustics' and params['method'] == 'hybrid':
        warnings.warn('Hybrid method only support pyroomacoustics, change to ISM!')
        params['method'] = 'ism'

    if params['mixturepath'].exists():
        if input('The mixture path {} exists! Do you want to overwrite it? (y/n)'.format(params['mixturepath'])) != 'y':
            exit(0)

    return params

if __name__ == '__main__':
    get_params()

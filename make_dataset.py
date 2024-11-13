import os
import sys
import pickle
from pathlib import Path

import utils
from data_generator.db_config import DBConfig
from get_parameters import get_params


def main(arg):

    taskid = int(arg[1])
    params = get_params(arg)
    print('\n TASK-ID: {}\n'.format(taskid))
    
    Path(params['mixturepath']).mkdir(parents=True, exist_ok=True)
    param_path = os.path.join(params['mixturepath'], 'params.txt')
    f = open(param_path, 'w')
    for key, value in params.items():
        word = "\t{}: {}\n".format(key, value)
        f.writelines(word)
        print(word)
    f.close()

    db_config_path = './db_config_{}.obj'.format(params['db_name'])
    ### Create database config based on params (e.g. filelist name etc.)
    if not os.path.isfile(db_config_path):
        db_config = DBConfig(params)
        # WRITE DB-config
        with open(db_config_path, 'wb') as f:
            pickle.dump(db_config, f)
        print('################')
        print(db_config_path, 'has been saved!')
        sys.exit()
    else:    
        # LOAD DB-config which is already done
        f = open(db_config_path, 'rb')
        with open(db_config_path, 'rb') as f:
            db_config = pickle.load(f)
        print('################')
        print(db_config_path, 'has been loaded!')
    
    if taskid == 1:
        from data_generator.data_synthesis import DataSynthesizer
        
        data_synth = DataSynthesizer(db_config, params)
        
        data_synth.create_mixtures(scenes='target_classes')
        # data_synth.create_mixtures(scenes='interf_classes')
        data_synth.create_metadata(add_interf=params['add_interf'])

        data_synth.write_metadata(scenes='target_classes')
        data_synth.synthesize_mixtures(
            add_interf=params['add_interf'], 
            audio_format=params['audio_format'],
            add_noise=params['add_noise'])
    elif taskid == 2:
        # Synthesize test data using TAU-SRIR DB
        from data_generator.data_synthesis_test import (MetadataSynthesizer, 
                                                        AudioSynthesizer,AudioMixer)
        metadata_synth = MetadataSynthesizer(db_config, params)
        metadata_synth.create_mixtures()
        metadata_synth.prepare_metadata_and_stats()
        metadata_synth.write_metadata()
        audio_synth = AudioSynthesizer(
            params, db_config, 
            metadata_synth._rirdata, 
            metadata_synth._mixtures,
            metadata_synth._mixture_setup)
        
        if params['add_interf']:
            params['target_classes'] = params['interf_classes']
            params['mixturepath'] = params['mixturepath'] / 'interf'
            params['max_polyphony_target'] = params['max_polyphony_interf']
            Path(params['mixturepath']).mkdir(parents=True, exist_ok=True)
            metadata_synth_interf = MetadataSynthesizer(db_config, params)
            metadata_synth_interf.create_mixtures()
            metadata_synth_interf.prepare_metadata_and_stats()
            metadata_synth_interf.write_metadata()
            audio_synth_interf = AudioSynthesizer(
                params, db_config, 
                metadata_synth._rirdata, 
                metadata_synth._mixtures,
                metadata_synth._mixture_setup)
            audio_mixer = AudioMixer(params)

        if params['audio_format'] in ['foa', 'both']:
            audio_synth.synthesize_mixtures('foa')
            if params['add_interf']:
                audio_synth_interf.synthesize_mixtures('foa')
                audio_mixer.mix_audio('foa')

        if params['audio_format'] in ['mic', 'both']:
            audio_synth.synthesize_mixtures('mic')
            if params['add_interf']:
                audio_synth_interf.synthesize_mixtures('mic')
                audio_mixer.mix_audio('mic')


if __name__ == '__main__':
    main(sys.argv)

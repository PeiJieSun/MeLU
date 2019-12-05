import os
import torch
import pickle
import numpy as np

from MetaCF import MetaCF
from options import config
from model_training import training
from data_generation import generate


if __name__ == "__main__":
    master_path= "/home/sunpeijie/files/task/data/meta_cf/ml"
    if not os.path.exists("{}/".format(master_path)):
        os.mkdir("{}/".format(master_path))
        # preparing dataset. It needs about 22GB of your hard disk space.
        generate(master_path)
    import pdb; pdb.set_trace()

    # training model.
    meta_cf = MetaCF(config)
    model_filename = "{}/models.pkl".format(master_path)
    if not os.path.exists(model_filename):
        # Load training dataset.
        training_data = np.load("{}/training_data.npy".format(master_path)).item()
        training_set_size = len(training_data['warm_state'].keys())
        
        #training_set_size = int(len(os.listdir("{}/warm_state".format(master_path))) / 4)
        supp_xs_s = []
        supp_ys_s = []
        query_xs_s = []
        query_ys_s = []
        '''
        for idx in range(training_set_size):
            supp_xs_s.append(pickle.load(open("{}/warm_state/supp_x_{}.pkl".format(master_path, idx), "rb")))
            supp_ys_s.append(pickle.load(open("{}/warm_state/supp_y_{}.pkl".format(master_path, idx), "rb")))
            query_xs_s.append(pickle.load(open("{}/warm_state/query_x_{}.pkl".format(master_path, idx), "rb")))
            query_ys_s.append(pickle.load(open("{}/warm_state/query_y_{}.pkl".format(master_path, idx), "rb")))
        '''
        state = 'warm_state'
        for u_id in training_data[state]:
            for x in training_data[state]['supp_x']:
                supp_xs_s.append([u_id, i_id])
            for y in training_data[state]['supp_y']:
                supp_ys_s.append([u_id, i_id])
            for x in training_data[state]['query_x']:
                query_xs_s.append([u_id, i_id])
            for y in training_data[state]['query_y']:
                query_ys_s.append([u_id, i_id])
        total_dataset = list(zip(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s))
        del(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s)
        training(meta_cf, total_dataset, batch_size=config['batch_size'], num_epoch=config['num_epoch'], model_save=True, model_filename=model_filename)
    else:
        trained_state_dict = torch.load(model_filename)
        meta_cf.load_state_dict(trained_state_dict)

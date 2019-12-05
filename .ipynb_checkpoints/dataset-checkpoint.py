import datetime
import pandas as pd


class movielens_1m(object):
    def __init__(self):
        #self.user_data, self.item_data, self.score_data = self.load()
        #self.score_data = self.load()

    def load(self):
        path = "movielens/ml-1m"
        
    
    def create(self):
        path = "movielens/ml-1m"
        
        training_data = defaultdict(dict)
    
        for state in states:
            state_training_data = defaultdict(dict)

            with open("{}/{}.json".format(dataset_path, state), encoding="utf-8") as f:
                dataset = json.loads(f.read())
            with open("{}/{}_y.json".format(dataset_path, state), encoding="utf-8") as f:
                dataset_y = json.loads(f.read())
            for _, user_id in tqdm(enumerate(dataset.keys())):
                u_id = int(user_id)
                seen_movie_len = len(dataset[str(u_id)])
                indices = list(range(seen_movie_len))

                if seen_movie_len < 13 or seen_movie_len > 100:
                    continue

                random.shuffle(indices)
                tmp_x = np.array(dataset[str(u_id)])
                tmp_y = np.array(dataset_y[str(u_id)])

                support_x = tmp_x[indices[:-10]]
                support_y = tmp_y[indices[:-10]]

                query_x = tmp_x[indices[-10:]]
                query_y = tmp_y[indices[-10:]]

                state_training_data[u_id] = {
                    'supp_x': support_x,
                    'supp_y': support_y,
                    'query_x': query_x,
                    'query_y': query_y
                }

            training_data[state] = state_training_data
        np.save('{}/training_data'.format(master_path), training_data)

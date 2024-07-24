import os
import os.path as osp
import json
import numpy as np

retri_base = ''  # the directory that saves the retrieve results of Faiss.  

scene_trailer_base = ''  # the directory of trailer's scene json file output by BaSSL
scene_movie_base = ''  # # the directory of movie's scene json file output by BaSSL
save_base = ''  # the save directory of the alignment json files. 

file_base = ''  # the directory of the files to be processed
files = os.listdir(file_base)


def findf(candidate, data):
    # depend on the candidate frame index, find its corresponding movie shot index 
    n_item = len(data)
    for idx in range(n_item):
        item = data[idx]
        start = int(item["frame"][0])
        end = int(item["frame"][1])
        if candidate >= start and candidate <= end:
            return idx

    return -1


for movidx in files:
    testidx = movidx.split('-')[0]  # movie index 
    tidx = movidx.split('-')[1]
    print(movidx)

    retri_data = np.load(osp.join(retri_base, 'result-flat-{}.npy'.format(movidx)))
    with open(osp.join(scene_trailer_base, movidx + '.json'), 'r') as f:
        scene_t_data = json.load(f)["shot_meta_list"]
    with open(osp.join(scene_movie_base, testidx + '.json'), 'r') as f:
        scene_m_data = json.load(f)["shot_meta_list"]

    n_t_shots = len(scene_t_data)  # the number of shots in the trailer
    n_m_shots = len(scene_m_data)  # the number of shots in the movie

    result_json = dict()

    for shotidx in range(n_t_shots):
        sets = []
        start1 = int(scene_t_data[shotidx]["frame"][0])
        end1 = int(scene_t_data[shotidx]["frame"][1])
        for tidx in range(start1, end1 + 1):
            four_candidate_frames = retri_data[tidx]
            for candidate_frame in four_candidate_frames:
                sets.append(findf(candidate_frame, scene_m_data))

        maxd = max(set(sets), key=sets.count)
        result_json[shotidx] = maxd

    save_path = osp.join(save_base, movidx + '.json')
    with open(save_path, 'w') as f:
        json.dump(result_json, f)

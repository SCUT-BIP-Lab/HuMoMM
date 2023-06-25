import pickle
out_train_path = '/data/pose_datasets/scut_sp/scut_sp_train.pkl'
out_val_path = '/data/pose_datasets/scut_sp/scut_sp_val.pkl'

with open(out_train_path, 'rb') as f:
    data = pickle.load(f)
print(len(data))

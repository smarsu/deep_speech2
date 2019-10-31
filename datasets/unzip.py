import os

root = '/share/datasets/data_aishell/wav'
files = os.listdir(root)
for file in files:
    os.system('tar -xvf {}'.format(os.path.join(root, file)))


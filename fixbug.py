import pickle

# 假设你的.pkl文件路径是'./data/data.pkl'
pkl_file_path = './data/data.pkl'

with open(pkl_file_path, 'rb') as file:
    data = pickle.load(file)

# 打印前几个样本来检查它们的结构
    print(data[1])
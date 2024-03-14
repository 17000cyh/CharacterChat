import pickle 

def data_align(plot_pkl_path, questoin_pkl_path):
    plot = pickle.load(open(plot_pkl_path),"rb")
    question = pickle.load(open(questoin_pkl_path),"rb")
    
import pickle
pos_list = pickle.load( open( "posList.p", "rb" ) )
neg_list = pickle.load( open( "negList.p", "rb" ) )
counter = pickle.load( open( "count_vect.p", "rb" ) )
training_counts =  pickle.load( open( "train.p", "rb" ) )

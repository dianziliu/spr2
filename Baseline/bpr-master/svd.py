import surprise    
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate


data=Dataset.load_builtin("ml-100k")
algo=SVD()
algo.train

cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
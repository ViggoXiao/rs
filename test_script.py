import os
import random
import numpy
import torch
import pandas as pd

from mlp.model import RatingModel, RatingModelMapper
from mlp.data_pipeline import load_recipes_info, load_testing_data_restricted


seed = 0
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)
random.seed(seed)
numpy.random.seed(seed)

threshold = 2000
rating_thres = 0.25

# set up devices
if not torch.cuda.is_available():
	device = torch.device('cpu')
else:
	device = torch.device('cuda')

pred_res = []
pred_res_true = []
pred_res_thres = []


model_name = 'THLELR_29_70'
model_epoch = 9


if __name__ == '__main__':
	print('loading recipes info to memory ...')
	recipe_info = load_recipes_info()
	print('loading testing dataset')
	iter = load_testing_data_restricted('processed/train_discretization.csv', 'processed/test_discretization.csv', threshold, 'processed/test_dataset_restricted.pth')
	print('creating model ...')
	model = RatingModel(
		input_dim={
			'sentence': 768,
			'tags': 551,
			'nutrition': 7,
		},
		block_embedding_dims=[1024, 512, 256],
		fc_dims=[1024, 512, 1],
	)
	mapper = RatingModelMapper(epsilon=-0.7)
	print('loading model ...')
	model.load_state_dict(torch.load(f'models/{model_name}-model-{model_epoch}.pth'))
	model.to(device)
	model.eval()
	print('predicting ...')
	with torch.no_grad():
		num_batches = len(iter)
		for i, (Xi, R) in enumerate(iter):
			print(f'Testing {i + 1}/{num_batches}')
			Xi = Xi.squeeze(0).numpy().tolist()
			R = R.squeeze(0)
			# move to device
			X = recipe_info[Xi].to(device)
			R = R.to(device)
			# predict
			R_pred = model(X, R)
			R_pred = mapper(R_pred)
			# print(X, R)
			# print(X.shape, R.shape)
			# print(R_pred)
			# print(R_pred[-1])
			pred_rating = float(R_pred[-1][0])
			pred_rating_round = round(pred_rating)
			if abs(pred_rating - pred_rating_round) <= rating_thres:
				pred_rating_thres = pred_rating_round
			else:
				pred_rating_thres = pred_rating
			pred_res_true.append(pred_rating)
			pred_res.append(pred_rating_round)
			pred_res_thres.append(pred_rating_thres)
			print('predicted rating:', pred_rating)
	os.makedirs('predictions', exist_ok=True)
	pd.DataFrame({
		'Id': numpy.arange(1, len(pred_res) + 1),
		'Predicted': pred_res_true
	}).to_csv(f'predictions/{model_name}-prediction-{model_epoch}-true.csv', index=False)
	pd.DataFrame({
		'Id': numpy.arange(1, len(pred_res) + 1),
		'Predicted': pred_res_thres
	}).to_csv(f'predictions/{model_name}-prediction-{model_epoch}-thres.csv', index=False)
	pd.DataFrame({
		'Id': numpy.arange(1, len(pred_res) + 1),
		'Predicted': pred_res
	}).to_csv(f'predictions/{model_name}-prediction-{model_epoch}.csv', index=False)

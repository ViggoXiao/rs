import random
import numpy
import torch

from mlp.utils import CosineAnnealingScheduler, LearningRateSchedulerComposer
from mlp.model import RatingModel
from mlp.winit import weight_init
from mlp.trainer import train
from mlp.data_pipeline import load_recipes_info, load_training_data_restricted


seed = 29
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)
random.seed(seed)
numpy.random.seed(seed)

num_epoch = 10
grad_accum_cnt = 5
epsilon = -0.7
max_lr = 0.005
weight_decay = 5e-4
threshold = 1500
step = 300

lr = LearningRateSchedulerComposer([
	CosineAnnealingScheduler(max_lr, num_epoch),
])

if __name__ == '__main__':
	print('loading recipes info to memory ...')
	recipe_info = load_recipes_info()
	print('loading training dataset ...')
	train_iter = load_training_data_restricted('processed/train_discretization.csv', threshold, step)
	print('creating model ...')
	model = RatingModel(
		input_dim={
			'sentence': 768,
			'tags': 551,
			'nutrition': 7,
		},
		block_embedding_dims=[1024, 512, 256],
		fc_dims=[1024, 512, 1]
	)
	print('initializing model ...')
	model.apply(weight_init)
	model.to('cuda')
	optimizer = torch.optim.Adam(model.parameters(), lr=lr(0), weight_decay=weight_decay)
	print('training model ...')
	train(model, train_iter, recipe_info, num_epoch, lr, optimizer, grad_accum_cnt, epsilon, 'THLELR_29_70')

import os
from mlp.data_pipeline import TestRatingDataPipelineRestricted


threshold = 2000

os.makedirs('processed', exist_ok=True)
TestRatingDataPipelineRestricted('processed/train_discretization.csv', 'processed/test_discretization.csv', threshold).save('processed/test_dataset_restricted.pth')

# #Evaluation
# # Use the following function to measure the test accuracy of your trained model. 
# import torch 
# from model.logistic_regression import module;
# from data.mnist_dataset import test_dataloader

# correct_predictions = 0
# predictions = 0

# # Iterate through test dataset
# for test_images, test_targets in test_dataloader:
#     test_images = test_images.view(-1, 28 * 28)

#     # Forward pass only to get logits/output
#     outputs = module(test_images)

#     # Get predictions from the maximum value
#     _, predicted = torch.max(outputs.data, 1)

#     predictions += test_targets.size(0)

#     if torch.cuda.is_available():
#         correct_predictions += (predicted.cpu() == test_targets.cpu()).sum()
#     else:
#         correct_predictions += (predicted == test_targets).sum()

# print(correct_predictions.item() / predictions)
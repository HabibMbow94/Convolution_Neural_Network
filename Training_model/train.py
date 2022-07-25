# # Training Model

# import torch;
# from data.mnist_dataset import train_dataloader;
# from model.logistic_regression import optimizer, module, criterion

# if torch.cuda.is_available():
#       DEVICE = "cuda:0" # CUDA signifie « Compute Unified Device Architecture »
# else:
#   DEVICE = "cpu" # CPU « signifie Central Processing Unit »

# # Train the model. If everything is correct, the loss should go below 0.45.
# # We will use the following generic training loop for a PyTorch model.
# EPOCHS = 5

# # Exponential moving average of the loss:
# ema = None

# for epoch in range(EPOCHS):
#   for batch_index, (train_images, train_targets) in enumerate(train_dataloader):
#     train_images = train_images.view(-1, 28 * 28).requires_grad_().to(device=DEVICE)
#     train_targets = train_targets.to(device=DEVICE)

#     # Clear gradients w.r.t. parameters
#     optimizer.zero_grad()

#     # Forward pass to get output/logits
#     outputs = module(train_images)

#     # Calculate Loss: softmax --> cross entropy loss
#     loss = criterion(outputs, train_targets)

#     # Getting gradients w.r.t. parameters
#     loss.backward()

#     # Updates parameters:
#     optimizer.step()

#     # NOTE: It is important to call .item() on the loss before summing.
#     if ema is None:
#         ema = loss.item()
#     else:
#         ema += (loss.item() - ema) * 0.01

#     if batch_index % 500 == 0:
#         print(
#             "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
#                 epoch,
#                 batch_index * len(train_images),
#                 len(train_dataloader.dataset),
#                 100.0 * batch_index / len(train_dataloader),
#                 ema,
#             ),
#         )
        
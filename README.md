For a relatively simple project, I'd like to build a neural network that takes RGB values, and outputs wether the colour is light, or dark.

This project is inspired by Jabrils (YouTuber), he mentioned it as being a good first project.

## Architecture

High level, this _seems_ relatively simple.

Our input is an array of 3 numbers, 0 - 255. For example a valid input is `[173, 158, 114]`

This is a non-linear problem, so we need to have some non-linearity in our neural net. This means we will want more than one hidden layer, and will need a non-linear activation function after each layer.

We will use ReLU as our activation functions for the hidden layer neurons. We'll have 4 neurons per hidden layer. (an arbitrary decision)

We've now defined our input, and hidden layers. We still need to define our output, loss function, and optimizer.

This is a binary classification problem as we are trying to predict between light and dark. We'll use a single neuron as our output. [From my research, online sources suggest using a sigmoid function to detemine our output normalization](https://towardsdatascience.com/sigmoid-and-softmax-functions-in-5-minutes-f516c80ea1f9#:~:text=But%20if%20both%20functions%20map,extension%20of%20the%20Sigmoid%20function.)

A sigmoid function outputs a probability that the input belongs to the first class. I.e. if 0 is light, and 1 is dark, a sigmoid function could output 0.8. This means that there is an 80% chance that the input is light.

We now understand our input, hidden layers, and output. We finally need to determine the optimizer for backpropogation. There seems to not be many strong opinions on optimizers for binary classification online. [Some people suggest ADAM but it can cause overfitting for smaller networks?](https://ai.stackexchange.com/questions/18206/what-kind-of-optimizer-is-suggested-to-use-for-binary-classification-of-similar) I don't fully understand this yet, so I'm going to use [stochastic gradient descent (SGD)](https://www.youtube.com/watch?v=vMh0zPT0tLI)) for now as I have an intuition for how it works.

We now understand our input, hidden layers, output (and output normalization), and optimizer. We still need to determine our loss function to understand the difference between our labeled input, vs our networks prediction. Online sources suggest using [Binary Cross Entropy Loss](https://stats.stackexchange.com/questions/186091/what-loss-function-should-i-use-for-binary-detection-in-face-non-face-detection) for our binary classification problem. [PyTorch has BCELoss baked in](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html)

We now know our input, hidden layers, output, output normalization, optimizer, and loss function. I believe these are all of the pieces of our architecture that we need to worry about. Next up, creating our dataset.

## Dataset

Based on our architecture, our dataset should be RGB values, labeled with either `light`, or `dark`.

I want the dataset to be human readable, hence using the label `light` and `dark`, rather an `0` or `1`.

Once we've built out dataset, we'll need to create a DataLoader in PyTorch that can retrieve values. Pulling the RGB values should be relatively simple as they are already numeric values. However, we'll need to convert the labels into an integer. `0` for `light` and `1` for `dark`.

I'm not sure the _best_ way to achieve this, however I do know the _simplest_. In our `__getitem__` function in the `DataLoader`, I'll create a simple if statement that returns `0` if the label is `light` , and `1` if the label is dark.

## Implementation

We've now got an idea for creating our dataset, as well as how to build the neural network. We now need to implement this design in PyTorch.

I'm going to begin by creating a small dataset as a CSV, conforming to the design in the Dataset section of this note. I'll then write a DataLoader in PyTorch and test that it accurately returns the data and the label. Once that is completed, I'll return to this note to move to the next stage, likely building the Neural Network Architecture. After building the NN, I will define our training, and validation functions. At that point we can run our epochs (train+validate) and see how we fare!

I've created a DataLoader that _appears_ to work. I parsed the input rgb string, mapped it to a new list and converted the numeric strings to numbers. For the labels, I did as mentioned above, a simple if statement to return 0 if light, and 1 if dark.

I initially thought that I needed to convert the list into a tensor, however it seems as though returning the list in `__getitem__` automatically converts it into a tensor. Unfortunately, it seems like this autoconversion is actually incorrect, so I've reverted back to making my own tensor. We'll see if this needs to be changed as we progress.

I've managed to get a batch of data through the network which returns some results. I'm now going to run it through a sigmoid function to get the probability for which class it belongs to. We now have the predicted probability after running the results through sigmoid. We are now going to take the results and run it through our Binary Cross Entropy Loss function to determine the loss.

We now have determined the loss between the input and output, next I have to work out how to run backpropogation. I'm going to do this step by step to begin with, then will abstract it into the train and validate functions later once I understand the process.

## Debugging

I'm not sure why, but now that I've created the train and test loop, we're not making any progress with the model. It's seemingly not updating at all.

I'm writing a script to generate a tonne of colors and their respective labels, `0-127` for dark and `127-255` for light. It's not perfect representation of dark and light colors, but it's a start. I'm going to use the output of this script to continue training the model in case a lack of data was the issue.

I've changed the format slightly so that there are three rows `[red, green, blue, label]` so I need to update my dataloader accordingly.

I've been debugging the model for a while now and finally stumbed upon the reason why my gradients were stuck at zero (or not updating). Thanks to [this article I found online](https://discuss.pytorch.org/t/loss-does-not-change-and-weights-remain-zero/33717) It mentioned that the person asking the question was using two non-linear layers in their output. Both `ReLU` and `Sigmoid` which is *exactly* what I was doing! I've removed the final `ReLU` layer and the accuracy is improving during the epochs!

I'm going to experiment with changing the learning rate, batch size, and epochs (aka the hyperparameters) to see if I can get something working.

Once I'm at a better accuracy rate during training, I'd like to see how to save the model and make new predictions with it.

Right after I wrote the above I ran the code again, with no changes, and accuracy was stuck at 0%... I'm not sure what's going on here that causes it to work _sometimes_.

### Accuracy

I'm not sure how the accuracy score works, I'm going to look more indepth as to how the accuracy is calculated, as without that I can't really tell if my model is working. I can rely on the loss decreasing, but I'd like to know how accurate the model is.

From what I can tell, calling `torch.sum(a == b)` will return the sum of all values in each tensor that satisfy the condition `a == b`. Therefore, to get the number of results where `pred === label` in a particular batch, you can call `torch.sum(pred == label)`.

It compares each item in the next tensor and outputs how many match.

Example:

```python

def test_sum():
  a = torch.tensor([0, 1, 1, 0])
  b = torch.tensor([0, 0, 1, 1])
  print(torch.sum(a == b))

test_sum()
# tensor(2)
```

### Saving the Model

I'm not entirely sure why it didn't always work, but after correcting the accuracy, I wanted to move on. So I took a look at what it would take to save the model.

Initially, I saved the weights to a pth file, but realised I needed to load a bunch of other data if I wanted to use the model. So I saved the model using `torch.save`

Then, in another file, I imported the `model.pth` and ran it. However, it seems that due to python serialization, it required me to have the NN class present, so I copied it over to the file. I also needed the sigmoid function to interperate the results.

Once this was done, I was able to make some predictions on new data, which worked pretty well!

I'm very happy with my first attempt at building a basic neural network, and I'm excited to continue learning.
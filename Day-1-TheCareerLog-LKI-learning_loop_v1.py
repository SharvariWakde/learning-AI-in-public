# Day 1: Trying to understand how "learning" actually happens
# Goal (in my own words):
# Start with a wrong guess and slowly improve it using feedback

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Starting with zero because I want to see how improvement happens
weight = 0.0
learning_rate = 0.01  # not sure if this is ideal yet

def predict(x, w):
    return w * x

# Squared error feels intuitive for now
def loss(y_true, y_pred):
    return (y_true - y_pred) ** 2

# Training loop
for epoch in range(20):  # What is an epoch? its a one complete pass through the entire training dataset, allowing the model to see and learn from every single data point once to adjust its internal parameters
    total_loss = 0

    for i in range(len(x)):
        y_pred = predict(x[i], weight)
        error = y_pred - y[i]

        # This part confused me at first:
        # how much should the weight change?
        gradient = 2 * error * x[i]

        # Small steps instead of jumping to the answer
        weight -= learning_rate * gradient

        total_loss += loss(y[i], y_pred)

    print(f"Epoch {epoch+1} | weight={weight:.4f} | loss={total_loss:.4f}")

print("Final weight after learning:", weight)

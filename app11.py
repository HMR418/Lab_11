import tensorflow as tf

# Load the .h5 model file
model = tf.keras.models.load_model("tf_bridge_model.h5")

# 1. Print a summary of the model's layers
model.summary()

# 2. Get the model architecture in JSON
model_json = model.to_json()
with open("model_architecture.json", "w") as json_file:
    json_file.write(model_json)

print("Model architecture saved to model_architecture.json")

# 3. (Optional) If you want to see all layer weights, you can iterate:
for layer in model.layers:
    weights = layer.get_weights()
    # 'weights' is a list of numpy arrays, one for each parameter tensor (kernel, bias, etc.)
    print(layer.name, "has", len(weights), "weight tensors.")

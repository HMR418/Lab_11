import tensorflow as tf

# When loading the model, map "mse" to the built-in mean_squared_error function
model = tf.keras.models.load_model(
    "tf_bridge_model.h5",
    custom_objects={"mse": tf.keras.metrics.mean_squared_error}
)

# 1. Print a summary of the model's layers
model.summary()

# 2. Get the model architecture in JSON and save it
model_json = model.to_json()
with open("model_architecture.json", "w") as json_file:
    json_file.write(model_json)

print("Model architecture saved to model_architecture.json")

# 3. (Optional) If you want to see all layer weights, you can iterate:
for layer in model.layers:
    weights = layer.get_weights()
    # 'weights' is a list of numpy arrays, one for each parameter tensor (kernel, bias, etc.)
    print(layer.name, "has", len(weights), "weight tensors.")

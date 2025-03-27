import tensorflow as tf

# Map the "mse" reference to tf.keras.losses.mean_squared_error
model = tf.keras.models.load_model(
    "tf_bridge_model.h5",
    custom_objects={"mse": tf.keras.losses.mean_squared_error}
)

# 1. Print a summary of the model's layers
model.summary()

# 2. Get the model architecture in JSON and save it
model_json = model.to_json()
with open("model_architecture.json", "w") as json_file:
    json_file.write(model_json)

print("Model architecture saved to model_architecture.json")

# 3. (Optional) Iterate over layers to inspect weight tensors
for layer in model.layers:
    weights = layer.get_weights()
    print(layer.name, "has", len(weights), "weight tensors.")


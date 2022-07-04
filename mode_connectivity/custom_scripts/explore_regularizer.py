# %%
import tensorflow as tf
from mode_connectivity.curves.layers import DenseCurve


# %%
def print_info(layer):
    def names(list_):
        print([i.name for i in list_])

    def print_list_info(name, list_, only_names=True):
        print(f"{name}:")
        names(list_) if only_names else print(list_)
        print(f"# {name} = {len(list_)}")
        print("\n")

    print("\n")
    print(f"LAYER: {layer.name.upper()}")
    print_list_info("Losses", layer.losses, only_names=False)
    print_list_info("Trainable Variables", layer.trainable_variables)
    print_list_info("Non Trainable Variables", layer.non_trainable_variables)


# %%
units = 5
shape = (units, units)
# %%
print("\nAFTER INIT")
layer_base = tf.keras.layers.Dense(
    units=5,
    kernel_initializer="ones",
    kernel_regularizer=tf.keras.regularizers.L2(0.5),
)
print_info(layer_base)
#%%
layer_curve = DenseCurve(
    units=5,
    fix_points=[True, False, True],
    kernel_initializer="ones",
    kernel_regularizer=tf.keras.regularizers.L2(0.5),
)
print_info(layer_curve)

# %%
print("\nAFTER BUILD")
layer_curve.build(shape)
print_info(layer_curve)
# %%
layer_base.build(shape)
print_info(layer_base)


# %%
inputs = tf.ones(shape)
print("\nAFTER CALL")
layer_base(inputs)
print_info(layer_base)

#%%
layer_curve((inputs, tf.ones((3,)) / 3))
print_info(layer_curve)


# %%
tf.random.set_seed(1)
curve_weights = tf.ones((3,)) / 3
regularizer = tf.keras.regularizers.L2(0.5)

#%%
class ModelDense(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(
            units=5,
            kernel_initializer="ones",
        )

    def call(self, inputs, **kwargs):
        x = self.dense(inputs)
        return x


#%%
class ModelDenseReg(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(
            units=5,
            kernel_initializer="ones",
            kernel_regularizer=regularizer,
        )

    def call(self, inputs, **kwargs):
        x = self.dense(inputs)
        return x


#%%
class ModelNoL2(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = DenseCurve(
            units=5,
            fix_points=[True, False, True],
            kernel_initializer="ones",
        )

    def call(self, inputs, **kwargs):
        x = self.dense((inputs, curve_weights))
        return x


#%%
class ModelL2(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = DenseCurve(
            units=5,
            fix_points=[True, False, True],
            kernel_initializer="ones",
            kernel_regularizer=regularizer,
        )

    def call(self, inputs, **kwargs):
        x = self.dense((inputs, curve_weights))
        # self.add_loss(tf.reduce_sum(model.losses))
        return x


#%%
loss = tf.keras.losses.MeanAbsoluteError()
model = ModelDense()
model.compile(optimizer=tf.keras.optimizers.SGD(), loss=loss)

model_reg = ModelDenseReg()
model_reg.compile(optimizer=tf.keras.optimizers.SGD(), loss=loss)

nol2_model = ModelNoL2()
nol2_model.compile(optimizer=tf.keras.optimizers.SGD(), loss=loss)

l2_model = ModelL2()
l2_model.compile(optimizer=tf.keras.optimizers.SGD(), loss=loss)
#%%
inputs = tf.ones(shape=(5, 5))
targets = tf.ones(shape=(5, 5)) * 4
# MAE should be 1
#%%
print("Normal Model with Dense layer")
out = model(inputs)
expected_loss = loss(out, targets)
print("Expected Loss", expected_loss)
results = model.fit(inputs, targets)
print("Fit Loss", results.history["loss"])

#%%
print("Normal Model with Dense layer + L2Regularizer")
out = model_reg(inputs)
expected_loss = loss(out, targets) + regularizer(model_reg.dense.kernel)
print("Expected Loss", expected_loss)
results = model_reg.fit(inputs, targets)
print("Fit Loss", results.history["loss"])

#%%
print("Model with DenseCurve layer")
out = nol2_model(inputs)
expected_loss = loss(out, targets)
print("Expected Loss", expected_loss)
results = nol2_model.fit(inputs, targets)
print("Fit Loss", results.history["loss"])

#%%
print("Model with DenseCurve layer + L2Regularizer")
out = l2_model(inputs)
expected_loss = loss(out, targets) + tf.add_n(
    regularizer(kernel) for kernel in l2_model.dense.curve_kernels
)
print("Expected Loss", expected_loss)
results = l2_model.fit(inputs, targets)
print(results.history["loss"])

#%%
print("\nAFTER CALL")
#%%
print_info(model)
#%%
print_info(model_reg)
#%%
print_info(nol2_model)
#%%
print_info(l2_model)

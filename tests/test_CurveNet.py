import tensorflow as tf
import numpy as np

from mode_connectivity.curves.curves import Bezier
from mode_connectivity.curves.net import CurveNet
from mode_connectivity.models import CNN, MLP

import pytest

from mode_connectivity.models.cnn import CNNCurve


class TestCurveNet:
    @pytest.fixture
    def model_dir(tmpdir):
        return tmpdir.mkdir("model")

    @pytest.fixture
    def initialized_curve_net(self):
        return CurveNet(
            num_classes=10,
            num_bends=3,
            weight_decay=1e4,
            curve=Bezier,
            curve_model=CNN.curve,
            fix_start=True,
            fix_end=True,
            architecture_kwargs=CNN.kwargs,
        )
    # Same for MLP.curve

    @pytest.fixture
    def built_curve_net(self, initialized_curve_net):
        input_shape = [
            tf.TensorShape((128, 28, 28, 1)),
            tf.TensorShape((3)),
        ]
        initialized_curve_net.curve_model.build(input_shape)
        built_curve_net= initialized_curve_net
        return built_curve_net

        #OLD:
        #initialized_curve_net(tf.random.uniform((128, 28, 28, 1)))

    # Same for MLP.curve

    def test_init(self, initialized_curve_net):
        temp_net = initialized_curve_net

    def test_call(self, initialized_curve_net):
        output= initialized_curve_net(inputs= tf.random.uniform((128, 28, 28, 1)))

        assert output.shape == (128, 10) #(input_size[0], num_classes)

    # def test_import_base_parameters(self, initialized_curve_net):
    #     base_model= CNN.base(10)
    #     base_model(tf.random.uniform((128, 28, 28, 1)))

    #     initialized_curve_net.import_base_parameters(base_model, 0)

    def test_import_base_parameters(self, initialized_curve_net):
    #def test_import_base_parameters(self):
       #TO DO: Als Fixture mit anderen Parametrisierungen, zb index letzter
        #TO DO: Noch vergleichen ob in curve die mit index anders ist als vor load
        # und alle anderen gleich wie vor laod

        #Weg dann
        # initialized_curve_net= CurveNet(
        #     num_classes=10,
        #     num_bends=3,
        #     weight_decay=1e4,
        #     curve=Bezier,
        #     curve_model=CNN.curve,
        #     fix_start=True,
        #     fix_end=True,
        #     architecture_kwargs=CNN.kwargs)

        curve_net = initialized_curve_net

        base_model= CNN.base(10, 1e4)
        base_model.build(input_shape=(None, 28, 28, 1))

        curve_net._build_from_base_model(base_model)

        temp_var= curve_net.curve_model.variables


        #curve_weights_old = {w.name.split("/", 2)[2]: w for w in curve_net.curve_model.variables}
        curve_weights_old = {w.name: tf.Variable(w) for w in curve_net.curve_model.variables}

        #copy, remove

        index= 0
        curve_net.import_base_parameters(base_model, index)

        #curve_weights = {w.name.split("/", 2)[2]: w for w in curve_net.curve_model.variables}
        curve_weights = {w.name: w for w in curve_net.curve_model.variables}
        base_weights = {w.name: w for w in base_model.variables}

        for curve_param_name, curve_param in curve_weights.items():
            curve_param_old= curve_weights_old.get(curve_param_name)
            parameter_index= curve_net._find_parameter_index(curve_param_name)

            if parameter_index == index:
                base_param_name= curve_net._get_base_name(curve_param_name, index)
                base_param = base_weights.get(base_param_name)
                if base_param is None:
                    continue

                assert tf.experimental.numpy.allclose(base_param, curve_param)
                assert not tf.experimental.numpy.allclose(curve_param_old, curve_param)

            elif parameter_index != index:
                assert tf.experimental.numpy.allclose(curve_param_old, curve_param)





        # for base_layer_name, base_layer_param in base_weights.items():
        #     curve_layer_param= weights[base_layer_name]
            



        # for name, weight in weights.items():
        #     parameter_index = self._find_parameter_index(name)

        # base_name = self._get_base_name(name, index)
        # base_weight = base_weights.get(base_name)


        # curve_layers = initialized_curve_net.curve_layers.layers
        # for i in range(len(curve_layers)):
        #     curve_net_kernel= curve_layers[i].curve_kernels[index]
        #     curve_net_bias= curve_layers[i].curve_biases[index]

        #     #base_model_kernel= base_model.trainable_variables[(2*i)-1]
        #     #base_model_bias= base_model.trainable_variables[(2*i)]


        #     assert tf.experimental.numpy.allclose(curve_net_bias, base_model_bias)
        #     assert tf.experimental.numpy.allclose(curve_net_kernel, base_model_kernel)

    def test_init_linear(self):
        pass
        #TO DO

    def test_get_weighted_parameters(self):
        pass
        #TO DO



# tempcl= TestCurveNet()
# tempcl.test_import_base_parameters()



# #################
# # %%

# curve_net= CurveNet(
#     num_classes=10,
#     num_bends=3,
#     weight_decay=1e4,
#     curve=Bezier,
#     curve_model=CNN.curve,
#     fix_start=True,
#     fix_end=True,
#     architecture_kwargs=CNN.kwargs,
# )

# curve_net(tf.random.uniform((128, 28, 28, 1)))

# base_model= CNN.base(10, 1e4)
# base_model(tf.random.uniform((128, 28, 28, 1)))

# curve_weights_old = {w.name: w for w in curve_net.curve_model.variables}
# print(curve_weights_old.keys())

# index= 0
# curve_net.import_base_parameters(base_model, index)

# curve_weights = {w.name: w for w in curve_net.curve_model.variables}




# class TestCurveNet:
    # def test_init(self):
    #     net = CurveNet(
    #         num_classes=10,
    #         num_bends=3,
    #         weight_decay=1e4,
    #         curve=Bezier,
    #         curve_model=CNN.curve,
    #         fix_start=True,
    #         fix_end=True,
    #         architecture_kwargs=CNN.kwargs,
    #     )

    # def test_call(self):
    #     net = CurveNet(
    #         num_classes=10,
    #         num_bends=3,
    #         weight_decay=1e4,
    #         curve=Bezier,
    #         curve_model=CNN.curve,
    #         fix_start=True,
    #         fix_end=True,
    #         architecture_kwargs=CNN.kwargs,
    #     )
    #     net(tf.random.uniform((128, 28, 28, 1)))

    # def test_save_model(self, model_dir):
    #     net = CurveNet(
    #         num_classes=10,
    #         num_bends=3,
    #         weight_decay=1e4,
    #         curve=Bezier,
    #         curve_model=CNN.curve,
    #         fix_start=True,
    #         fix_end=True,
    #         architecture_kwargs=CNN.kwargs,
    #     )
    #     net(tf.random.uniform((128, 28, 28, 1)))
    #     net.save(filepath=model_dir)

    # def test_save_weights(self, model_dir):
    #     net = CurveNet(
    #         num_classes=10,
    #         num_bends=3,
    #         weight_decay=1e4,
    #         curve=Bezier,
    #         curve_model=CNN.curve,
    #         fix_start=True,
    #         fix_end=True,
    #         architecture_kwargs=CNN.kwargs,
    #     )
    #     net(tf.random.uniform((128, 28, 28, 1)))
    #     net.save_weights(filepath=model_dir)

    # def test_load_weights(self, model_dir):
    #     net = CurveNet(
    #         num_classes=10,
    #         num_bends=3,
    #         weight_decay=1e4,
    #         curve=Bezier,
    #         curve_model=CNN.curve,
    #         fix_start=True,
    #         fix_end=True,
    #         architecture_kwargs=CNN.kwargs,
    #     )
    #     net(tf.random.uniform((128, 28, 28, 1)))
    #     net(tf.random.uniform((128, 28, 28, 1)))
    #     net(tf.random.uniform((128, 28, 28, 1)))
    #     net.save_weights(filepath=model_dir)

    #     net2 = CurveNet(
    #         num_classes=10,
    #         num_bends=3,
    #         weight_decay=1e4,
    #         curve=Bezier,
    #         curve_model=CNN.curve,
    #         fix_start=True,
    #         fix_end=True,
    #         architecture_kwargs=CNN.kwargs,
    #     )
    #     net2.build(input_shape=tf.TensorShape((None, 28, 28, 1)))

    #     net_weights = net.get_weights()
    #     net2_weights = net2.get_weights()
    #     assert not all(np.allclose(w1, w2) for w1, w2 in zip(net_weights, net2_weights))

    #     net2.load_weights(filepath=model_dir)

    #     net_weights = net.get_weights()
    #     net2_weights = net2.get_weights()
    #     assert all(np.allclose(w1, w2) for w1, w2 in zip(net_weights, net2_weights))


# # %%
# import tensorflow as tf
# import numpy as np

# from mode_connectivity.curves.curves import Bezier
# from mode_connectivity.curves.net import CurveNet
# from mode_connectivity.models import CNN, MLP

# import pytest

# from mode_connectivity.models.cnn import CNNCurve
# curve_net= CurveNet(
#             num_classes=10,
#             num_bends=3,
#             weight_decay=1e4,
#             curve=Bezier,
#             curve_model=CNN.curve,
#             fix_start=True,
#             fix_end=True,
#             architecture_kwargs=CNN.kwargs)
# # %%
# input_shape = [
#     tf.TensorShape((128, 28, 28, 1)),
#     tf.TensorShape((3)),
# ]
# curve_net.curve_model.build(input_shape)
# # %%
# curve_net.curve_model.build(tf.TensorShape((128, 28, 28, 1)))
# # %%


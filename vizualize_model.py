import coremltools
model = coremltools.models.MLModel("inception_net_6e.mlmodel")
model.visualize_spec()


import torch
from pytorch_nets import InceptionFeatures, CompressFeatures
import torch.onnx
from onnx_coreml import convert
import onnx
import coremltools
from PIL import Image

inception_onnx_name = "inception_net_6e.onnx"
compress_onnx_name = "compress_features.onnx"
inception_ml_name = "inception_net_6e.mlmodel"
compress_ml_name = "compress_features.mlmodel"
img_size = 256
inception_in_name = 'input_img'
inception_out_name = 'inception_features'
compress_in_name = inception_out_name
compress_out_name = 'compress_features'
target_ios_verion = '13'
test_image_name = 'test_img.jpg'
onnx_opset = 9

def convert_to_onnx():
    sample_img = torch.rand(3,img_size,img_size)
    inception_net = InceptionFeatures()
    compress_features = CompressFeatures()
    inception_net.eval()
    compress_features.eval()
    
    inception_features = inception_net(sample_img)
    compressed_features = compress_features(inception_features)
    
    torch.onnx.export(inception_net, (sample_img), inception_onnx_name, export_params=True,
        opset_version=onnx_opset,
        input_names=[inception_in_name],
        output_names=[inception_out_name])
    
    torch.onnx.export(compress_features, (inception_features), compress_onnx_name, export_params=True,
    opset_version=onnx_opset,
    input_names=[compress_in_name],
    output_names=[compress_out_name])
            
def convert_to_ml():
    model = onnx.load(inception_onnx_name)
    mlmodel = convert(model, image_input_names=[inception_in_name],target_ios=target_ios_verion)
    mlmodel.save(inception_ml_name)

    model = onnx.load(compress_onnx_name)
    mlmodel = convert(model,target_ios=target_ios_verion)
    mlmodel.save(compress_ml_name)

def run_mlmodel():
    print('loading mlmodel')
    inception_6e = coremltools.models.MLModel(inception_ml_name)
    compress = coremltools.models.MLModel(compress_ml_name)
    test_input_img = Image.open(test_image_name)
    print('computing predictions using mlmodel')
    inception_out = inception_6e.predict({inception_in_name: test_input_img}, usesCPUOnly=True)
    compress_out = compress.predict({compress_in_name: inception_out[inception_out_name]})
    print(compress_out)


if __name__ == '__main__':
  convert_to_onnx()
  convert_to_ml()
  run_mlmodel()
  print('finished')

#include <torch/script.h>
#include <torch/torch.h>
//#include <ATen/Tensor.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#define GPU 1
/* main */
int main(int argc, const char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage: example-app <path-to-exported-script-module> "
      << "<path-to-image>\n";
    return -1;
  }

  // Deserialize the ScriptModule from a file using torch::jit::load().
  std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(argv[1]);
  assert(module != nullptr);
  std::cout << "load model ok\n";

  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
  auto tensor_in = torch::ones({1,3,32,96});
  if(GPU){
    module->to(at::kCUDA);
    tensor_in = tensor_in.to(at::kCUDA);
  }
  inputs.push_back(tensor_in);
  // evalute time
  double t = (double)cv::getTickCount();
  torch::Tensor out = module->forward(inputs).toTensor();
//  std::cout << out[0][0] << out[0][1] << std::endl;
  t = (double)cv::getTickCount() - t;
  printf("execution time = %gs\n", t / cv::getTickFrequency());
  inputs.pop_back();

  // load image with opencv and transform
  cv::Mat image;
  image = cv::imread(argv[2], 1);
  cv::cvtColor(image, image, CV_BGR2RGB);
  cv::Mat img_float;
  cv::resize(image, image, cv::Size(96, 32));
  image.convertTo(img_float, CV_32F, 1.0/255);
  //std::cout << img_float.at<cv::Vec3f>(56,34)[1] << std::endl;
  auto img_tensor = torch::CPU(torch::kFloat32).tensorFromBlob(img_float.data, {1, 32, 96, 3});
  img_tensor = img_tensor.permute({0,3,1,2});
  img_tensor[0][0] = img_tensor[0][0].sub_(0.1307).div_(0.3081);
  img_tensor[0][1] = img_tensor[0][1].sub_(0.1307).div_(0.3081);
  img_tensor[0][2] = img_tensor[0][2].sub_(0.1307).div_(0.3081);
  auto img_var = torch::autograd::make_variable(img_tensor, false);
  if(GPU)
    img_var = img_var.to(at::kCUDA);
  inputs.push_back(img_var);

  // Execute the model and turn its output into a tensor.
  torch::Tensor out_tensor = module->forward(inputs).toTensor();
  std::cout << out_tensor.slice(/*dim=*/0, /*start=*/0, /*end=*/2) << '\n';
//  std::cout << out_tensor[0][0] << out_tensor[0][1] << std::endl;
//  std::cout << out_tensor.slice(/*dim=*/0, /*start=*/0, /*end=*/1) << '\n';

  return 0;
}


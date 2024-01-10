# HLS Implementation of Llama 2

## Prerequisites
1. EC2 instance (z1d.2xlarge recommended)
3. AWS FPGA Developer AMI [(install)](https://github.com/aws/aws-fpga/blob/master/Vitis/README.md)
4. S3 Bucket
5. Pretrained model parameters & tokenizer can be found [here](https://drive.google.com/drive/folders/1ivGgCHgk8W_PgZdNSNwyBwqFwCQr_FvS?usp=sharing)

## Build Instructions 
1. Follow the setup for AWS FPGA Developer AMI
2. Open the project in Vitis IDE
3. Verify the install by building the Software Emulation
4. If no run configurations exist, add a new System Project Debug configuration with the following user provided arguments
     `${project_loc:llama_xrt}/src/weights.bin -z ${project_loc:llama_xrt}/src/tokenizer.bin -t 0.8 -n 256 -i "{prompt}" -k`

## Run Instructions
1. Run the Hardware build, should take around ~12 hours.
2. Extract the .xclbin file and export to an AWS AMI following the directions [here](https://github.com/aws/aws-fpga/blob/master/Vitis/README.md#2-create-an-amazon-fpga-image-afi)
3. Launch an EC2 F1 instance and copy the generated .awsxclbin file to the same directory as the host code
4. Start the FPGA runtime (more instructions [here](https://github.com/aws/aws-fpga/blob/master/Vitis/README.md#2-create-an-amazon-fpga-image-afi))

    `cd $AWS_FPGA_REPO_DIR`
   
    `source vitis_setup.sh`
   
    `source vitis_runtime_setup.sh`
6. Ensure that devtoolset-9 (or g++-9) is installed and enabled
7. Build the host executable
   
   `g++ -Wall -O3 -std=c++17 src/llama2.cpp -o llama2 -I${XILINX_XRT}/include  -L${XILINX_XRT}/lib -lxrt_coreutil -lpthread -lrt -lstdc++`
8. Run the host executable

    `./llama2 {path to weights} -z {path to tokenizer} -t {temp} -n {steps} -i {prompt} -k {path to kernel}`
   

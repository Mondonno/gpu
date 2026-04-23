#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Metal/Metal.hpp>

#include <iostream>
#include <fstream>
#include <istream>

#define CHECK_ERROR(error) if(error != nullptr) { std::cerr << (*error).description() << "\n"; exit(EXIT_FAILURE); }
#define KERNEL_SOURCE "kernel.metal"

NS::SharedPtr<MTL::Library> loadKernelLibary(NS::SharedPtr<MTL::Device> device, const std::string& path) {
    NS::Error* error = nullptr;
    std::ifstream kernelSource(path);

    std::string source((std::istreambuf_iterator<char>(kernelSource)), std::istreambuf_iterator<char>());
    
    auto libraryContents = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
    auto libraryFromDevice = device->newLibrary(libraryContents, nullptr, &error);
    
    NS::SharedPtr<MTL::Library> library = NS::TransferPtr<MTL::Library>(libraryFromDevice);
    
    CHECK_ERROR(error);
    
    return library;
}

NS::SharedPtr<MTL::Buffer> floatVectorToDeviceBuffer(NS::SharedPtr<MTL::Device> device, std::vector<float> vec) {
    auto payload = NS::TransferPtr<MTL::Buffer>(device->newBuffer(vec.data(), NS::UInteger(vec.size() * sizeof(float)), MTL::ResourceOptions(MTL::ResourceStorageModeShared)));
    return payload;
}

//std::vector<float> executeOnGPU(NS::SharedPtr<MTL::Device> device, NS::SharedPtr<MTL::ComputeCommandEncoder> encoder, NS::SharedPtr<MTL::Function> function, std::vector<float> a, std::vector<float> b, std::vector<float> &res) {
//    NS::Error* err = nil;
//    MTL::ComputePipelineState* pipelineState = device->newComputePipelineState(function.get(), &err);
////    NS::TransferPtr<MTL::ComputePipelineState> pipelineState = NS::TransferPtr<MTL::ComputePipelineState>(MTL::ComputePipelineState(function.get(), &res));
//    encoder->setComputePipelineState(pipelineState);
//    MTL::Size threadsPerThreadgroup = MTL::Size(1024,1024,1024);
//    
//    std::vector<float> concatArray = {};
//    for(auto el : a) {
//        concatArray.push_back(el);
//    }
//    
//    for(auto el : b) {
//        concatArray.push_back(el);
//    }
//        
//    encoder->dispatchThreadgroups(device->newBuffer(&concatArray, concatArray.size() * sizeof(float), MTL::ResourceOptions()), 1024, threadsPerThreadgroup);
//    return res;
//}
//

/*
 NS::Error* error = nullptr;
 NS::SharedPtr<MTL::ComputePipelineState> pipelineState = NS::TransferPtr<MTL::ComputePipelineState>(device->newComputePipelineState(function.get(), &error));
 
 CHECK_ERROR(error);
 computeCommandEncoder->setComputePipelineState(pipelineState.get());
 
 MTL::Size threadsPerThreadgroup = MTL::Size(1024, 1, 1);
 MTL::Size threadGroupsPerGrid = MTL::Size(1, 1, 1);
 
 computeCommandEncoder->dispatchThreadgroups(threadGroupsPerGrid, threadsPerThreadgroup);
 computeCommandEncoder->endEncoding();
 commandBuffer->commit();
 commandBuffer->waitUntilCompleted
 */

void executeOnGPU(NS::SharedPtr<MTL::Device> device, MTL::CommandBuffer* commandBuffer, NS::SharedPtr<MTL::Function> function, NS::SharedPtr<MTL::Buffer> bufferA, NS::SharedPtr<MTL::Buffer> bufferB, NS::SharedPtr<MTL::Buffer> bufferResult) {
    MTL::ComputeCommandEncoder* computeCommandEncoder = commandBuffer->computeCommandEncoder();
    
    computeCommandEncoder->setBuffer(bufferA.get(), 0, 0);
    computeCommandEncoder->setBuffer(bufferB.get(), 0, 1);
    computeCommandEncoder->setBuffer(bufferResult.get(), 0, 2);
    
    NS::Error* error = nullptr;
    NS::SharedPtr<MTL::ComputePipelineState> pipelineState = NS::TransferPtr<MTL::ComputePipelineState>(device->newComputePipelineState(function.get(), &error));
    
    CHECK_ERROR(error);
    computeCommandEncoder->setComputePipelineState(pipelineState.get());
    
    MTL::Size threadsPerThreadgroup = MTL::Size(1024, 1, 1);
    MTL::Size threadGroupsPerGrid = MTL::Size(1, 1, 1);
    
    computeCommandEncoder->dispatchThreadgroups(threadGroupsPerGrid, threadsPerThreadgroup);
    computeCommandEncoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    
    return;
}

void test() {
    NS::Error* testError = nullptr;
    CHECK_ERROR(testError);
}

int main(int argc, const char * argv[]) {
    test();
        
    std::cout << "Starting to run GPU calculations on Apple Sillicon" << "\n";
    
    NS::SharedPtr<MTL::Device> device = NS::TransferPtr<MTL::Device>(MTLCreateSystemDefaultDevice());
    NS::SharedPtr<MTL::Library> kernelLibrary = loadKernelLibary(device, KERNEL_SOURCE);
    
    NS::SharedPtr<MTL::Function> function = NS::TransferPtr<MTL::Function>(
            kernelLibrary->newFunction(NS::String::string("vec_add", NS::UTF8StringEncoding))
           );
    
    MTL::CommandQueue* commandQueue = device->newCommandQueue();
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    
    std::vector<float> a(1024, 1.0f);
    std::vector<float> b(1024, 2.0f);
    std::vector<float> result(1024, 0.0f);
    
    auto bufferA = floatVectorToDeviceBuffer(device, a);
    auto bufferB = floatVectorToDeviceBuffer(device, b);
    auto bufferResult = floatVectorToDeviceBuffer(device, result);
    
    executeOnGPU(device, commandBuffer, function, bufferA, bufferB, bufferResult);
    
    std::cout << "Calculated" << "\n";
    
    auto contents = bufferResult->contents();
    std::copy(static_cast<float*>(contents), static_cast<float*>(contents) + result.size(), result.begin());
    
    for(auto el : result) {
        std::cout << el << " ";
    }
    std::cout << "\n";
    

    return EXIT_SUCCESS;
}

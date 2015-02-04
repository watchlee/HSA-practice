// MP 1
#include <wb.h>
#include <math.h>
#include <cuda.h>

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)


/*----------Print All output data information-----------------------*/
void revel(float *data,int length)
{
	for(int count = 0;count < length;count++)
		printf("index = %d data = %f\n",count,data[count]);
}


__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
	
  //@@ Insert code to implement vector addition here
	int index = blockDim.x * blockIdx.x+ threadIdx.x;
	while(index < len)
	{
		out[index] = in1[index]+in2[index];
		/*in case if input data is big than thread blocks*/
		index+=gridDim.x;
	}
		
	
}

/*-----------------CPU version----------------------*/
void vectorAdd(float *in1,float *in2,float *out,int len)
{
	for(int count = 0;count < len;count++)
	{
		out[count] = in1[count]+in2[count];
	}
}


int main(int argc, char **argv) {
	
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;
  float *temp;
  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 = ( float * )wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 = ( float * )wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = ( float * )malloc(inputLength * sizeof(float));
	
/*----------------------------test CPU version------------------------------*/	
  /*
  temp = (float *)malloc(inputLength*sizeof(float));
  vectorAdd(hostInput1,hostInput2,temp,inputLength);
  revel(temp,inputLength);
  printf("\n");
  */
  
/*----------------------------test CPU version------------------------------*/

	
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  int size = inputLength *sizeof(float);
  wbCheck(cudaMalloc((void**)&deviceInput1,size));
  wbCheck(cudaMalloc((void**)&deviceInput2,size));
  wbCheck(cudaMalloc((void**)&deviceOutput,size));
	
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
	
  printf("length = %d\n",inputLength);
  cudaMemcpy(deviceInput1,hostInput1,size,cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2,hostInput2,size,cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 grid(ceil(inputLength/256.0),1,1);
  dim3 block(256,1,1);
	
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  
  vecAdd<<<grid,block>>>(deviceInput1,deviceInput2,deviceOutput,inputLength);		
  cudaDeviceSynchronize();
  
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput,deviceOutput,size,cudaMemcpyDeviceToHost);
	
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  //revel(hostOutput,inputLength);
  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}

#include <wb.h>

#define DEBUG_ENABLE 0
#define GPU_ENABLE 1


#define wbCheck(stmt)                                                          \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                              \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));           \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#define THREAD 32

#define MAX(size)\
	(size-1)/THREAD+1
 


// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows, int numBColumns,
                               int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  /*I can't believe that "printf" could be used on GPUDevice  . . . .lol*/
  printf("col = %d row = %d\n",col,row);
  if(row < numARows&&col<numBColumns)
  {
	  float sum = 0.0;
	  for(int loop = 0;loop<numAColumns;loop++)
	  {
		 sum+=A[row*numAColumns + loop]*B[col+loop*numBColumns];
	  }
	  C[row*numBColumns+col] = sum;
		  
  }
  
  
}

//CPU_Version Matrix x Matrix
void CPU_matrix(float *A,float *B,float *C,int rowA,int columnA,int rowB,int columnB,
				int rowC,int columnC)
{
	
	for(int Row = 0;Row < rowA;Row++)
	{
		for(int Col = 0;Col < columnB;Col++)
		{
			float sum = 0.0;
			for(int loop = 0;loop<columnA;loop++)
			{
				float a = A[Row*columnA+loop];
				float b = B[Col+loop*columnB];
				if(Row<10&&Col<10)
				printf(" A[%d]+B[%d]\n",Row*columnA+loop,Col+loop*columnB);
				sum+=a*b;
				
			}
			C[Row*columnB + Col] = sum;
		}
	}
}


void revel_result(float *hostC,int row,int column)
{
	for(int out_loop = 0;out_loop <row;out_loop++)
		for(int inner_loop = 0;inner_loop<column;inner_loop++)
			printf("C[%d][%d] = %f\n",out_loop,inner_loop,hostC[out_loop*column+inner_loop]);
}


int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set this)


  
	
	
  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
	
 
  hostA =
      ( float * )wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
//size_A be used to record C's matrix size
  int size_A = numARows*numAColumns*sizeof(float);
  
  hostB =
      ( float * )wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
//size_B be used to record C's matrix size
  int size_B = numBRows * numBColumns*sizeof(float);
	
  //@@ Set numCRows and numCColumns
  // row X column ==  A's row x B's column
  
  numCRows = numARows;
  numCColumns = numBColumns;
//size_C be used to record C's matrix size
  int size_C = numCRows * numCColumns*sizeof(float);
	
  #if DEBUG_ENABLE
	printf("C's row = %d , C's column = %d size_A = %d size_B = %d size_C = %d\n"
		   ,numCRows,numCColumns,size_A,size_B,size_C);
  #endif
	
  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(size_C );
	
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  #if GPU_ENABLE
  wbCheck(cudaMalloc((void**)&deviceA,size_A));
  wbCheck(cudaMalloc((void**)&deviceB,size_B));
  wbCheck(cudaMalloc((void**)&deviceC,size_C));
  #endif

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  #if GPU_ENABLE
  cudaMemcpy(deviceA,hostA,size_A,cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB,hostB,size_B,cudaMemcpyHostToDevice);
  #endif	
	
  
  #if DEBUG_ENABLE
  CPU_matrix(hostA,hostB,hostC,numARows,numAColumns,numBRows,numBColumns,
				numCRows,numCColumns);
  #endif
  
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  #if GPU_ENABLE
  dim3 grids(MAX(numARows),MAX(numBColumns),1);
  dim3 blocks(THREAD,THREAD,1);
  #endif
  
  wbTime_start(Compute, "Performing CUDA computation");
	
	
  //@@ Launch the GPU Kernel here
  #if GPU_ENABLE
  matrixMultiply<<<grids,blocks>>>(deviceA,deviceB,deviceC,numARows,numAColumns,
								   numBRows,numBColumns,numCRows,numCColumns);
  cudaDeviceSynchronize();
  #endif
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
	
  //@@ Copy the GPU memory back to the CPU here
  #if GPU_ENABLE
  cudaMemcpy(hostC,deviceC,size_C,cudaMemcpyDeviceToHost);
  #endif
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  #if GPU_ENABLE
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  revel_result(hostC,numCRows,numCColumns);
  #endif
	
  wbTime_stop(GPU, "Freeing GPU Memory");
  #if DEBUG_ENABLE
  revel_result(hostC,numCRows,numCColumns);
  #endif
	
  wbSolution(args, hostC, numCRows, numCColumns);
  
  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}


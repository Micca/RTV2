#include <math.h>
#include <algorithm>
#include <string>
//////////////////////////////////////////////////////////////////
///// CUDA IMPLEMENTATION
//////////////////////////////////////////////////////////////////

#define TILESIZE 4096
#define BLOCKSIZE 1024

__device__ // runs on GPU, callable by threads on GPU only
float isotropicGaussKernel2DCUDA(float x, float y, float center_x, float center_y, float sigma)
{
	register float xToCenterX = x - center_x;
	register float yToCenterY = y - center_y;
	register float scaleFactor = 0.5f/(sigma*sigma);

	return expf(-(xToCenterX*xToCenterX*scaleFactor + yToCenterY*yToCenterY*scaleFactor));
}

__global__ // runs on GPU, callable by host CPU
void KDEEstimator2DCUDAKernel(const float *observationsX, const float *observationsY, size_t numObservations,
	float sigma, float minX, float maxX, float minY, float maxY, float *kde_image, size_t numBins)
{
	//int TILESIZE = 256;
	//	register float xr[2];
	__shared__ float yd[TILESIZE][2];

	int threadGridIdx = blockIdx.x * blockDim.x + threadIdx.x;

	// get 2D bin position from 1D thread grid index
	float rangeX = maxX - minX; // min-max range of observed values for random variable X
	float rangeY = maxY - minY; // min-max range of observed values for random variable Y
	int binX = threadGridIdx % numBins;
	int binY = threadGridIdx / numBins;
	float x = float(binX) / (numBins - 1) * rangeX + minX; // bins mapped linearly to values between min-max observations
	float y = float(binY) / (numBins - 1) * rangeY + minY; // bins mapped linearly to values between min-max observations

	kde_image[threadGridIdx] = 0.0f;
	

	for (int j = 0; (j + threadIdx.x) < numObservations; j += TILESIZE) {
		for(int i = 0; i < TILESIZE && threadIdx.x + i + j < numObservations; i += BLOCKSIZE){
		yd[threadIdx.x + i][0] = observationsX[threadIdx.x + i + j];
		yd[threadIdx.x + i][1] = observationsY[threadIdx.x + i + j];
		}
		__syncthreads();

		int t = TILESIZE;
		if (j + TILESIZE > numObservations) {
			t = numObservations - j;
		}

		//if ((threadIdx.x + j) % TILESIZE == 0) printf("blocknr: %d, observation nr: %d\n", blockIdx.x, (threadIdx.x + j));

		for (int jj = 0; jj < t; jj++) {
			//kde_image[threadGridIdx] += isotropicGaussKernel2DCUDA(xr[0], xr[1], yd[jj][0], yd[jj][1], sigma);
			kde_image[threadGridIdx] += isotropicGaussKernel2DCUDA(x, y, yd[jj][0], yd[jj][1], sigma);
		}
	}

	/*for (size_t i = 0; i < numObservations; ++i) {
	kde_image[threadGridIdx] += isotropicGaussKernel2DCUDA(x, y, observationsX[i], observationsY[i], sigma);
	}*/
}

void chkCudaStatus(cudaError_t status, std::string errormsg) {
	if (status != cudaSuccess) {
		std::string msg = errormsg + ": error code %d\n";
		fprintf(stderr, msg.c_str(), cudaGetErrorString(status));
	}
}

extern "C"
float KDEEstimator2D(const float *observationsX, const float *observationsY, size_t numObservations, float sigma, float minX, float maxX, float minY, float maxY, float *kde_image, size_t numBins)
{

	cudaError_t cudaStatus;

	// allocate device data vectors in GPU global memory
	float *d_observationsX = NULL, *d_observationsY = NULL, *d_kde_image = NULL;
	cudaMalloc((void **)&d_observationsX, numObservations * sizeof(float)); // pointer to our array pointer
	cudaMalloc((void **)&d_observationsY, numObservations * sizeof(float));
	cudaStatus = cudaMalloc((void **)&d_kde_image, numBins*numBins * sizeof(float)); // resulting kde image will be stored as linearly indexed array
	chkCudaStatus(cudaStatus, "Memory allocation failed");

	// copy data to allocated device memory
	// first parameter of cudaMemcpy is destination
	cudaMemcpy(d_observationsX, observationsX, numObservations * sizeof(float), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(d_observationsY, observationsY, numObservations * sizeof(float), cudaMemcpyHostToDevice);
	chkCudaStatus(cudaStatus, "Memory copy failed");

	// launch CUDA kernel
	int numThreads = numBins*numBins; // each thread does kernel summation (pdf estimation) for one bin, i.e. one value pair (x,y)
	int numBlocks = (numThreads + BLOCKSIZE - 1) / BLOCKSIZE; // grid size, round up if numElems not multiple of block size

	KDEEstimator2DCUDAKernel <<<numBlocks, BLOCKSIZE >>>(d_observationsX, d_observationsY, numObservations,
	sigma, minX, maxX, minY, maxY, d_kde_image, numBins);
	cudaStatus = cudaGetLastError();
	chkCudaStatus(cudaStatus, "Kernel failed");
	// wait until all threads return
	cudaDeviceSynchronize();


	// copy resulting data back to host
	cudaStatus = cudaMemcpy(kde_image, d_kde_image, numBins*numBins * sizeof(float), cudaMemcpyDeviceToHost);
	chkCudaStatus(cudaStatus, "Retrieving result failed");
	// free memory
	// make sure to free in exact reverse order as allocation!

	cudaFree(d_kde_image);
	cudaFree(d_observationsY);
	cudaStatus = cudaFree(d_observationsX);
	chkCudaStatus(cudaStatus, "Freeing resources failed");

	// determine highest sum of all bins (values of X and values of Y are compared equally)
	// for normalization to get probability
	float maxIntensity = 0.0f;
	maxIntensity = *std::max_element(kde_image, kde_image + (numBins*numBins)); // yields pointer to variable with max value, we want the value

	/*/ normalization slow!! TODO find better way to normalize!!
	for (size_t i = 0; i < numBins*numBins; ++i) {
	kde_image[i] /= maxIntensity;
	}//*/

	return maxIntensity;
}

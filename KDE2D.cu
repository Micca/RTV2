#include <math.h>
#include <algorithm>
#include <string>
//////////////////////////////////////////////////////////////////
///// CUDA IMPLEMENTATION
//////////////////////////////////////////////////////////////////

#define num_var 2
#define threadsPerBlock 256

__device__ // runs on GPU, callable by threads on GPU only
float isotropicGaussKernel2DCUDA(float x, float y, float center_x, float center_y, float sigma)
{
	register float xToCenterX = x - center_x;
	register float yToCenterY = y - center_y;
	register float scaleFactor = 0.5f/(sigma*sigma);

	return expf(-(xToCenterX*xToCenterX*scaleFactor + yToCenterY*yToCenterY*scaleFactor));
}

__global__ // runs on GPU, callable by host CPU
void KDEEstimator2DCUDAKernel(const float *observationsX, const float *observationsY, size_t numObservations, float sigma, float minX, float maxX, float minY, float maxY, float *kde_image, size_t numBins)
{
    // TODO

	// we have two random variables X and Y (2D), for each we have some observed values x_observations and y_observations
	// the observations give us a rough histogram / rough approximation of the probability distribution function (PDF),
	// i.e. how often does each value occur? but only for the observed values.
	// we want to estimate the PDF using kernel density estimation (KDE) for other values x, y of our random variables X, Y,
	// on a regular spacing of numBins bins between the min and max observed values.

	// the grid size (number of threads) corresponds to kde_image size, i.e. numBins*numBins, however we use a 1D grid and also image
	// each thread does kernel summation (pdf estimation) for one value pair (x,y), i.e. one bin of the 2D value range
	int threadGridIdx = blockIdx.x * blockDim.x + threadIdx.x; // global thread index in grid of blocks

	// since number of elements might not be a multiple of block size, check to avoid threads running on empty data
	// actually in this case this is not necessary since kde_image size = number of elements = numBins*numBins
	if (threadGridIdx < numBins*numBins) {

		// get 2D bin position from 1D thread grid index
		float rangeX = maxX - minX; // min-max range of observed values for random variable X
		float rangeY = maxY - minY; // min-max range of observed values for random variable Y
		int binX = threadGridIdx % numBins;
		int binY = threadGridIdx / numBins;
		float x = float(binX)/(numBins-1) * rangeX + minX; // bins mapped linearly to values between min-max observations
		float y = float(binY)/(numBins-1) * rangeY + minY; // bins mapped linearly to values between min-max observations

		// model a gaussian probability distribution around each observation
		// estimate probability density for each value pair x, y by accumulating contributions from all the gaussian kernels
		kde_image[threadGridIdx] = 0.0f;
		for (size_t i = 0; i < numObservations; ++i) {
			kde_image[threadGridIdx] += isotropicGaussKernel2DCUDA(x, y, observationsX[i], observationsY[i], sigma);
		}

	}

}


__global__
void optimizedKDE2D(const float *observationsX, const float *observationsY, size_t numObservations, float sigma, float minX, float maxX, float minY, float maxY, float *kde_image_opt, size_t numBins)
{

	int bsize = blockDim.x;
	register float xr[num_var];
	__shared__ float yd[500][num_var];
	int threadGridIdx = blockDim.x * blockIdx.x + threadIdx.x;

	if (threadGridIdx < numBins*numBins) {

		// get 2D bin position from 1D thread grid index
		float rangeX = maxX - minX; // min-max range of observed values for random variable X
		float rangeY = maxY - minY; // min-max range of observed values for random variable Y
		int binX = threadGridIdx % numBins;
		int binY = threadGridIdx / numBins;
		xr[0] = float(binX) / (numBins - 1) * rangeX + minX; // bins mapped linearly to values between min-max observations
		xr[1] = float(binY) / (numBins - 1) * rangeY + minY; // bins mapped linearly to values between min-max observations

		// fill yd shared memory cooperatively
		for (int i = 0; i < numObservations; i += threadsPerBlock) {
			if (i + threadIdx.x < numObservations) {
				yd[i + threadIdx.x][0] = observationsX[i + threadIdx.x];
				yd[i + threadIdx.x][1] = observationsY[i + threadIdx.x];
			}
		}

		// wait until all other threads in block loaded their data
		__syncthreads();

		float sum_ker = 0.0;
		kde_image_opt[threadGridIdx] = 0.0f;
		for (int j = 0; j < numObservations; ++j) {
			sum_ker += isotropicGaussKernel2DCUDA(xr[0], xr[1], yd[j][0], yd[j][1], sigma);
		}
		kde_image_opt[threadGridIdx] = sum_ker; // / (double)numObservations;
	}
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
	float *d_observationsX = NULL, *d_observationsY = NULL, *d_kde_image = NULL, *d_kde_image_opt = NULL;
	cudaMalloc((void **) &d_observationsX, numObservations*sizeof(float)); // pointer to our array pointer
	cudaMalloc((void **) &d_observationsY, numObservations*sizeof(float));
	cudaStatus = cudaMalloc((void **)&d_kde_image, numBins*numBins * sizeof(float)); // resulting kde image will be stored as linearly indexed array
	chkCudaStatus(cudaStatus, "Memory allocation failed");

	// copy data to allocated device memory
	// first parameter of cudaMemcpy is destination
	cudaMemcpy(d_observationsX, observationsX, numObservations*sizeof(float), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(d_observationsY, observationsY, numObservations*sizeof(float), cudaMemcpyHostToDevice);
	chkCudaStatus(cudaStatus, "Memory copy failed");

	// launch CUDA kernel
	int numThreads = numBins*numBins; // each thread does kernel summation (pdf estimation) for one bin, i.e. one value pair (x,y)
	int numBlocks = (numThreads+threadsPerBlock-1)/threadsPerBlock; // grid size, round up if numElems not multiple of block size
	optimizedKDE2D <<<numBlocks, threadsPerBlock>>>(d_observationsX, d_observationsY, numObservations,
	                                                         sigma, minX, maxX, minY, maxY, d_kde_image, numBins);
	cudaStatus = cudaGetLastError();
	chkCudaStatus(cudaStatus, "Kernel1 failed");
	// wait until all threads return
	cudaDeviceSynchronize();
	
	/*KDEEstimator2DCUDAKernel <<<numBlocks, threadsPerBlock>>>(d_observationsX, d_observationsY, numObservations,
		sigma, minX, maxX, minY, maxY, d_kde_image, numBins);
	cudaStatus = cudaGetLastError();
	chkCudaStatus(cudaStatus, "Kernel2 failed");

	// wait until all threads return
	cudaDeviceSynchronize();*/

	
	// copy resulting data back to host
	cudaStatus = cudaMemcpy(kde_image, d_kde_image, numBins*numBins*sizeof(float), cudaMemcpyDeviceToHost);
	chkCudaStatus(cudaStatus, "Retrieving result failed");
	// free memory
	// make sure to free in exact reverse order as allocation!
	
	cudaFree(d_kde_image_opt);
	cudaFree(d_kde_image);
	cudaFree(d_observationsY);
	cudaStatus = cudaFree(d_observationsX);
	chkCudaStatus(cudaStatus, "Freeing resources failed");

	// determine highest sum of all bins (values of X and values of Y are compared equally)
	// for normalization to get probability
	float maxIntensity = 0.0f;
	maxIntensity = *std::max_element(kde_image, kde_image+(numBins*numBins)); // yields pointer to variable with max value, we want the value

	/*/ normalization slow!! TODO find better way to normalize!!
	for (size_t i = 0; i < numBins*numBins; ++i) {
		kde_image[i] /= maxIntensity;
	}//*/

	return maxIntensity;
}

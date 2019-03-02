#include "Search.cuh"
#include <ctime>

int main(int argc, char *argv[]) {
	if (argc != 2) {
		printf("Invalid arguments\n");
		return -1;
	}

	//load genome into device memory
	size_t genome_size;
	char* gpu_genome = loadGenome(&genome_size);

	//create results array
	long *results;
	int *resultCount;
	cudaError_t err = cudaMallocManaged(&results, sizeof(long) * MAX_RESULT_SIZE);
	if (err != cudaSuccess) printf("On Malloc Result: %s\n", cudaGetErrorString(err));
	for (int i = 0; i < MAX_RESULT_SIZE; i++) {
		results[i] = -1;
	}
	err = cudaMallocManaged(&resultCount, sizeof(int));
	if (err != cudaSuccess) printf("On Malloc Result Count: %s\n", cudaGetErrorString(err));
	*resultCount = 0;

	//generate query mask
	QueryMask queryMask;
	generateQueryMask(argv[1], &queryMask);
	if (queryMask.maskCount != 1) {
		printf("QUERIES THIS LARGE ARE NOT YET SUPPORTED!! RESULTS ARE UNDEFINED\n");
	}

	//preform search
	clock_t before = clock();
	search <<<1, 1000 >>> (queryMask, gpu_genome, genome_size, results, resultCount);
	err = cudaGetLastError();
	if (err != cudaSuccess) printf("On Run: %s\n", cudaGetErrorString(err));
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) printf("On Sync: %s\n", cudaGetErrorString(err));
	clock_t after = clock();
	int ms = ((float)after - before) / CLOCKS_PER_SEC * 1000;
	printf("Runtime: %d ms\n", ms);

	//print results
	for(int i = 0; i < *resultCount; i ++) {
	  long result = results[i];
	  if(result != -1) {
	    printf("%d, ", result);
	  }
	}
	printf("\nResult Count: %d\n", *resultCount);

	//clean
	cudaFree(queryMask.matchMask);
	cudaFree(gpu_genome);
	cudaFree(results);

	err = cudaGetLastError();
	if (err != cudaSuccess) printf("On End: %s\n", cudaGetErrorString(err));

	return 0;
}
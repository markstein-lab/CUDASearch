#include "Search.cuh"

__global__ void search(QueryMask queryMask, char *genome, size_t genome_size, long *results, int *resultCount) {
	long start = (genome_size * threadIdx.x) / blockDim.x;
	long end = (genome_size * (threadIdx.x + 1)) / blockDim.x + queryMask.queryLength;
	// int resultsIndex = (MAX_RESULT_SIZE * threadIdx.x) / blockDim.x;
	// int resultsIndexEnd = (MAX_RESULT_SIZE * (threadIdx.x + 1)) / blockDim.x;
	if (end > genome_size) {
		end = genome_size;
	}

	MASK_TYPE matchState = 0;
	for (long i = start; i < end; i++) {
		//Get the next 4 letters
		char g = genome[i];
		for (int b = 0; b < 4; b++) {
			//extract current letter
			char letter = (g >> ((3 - b) * 2)) & 3;
			//shift right 1
			matchState = matchState >> 1;
			//set MSB
			matchState |= (1 << (queryMask.queryLength - 1));
			//AND-in M
			matchState &= queryMask.matchMask[letter];
			if (matchState & 1 == 1) {
				int rc = atomicAdd(resultCount, 1);
				if (*resultCount >= MAX_RESULT_SIZE) {
					return;
				}
				 results[rc] = i*4 + b - queryMask.queryLength;
			}
		}
	}
}

char* loadGenome(size_t *genome_size) {
	char *gpu_genome;
	std::ifstream genomeFile(GENOME_FILE, std::ifstream::binary | std::ifstream::ate);
	*genome_size = genomeFile.tellg();
	genomeFile.seekg(0, std::ifstream::beg);
	cudaMallocManaged(&gpu_genome, (*genome_size) * sizeof(char));
	genomeFile.read(gpu_genome, *genome_size);
	printf("Loaded genome of size: %zd\n", *genome_size);
	return gpu_genome;
}
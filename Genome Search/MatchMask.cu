#include "Search.cuh"

void generateQueryMask(char *query, QueryMask *queryMask) {  //Add in transpose of query
	size_t queryLength = std::strlen(query);
	size_t maskCount = (queryLength - 1) / MASK_SIZE + 1;
	MASK_TYPE* matchMask;
	cudaError_t err = cudaMallocManaged(&matchMask, 4 * sizeof(MASK_TYPE) * maskCount);
	if (err != cudaSuccess) printf("On Malloc Query Mask: %s\n", cudaGetErrorString(err));
	size_t start_index = (maskCount * MASK_SIZE) - queryLength;//location of first active byte
	for (int i = start_index; i < queryLength + start_index; i++) {
		int maskX = i / MASK_SIZE;  //The current char being worked on
		matchMask[0 * maskCount + maskX] <<= 1;
		matchMask[1 * maskCount + maskX] <<= 1;
		matchMask[2 * maskCount + maskX] <<= 1;
		matchMask[3 * maskCount + maskX] <<= 1;
		char c = tolower(query[i - start_index]);
		switch (c) {
		case 'a':
			matchMask[0 * maskCount + maskX] |= 1;
			break;
		case 'c':
			matchMask[1 * maskCount + maskX] |= 1;
			break;
		case 'g':
			matchMask[2 * maskCount + maskX] |= 1;
			break;
		case 't':
			matchMask[3 * maskCount + maskX] |= 1;
			break;
		default:
			printf("Unrecognized query character: %c\n", c);
		}
	}
	queryMask->maskCount = maskCount;
	queryMask->queryLength = queryLength;
	queryMask->matchMask = matchMask;
}
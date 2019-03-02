#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <string>
#include <fstream>

#include <stdio.h>

#define GENOME_FILE "C:\\Users\\alexj\\Dropbox\\College\\8Spring2019 AAAAAAA\\lab\\Genomes\\Fruit Fly\\fruit_fly.gb"
// #define genomeFileName "C:\\Users\\alexj\\Downloads\\fly.fa"
#define MAX_RESULT_SIZE 100000

#define MASK_TYPE unsigned long int
#define MASK_SIZE (sizeof(MASK_TYPE) * 8)

struct QueryMask {
	size_t maskCount;
	size_t queryLength;
	MASK_TYPE* matchMask;
};

char* loadGenome(size_t *genome_size);

__global__ void search(QueryMask queryMask, char *genome, size_t genome_size, long *results, int *resultCount);

void generateQueryMask(char *query, QueryMask *queryMask);
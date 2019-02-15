/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2019 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: CUDAFastPBC.cu,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.2 $       $Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Fast PBC wrapping code.
 ***************************************************************************/
#include <stdio.h>

// XXX this needs to go
#include "sys/time.h"

// Uses thrust for vector ops, various scan() reductions, etc.
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/device_vector.h>

#include "FastPBC.h"

/// XXX globals need to go...
const int nStreams = 4;
const int threads = 128;

__global__ void inverseboxsize (float *boxsize, float* invboxsize) {
	int tid = threadIdx.x;
	if (tid < 3) {
		invboxsize[tid] = 1.0 / boxsize[tid];
	}
}


// This is an inefficient kernel. Much slower than the one that replaced 
// it below. (~100 us)
__global__ void repositionfragments(int fnum, float *pos, int *compoundmap, 
                                    int *indexlist, float *boxsize, 
                                    float *invboxsize) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int i, j, k;
	for (int l = tid; l < fnum; l+=blockDim.x * gridDim.x) {
		float ccenter[3];
		int lowbound = compoundmap[l];
		int highbound = compoundmap[l+1];
		i = indexlist[lowbound];
		//Use the first element within the compound as the center.
		for (j=0; j < 3; j++) {
			ccenter[j] = pos[i*3+j];
		}
		//move the compound, wrapping it to be within half a box dimension from the center
		for (k = lowbound; k < highbound; k++ ) {
			i = indexlist[k];
			for (j=0; j < 3; j++) {
				pos[i*3+j] = pos[i*3+j] - (rintf((pos[i*3+j] - ccenter[j]) * invboxsize[j]) * boxsize[j]);
			}
		}
	}
}


// Super-efficient kernel. ~8 us execution time
__global__ void repositionfragments(float *pos, int sellen, int *atomtofragmap,
                                    int *compoundmap, int *indexlist, 
                                    float *boxsize, float *invboxsize) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < 3*sellen) {
		int idx = tid / 3;
		int dim = tid % 3;
		int cidx = indexlist[compoundmap[atomtofragmap[idx]]];
		//printf("tid: %d, idx %d, dim %d, cidx %d\n", tid, idx, dim, cidx);
		float center = pos[3 * cidx + dim];
		pos[3*indexlist[idx]+dim] = pos[3*indexlist[idx]+dim] - (rintf((pos[3*indexlist[idx]+dim] - center) * invboxsize[dim]) * boxsize[dim]);
	}
}


__global__ void wrapcompound(float *pos, int sellen, float *center, 
                             int *atomtofragmap, int *indexlist, 
                             float *boxsize, float *invboxsize, 
                             float *fragcenters) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < 3*sellen) {
		int idx = tid / 3;
		int dim = tid % 3;
		int frag = atomtofragmap[idx];
		int aidx = indexlist[idx];
		pos[3*aidx+dim] = pos[3*aidx+dim] - (rintf((fragcenters[3*frag+dim] - center[dim]) * invboxsize[dim]) * boxsize[dim]);
	}
}


__global__ void wrapatomic(float *pos, int sellen, float *center, 
                           int *indexlist, float *boxsize, float *invboxsize) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < 3*sellen) {
		int idx = tid / 3;
		int dim = tid % 3;
		int aidx = indexlist[idx];
		pos[3*aidx+dim] = pos[3*aidx+dim] - (rintf((pos[3*aidx+dim] - center[dim]) * invboxsize[dim]) * boxsize[dim]);
	}
}


__global__ void unwrapatomic(float *pos, float *prev, int sellen, 
                             int *indexlist, 
                             float *boxsize, float *invboxsize) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < 3*sellen) {
		int idx = tid / 3;
		int dim = tid % 3;
		int aidx = indexlist[idx];
		pos[3*aidx+dim] = pos[3*aidx+dim] - (rintf((pos[3*aidx+dim] - prev[3*aidx+dim]) * invboxsize[dim]) * boxsize[dim]);
	}
}


__global__ void fragmentperatom(int fnum, int *compoundmap, 
                                int *atomtofragmap) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < fnum && tid != 0) {
		atomtofragmap[compoundmap[tid]] = 1;
	}
}


// XXX this likely duplicates stuff from prototypes in CUDAMeasure
__global__ void measurecenter(float *pos, float *center, int len, 
                              float *weights, int *weightidx, float *wscale) {
	__shared__ float reduce[96]; //96 is not an arbitrary number. Its divisible by 3! This lets us use
	//aligned memory accesses.
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	float mcenter = 0;
	int dim;
	if (tid < 3*len) {
		int idx = tid / 3;
		dim = tid % 3;
		mcenter = pos[3 * weightidx[idx] + dim] * weights[idx] * (*wscale);
	}
	reduce[threadIdx.x] = mcenter;
	__syncthreads();
	if (threadIdx.x < 48) {
		reduce[threadIdx.x] += reduce[threadIdx.x + 48];
	}
	__syncthreads();
	if (threadIdx.x < 24) {
		reduce[threadIdx.x] += reduce[threadIdx.x + 24];
	}
	__syncthreads();
	if (threadIdx.x < 12) {
		reduce[threadIdx.x] += reduce[threadIdx.x + 12];
	}
	__syncthreads();
	if (threadIdx.x < 3) {
		mcenter = reduce[threadIdx.x] + reduce[threadIdx.x + 3] + reduce[threadIdx.x + 6] + reduce[threadIdx.x + 9];
		atomicAdd(&center[dim], mcenter);
	}
}


// Only differs from measurecenter based on how the weights are indexed.
// Here the expectation is that the full mass array has been passed, 
// so we need to find only specific elements of the weight array.
__global__ void measurecenter_fullmass(float *pos, float *center, int len, 
                                       float *weights, int *weightidx, 
                                       float *wscale) {
	__shared__ float reduce[96]; //96 is not an arbitrary number. Its divisible by 3! This lets us use
	//aligned memory accesses.
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	float mcenter = 0;
	int dim;
	if (tid < 3*len) {
		int idx = tid / 3;
		dim = tid % 3;
		int widx = weightidx[idx];
		mcenter = pos[3 * widx + dim] * weights[widx] * (*wscale);
	}
	reduce[threadIdx.x] = mcenter;
	__syncthreads();
	if (threadIdx.x < 48) {
		reduce[threadIdx.x] += reduce[threadIdx.x + 48];
	}
	__syncthreads();
	if (threadIdx.x < 24) {
		reduce[threadIdx.x] += reduce[threadIdx.x + 24];
	}
	__syncthreads();
	if (threadIdx.x < 12) {
		reduce[threadIdx.x] += reduce[threadIdx.x + 12];
	}
	__syncthreads();
	if (threadIdx.x < 3) {
		mcenter = reduce[threadIdx.x] + reduce[threadIdx.x + 3] + reduce[threadIdx.x + 6] + reduce[threadIdx.x + 9];
		atomicAdd(&center[dim], mcenter);
	}
}


// Harrumph. This kernel is inefficient. Less inefficient than the prettier 
// way of doing it, but this has only 1 kernel call, whereas the other one 
// had as many calls as there were fragments.
__global__ void computefragcenters(float *pos, float *centers, int fnum, 
                                   float *weights, float *wscale, 
                                   int *compoundmap, int *indexlist) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int i, j, k, f;
	float ccenter = 0;
	if (tid < 3*fnum) {
		f = tid / 3;
		j = tid % 3;
		int lowbound = compoundmap[f];
		int highbound = compoundmap[f+1];
		//Find the center of the compound.
		for (k = lowbound; k < highbound; k++ ) {
			i = indexlist[k];
			ccenter += pos[i*3+j] * weights[i] * wscale[f];
		}
		centers[3*f+j] = ccenter ;
	}
}


__global__ void fragwscale(float *fragscales, float *prefixsums, int fragnum, 
                           int *compoundmap) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < fragnum) {
		fragscales[tid] = 1.0 / (prefixsums[compoundmap[tid+1]] - prefixsums[compoundmap[tid]]);
	}
}


// XXX this is redundant in normal VMD builds, so we'll chop it later...
class Timer {
public:
	timeval t;
	Timer() {
		gettimeofday(&t, NULL);
	}
	~Timer() {
		timeval theend;
		gettimeofday(&theend,NULL);
		double elapsedTime;
		elapsedTime = (theend.tv_sec - t.tv_sec) * 1000000.0;  //sec to us
		elapsedTime += (theend.tv_usec - t.tv_usec);
		printf("Time: %.f us\n", elapsedTime);
	}
};


void fpbc_exec_unwrap(Molecule* mol, int first, int last, int sellen, int* indexlist) {
	Timestep *ts;
	int f;
	float *pos;
	float *gpupos[nStreams];
	float boxsize[3];
	float *gpuboxsize[nStreams];
	float *gpuinvboxsize[nStreams];
	int *gpuindexlist;
	cudaStream_t stream[nStreams];
	cudaEvent_t events[nStreams];
	int blocks = (3*sellen + threads - 1) / threads;
	cudaHostRegister(indexlist, sizeof(int) * sellen,0);
	cudaHostRegister(boxsize, sizeof(float) * 3,0);
	for (f = 0; f< nStreams; f++) {
		cudaStreamCreate(&stream[f]);
		cudaEventCreateWithFlags(&events[f], cudaEventDisableTiming);
		cudaMalloc((void**) &gpupos[f], sizeof(float) * 3*mol->nAtoms);
		cudaMalloc((void**) &gpuboxsize[f], sizeof(float) * 3);
		cudaMalloc((void**) &gpuinvboxsize[f], sizeof(float) * 3);
	}
	cudaMalloc((void**) &gpuindexlist, sizeof(int) * sellen);
	cudaMallocHost(&pos, nStreams * sizeof(float) * 3*mol->nAtoms);
	cudaMemcpy(gpuindexlist, indexlist, sizeof(int) * sellen, cudaMemcpyHostToDevice);
	f = first;
	ts = mol->get_frame(f);
	memcpy(&pos[(f%nStreams)*3*mol->nAtoms], ts->pos, sizeof(float) * 3*mol->nAtoms);
	cudaMemcpyAsync(gpupos[f%nStreams], &pos[(f%nStreams)*3*mol->nAtoms], sizeof(float) * 3*mol->nAtoms, cudaMemcpyHostToDevice, stream[f%nStreams]);
	cudaEventRecord(events[f%nStreams],stream[f%nStreams]);
	//Do stuff
	for (f=first+1; f<=last; f++) {
		ts = mol->get_frame(f);
		boxsize[0] = ts->a_length;
		boxsize[1] = ts->b_length;
		boxsize[2] = ts->c_length;
		//Block here just so that I don't overwrite a buffer.
		cudaStreamSynchronize(stream[f%nStreams]);
		if (! (f-(first+1) < nStreams)) {
			memcpy(mol->get_frame(f-nStreams)->pos, &pos[(f%nStreams)*3*mol->nAtoms], sizeof(float) * 3*mol->nAtoms);
		}
		memcpy(&pos[(f%nStreams)*3*mol->nAtoms], ts->pos, sizeof(float) * 3*mol->nAtoms);
		cudaMemcpyAsync(gpupos[f%nStreams], &pos[(f%nStreams)*3*mol->nAtoms], sizeof(float) * 3*mol->nAtoms, cudaMemcpyHostToDevice, stream[f%nStreams]);
		cudaMemcpyAsync(gpuboxsize[f%nStreams], boxsize, sizeof(float) * 3, cudaMemcpyHostToDevice, stream[f%nStreams]);
		//Do math here.
		inverseboxsize<<<1,4,0,stream[f%nStreams]>>>(gpuboxsize[f%nStreams], gpuinvboxsize[f%nStreams]);
		//We must wait until the previous stream is done moving atoms around or loading. This part is inherently serial.
		cudaStreamWaitEvent(stream[f%nStreams], events[(f-1)%nStreams],0);
		unwrapatomic<<<blocks,threads,0,stream[f%nStreams]>>>(gpupos[f%nStreams], gpupos[(f-1)%nStreams], sellen, gpuindexlist, gpuboxsize[f%nStreams], gpuinvboxsize[f%nStreams]);
		cudaEventRecord(events[f%nStreams],stream[f%nStreams]);
		//Copy out.
		cudaMemcpyAsync(&pos[(f%nStreams)*3*mol->nAtoms], gpupos[f%nStreams], sizeof(float) * 3*mol->nAtoms, cudaMemcpyDeviceToHost,stream[f%nStreams]);
	}
	//Copy back the remaining elements.
	for (f = max(last - nStreams + 1,0); f <= last; f++) {
		cudaStreamSynchronize(stream[f%nStreams]);
		memcpy(mol->get_frame(f)->pos, &pos[(f%nStreams)*3*mol->nAtoms], sizeof(float) * 3*mol->nAtoms);
	}
	//Cleanup
	for (f = 0; f< nStreams; f++) {
		cudaStreamDestroy(stream[f]);
		cudaEventDestroy(events[f]);
		cudaFree(gpupos[f]);
		cudaFree(gpuboxsize[f]);
		cudaFree(gpuinvboxsize[f]);
	}
	cudaHostUnregister(indexlist);
	cudaHostUnregister(boxsize);
	cudaFree(gpuindexlist);
	cudaFreeHost(pos);
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		// print the CUDA error message and fallback to CPU
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		printf("Reverting to CPU algorithm\n");
		fpbc_exec_unwrap_cpu(mol, first, last, sellen, indexlist);
	}
}


void fpbc_exec_wrapcompound(Molecule* mol, int first, int last, int fnum, int *compoundmap, int sellen, int* indexlist, float* weights, AtomSel* csel, float* center, float* massarr) {
	//Declare variables.
	int f, i, j;
	Timestep *ts;
	float *pos;
	float *gpupos[nStreams];
	float boxsize[3];
	float *gpuboxsize[nStreams];
	float *gpucenters[nStreams];
	float *gpuinvboxsize[nStreams];
	float *gpuweights;
	float *wscale;
	float *gpufragweight;
	float *gpuwscan;
	int *gpuweightidx;
	int *gpuindexlist;
	int *gpuatomtofragmap;
	int *gpucompoundmap;
	float *gpufragcenters[nStreams];
	int blocks_frag = (fnum + threads - 1) / threads;
	cudaStream_t stream[nStreams];
	//Allocate memory for static things (weight sums, maps, etc.)
	cudaMalloc((void**) &gpuatomtofragmap, sizeof(int) * sellen);
	cudaMalloc((void**) &wscale, sizeof(float));
	cudaMemset(gpuatomtofragmap, 0, sizeof(int) * sellen);
	cudaMalloc((void**) &gpucompoundmap, sizeof(int) * (fnum+1));
	cudaMalloc((void**) &gpuindexlist, sizeof(int) * sellen);
	cudaMalloc((void**) &gpuweights, sizeof(float) * mol->nAtoms);
	cudaMalloc((void**) &gpufragweight, sizeof(float) * fnum);
	cudaHostRegister(compoundmap, sizeof(int) * (fnum+1),0);
	cudaHostRegister(indexlist, sizeof(int) * sellen,0);
	cudaMemcpy(gpuweights, massarr, sizeof(float) * mol->nAtoms, cudaMemcpyHostToDevice);//Unlike in wrapatomic, we'll pass over the full mass array to the GPU, since odds are we'll need it.
	cudaMemcpy(gpucompoundmap, compoundmap, sizeof(int) * (fnum+1), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuindexlist, indexlist, sizeof(int) * sellen, cudaMemcpyHostToDevice);
	//Make the atomtofragmap by setting elements to 1 and then doing a scan.
	fragmentperatom<<<blocks_frag, threads>>>(fnum, gpucompoundmap, gpuatomtofragmap);//Setup the gpu per atom map.
	thrust::inclusive_scan(thrust::device_ptr<int>(gpuatomtofragmap), thrust::device_ptr<int>(gpuatomtofragmap + sellen), thrust::device_ptr<int>(gpuatomtofragmap));
	//Get the mass per fragment (for scaling/finding center of mass for everything.)
	thrust::device_vector<float> mass (thrust::device_ptr<float>(gpuweights), thrust::device_ptr<float>(gpuweights+mol->nAtoms));
	mass.push_back(0.0);//This zero is here so we can scan the weights efficiently.
	cudaMalloc((void**) &gpuwscan, sizeof(float) * (mol->nAtoms+1));
	thrust::exclusive_scan(mass.begin(), mass.end(), thrust::device_ptr<float>(gpuwscan), 0, thrust::plus<float>());
	fragwscale<<<blocks_frag, threads>>>(gpufragweight, gpuwscan, fnum, gpucompoundmap);
	cudaHostRegister(boxsize, sizeof(float)*3,0);
	cudaMallocHost(&pos, nStreams * sizeof(float) * 3*mol->nAtoms);
	if (csel != NULL) {
		cudaMalloc((void**) &gpuweightidx, sizeof(int) * csel->selected);
		int *weightidx = new int[csel->selected];
		j=0;
		for (i=csel->firstsel; i<=csel->lastsel; i++) {
			if (csel->on[i]) {
				weightidx[j++] = i;
			}
		}
		cudaMemcpy(gpuweightidx, weightidx, sizeof(int) * csel->selected, cudaMemcpyHostToDevice);
		thrust::device_vector<int> ids (thrust::device_ptr<int>(gpuweightidx), thrust::device_ptr<int>(gpuweightidx+csel->selected));
		float tmp = 1.0 / thrust::reduce(thrust::make_permutation_iterator(mass.begin(), ids.begin()),
			thrust::make_permutation_iterator(mass.end(), ids.end()), 0, thrust::plus<float>());
		cudaMemcpy(wscale, &tmp, sizeof(float), cudaMemcpyHostToDevice);
		delete [] weightidx;
	}
	//Allocate memory and create streams for per-frame changables.
	for (f = 0; f< nStreams; f++) {
		cudaStreamCreate(&stream[f]);
		cudaMalloc((void**) &gpupos[f], sizeof(float) * 3*mol->nAtoms);
		cudaMalloc((void**) &gpuboxsize[f], sizeof(float) * 3);
		cudaMalloc((void**) &gpuinvboxsize[f], sizeof(float) * 3);
		cudaMalloc((void**) &gpucenters[f], sizeof(float) * 3);
		cudaMemcpyAsync(gpucenters[f], center, sizeof(float)*3, cudaMemcpyHostToDevice, stream[f]);
		cudaMalloc((void**) &gpufragcenters[f], sizeof(float) * 3 * fnum);
	}
	cudaDeviceSynchronize();
	cudaFree(gpuwscan);
	//Start looping over the frames.
	int blocks = (3*sellen + threads - 1) / threads;
	blocks_frag = (3*fnum + threads - 1) / threads;
	for (f=first; f<=last; f++) {
		ts = mol->get_frame(f);
		boxsize[0] = ts->a_length;
		boxsize[1] = ts->b_length;
		boxsize[2] = ts->c_length;
		//Block here just so that I don't overwrite a buffer.
		cudaStreamSynchronize(stream[f%nStreams]);
		if (! (f-first < nStreams)) {
			memcpy(mol->get_frame(f-nStreams)->pos, &pos[(f%nStreams)*3*mol->nAtoms], sizeof(float) * 3*mol->nAtoms);
		}
		memcpy(&pos[(f%nStreams)*3*mol->nAtoms], ts->pos, sizeof(float) * 3*mol->nAtoms);
		cudaMemcpyAsync(gpupos[f%nStreams], &pos[(f%nStreams)*3*mol->nAtoms], sizeof(float) * 3*mol->nAtoms, cudaMemcpyHostToDevice, stream[f%nStreams]);
		cudaMemcpyAsync(gpuboxsize[f%nStreams], boxsize, sizeof(float)*3, cudaMemcpyHostToDevice, stream[f%nStreams]);
		//Do math here.
		inverseboxsize<<<1,4,0,stream[f%nStreams]>>>(gpuboxsize[f%nStreams], gpuinvboxsize[f%nStreams]);
		if (csel != NULL) {
			cudaMemsetAsync(gpucenters[f%nStreams],0, 3 * sizeof(float), stream[f%nStreams]);
			//Measure the center of the selection if one is provided. Put it into the 3-vector gpucenters.
			//To exploit some of the symmetry of the problem, pick a blocksize that is a multiple of 3, and preferably
			//also a multiple of the warpsize (96 is good!)
			measurecenter_fullmass<<<(3*csel->selected + 95) / 96, 96, 0, stream[f%nStreams]>>>(gpupos[f%nStreams], gpucenters[f%nStreams], csel->selected, gpuweights, gpuweightidx, wscale);
		}
		//Fragment centers need to be determined.
		//TODO: make this not suck. At the moment, I think this is the biggest bottleneck.
		computefragcenters<<<blocks_frag,threads,0,stream[f%nStreams]>>>(gpupos[f%nStreams], gpufragcenters[f%nStreams], fnum, gpuweights, gpufragweight, gpucompoundmap, gpuindexlist);
		//Wrap.
		wrapcompound<<<blocks, threads, 0, stream[f%nStreams]>>> (gpupos[f%nStreams], sellen, gpucenters[f%nStreams], gpuatomtofragmap, gpuindexlist, gpuboxsize[f%nStreams], gpuinvboxsize[f%nStreams], gpufragcenters[f%nStreams]);
		//Copy back.
		cudaMemcpyAsync(&pos[(f%nStreams)*3*mol->nAtoms], gpupos[f%nStreams], sizeof(float) * 3 *mol->nAtoms, cudaMemcpyDeviceToHost, stream[f%nStreams]);
	}
	//Cleanup
	//Copy back the remaining elements.
	for (f = max(last - nStreams + 1,0); f <= last; f++) {
		cudaStreamSynchronize(stream[f%nStreams]);
		memcpy(mol->get_frame(f)->pos, &pos[(f%nStreams)*3*mol->nAtoms], sizeof(float) * 3*mol->nAtoms);
	}
	//Free memory.
	cudaHostUnregister(boxsize);
	cudaHostUnregister(compoundmap);
	cudaHostUnregister(indexlist);
	cudaFree(gpucompoundmap);
	cudaFree(gpuindexlist);
	cudaFree(gpuatomtofragmap);
	for (f = 0; f< nStreams; f++) {
		cudaStreamDestroy(stream[f]);
		cudaFree(gpupos[f]);
		cudaFree(gpuboxsize[f]);
		cudaFree(gpuinvboxsize[f]);
	}
	cudaFreeHost(pos);
	cudaFree(wscale);
	cudaFree(gpufragweight);
	cudaFree(gpuweights);
	if (csel != NULL) {
		cudaFree(gpuweightidx);
	}
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		// print the CUDA error message and fallback to CPU
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		printf("Reverting to CPU algorithm\n");
		fpbc_exec_wrapcompound_cpu(mol, first, last, fnum, compoundmap, sellen, indexlist, weights, csel, center, massarr);
	}
}


void fpbc_exec_wrapatomic(Molecule* mol, int first, int last, int sellen, int* indexlist, 
	float* weights, AtomSel* csel, float* center) {
	int f, i, j;
	Timestep *ts;
	float *pos;
	float *gpupos[nStreams];
	float boxsize[3];
	float *gpuboxsize[nStreams];
	float *gpucenters[nStreams];
	float *gpuinvboxsize[nStreams];
	float *gpuweights;
	float *wscale;
	cudaMalloc((void**) &wscale, sizeof(float));
	int *gpuweightidx;
	int *gpuindexlist;
	//Prepare GPU memory and streams
	cudaStream_t stream[nStreams];
	cudaMalloc((void**) &gpuindexlist, sizeof(int) * sellen);
	cudaMemcpy(gpuindexlist, indexlist, sizeof(int) * sellen, cudaMemcpyHostToDevice);
	cudaHostRegister(center, sizeof(float) * 3,0);
	cudaHostRegister(boxsize, sizeof(float) * 3,0);
	cudaMallocHost(&pos, nStreams * sizeof(float) * 3*mol->nAtoms);
	for (f = 0; f< nStreams; f++) {
		cudaStreamCreate(&stream[f]);
		cudaMalloc((void**) &gpupos[f], sizeof(float) * 3*mol->nAtoms);
		cudaMalloc((void**) &gpuboxsize[f], sizeof(float) * 3);
		cudaMalloc((void**) &gpuinvboxsize[f], sizeof(float) * 3);
		cudaMalloc((void**) &gpucenters[f], sizeof(float) * 3);
		cudaMemcpyAsync(gpucenters[f], center, sizeof(float)*3, cudaMemcpyHostToDevice, stream[f]);
	}
	if (csel != NULL) {
		cudaMalloc((void**) &gpuweights, sizeof(float) * csel->selected);
		cudaMalloc((void**) &gpuweightidx, sizeof(int) * csel->selected);
		cudaMemcpy(gpuweights, weights, sizeof(float) * csel->selected, cudaMemcpyHostToDevice);
		int *weightidx = new int[csel->selected];
		j=0;
		for (i=csel->firstsel; i<=csel->lastsel; i++) {
			if (csel->on[i]) {
				weightidx[j++] = i;
			}
		}
		cudaMemcpy(gpuweightidx, weightidx, sizeof(int) * csel->selected, cudaMemcpyHostToDevice);
		float tmp = 1.0 / thrust::reduce(thrust::device_ptr<float>(gpuweights), thrust::device_ptr<float>(gpuweights + csel->selected), 0, thrust::plus<float>());
		cudaMemcpy(wscale, &tmp, sizeof(float), cudaMemcpyHostToDevice);
		delete [] weightidx;
	}
	int blocks = (3*sellen + threads - 1) / threads;
	for (f=first; f<=last; f++) {
		ts = mol->get_frame(f);
		boxsize[0] = ts->a_length;
		boxsize[1] = ts->b_length;
		boxsize[2] = ts->c_length;
		//Block here just so that I don't overwrite a buffer.
		cudaStreamSynchronize(stream[f%nStreams]);
		if (! (f-first < nStreams)) {
			memcpy(mol->get_frame(f-nStreams)->pos, &pos[(f%nStreams)*3*mol->nAtoms], sizeof(float) * 3*mol->nAtoms);
		}
		memcpy(&pos[(f%nStreams)*3*mol->nAtoms], ts->pos, sizeof(float) * 3*mol->nAtoms);
		cudaMemcpyAsync(gpupos[f%nStreams], &pos[(f%nStreams)*3*mol->nAtoms], sizeof(float) * 3*mol->nAtoms, cudaMemcpyHostToDevice, stream[f%nStreams]);
		cudaMemcpyAsync(gpuboxsize[f%nStreams], boxsize, sizeof(float)*3, cudaMemcpyHostToDevice, stream[f%nStreams]);
		//Do math here.
		inverseboxsize<<<1,4,0,stream[f%nStreams]>>>(gpuboxsize[f%nStreams], gpuinvboxsize[f%nStreams]);
		if (csel != NULL) {
			cudaMemsetAsync(gpucenters[f%nStreams],0, 3 * sizeof(float), stream[f%nStreams]);
			//Measure the center of the selection if one is provided. Put it into the 3-vector gpucenters.
			//To exploit some of the symmetry of the problem, pick a blocksize that is a multiple of 3, and preferably
			//also a multiple of the warpsize (96 is good!)
			measurecenter<<<(3*csel->selected + 95) / 96, 96, 0, stream[f%nStreams]>>>(gpupos[f%nStreams], gpucenters[f%nStreams], csel->selected, gpuweights, gpuweightidx, wscale);
		}
		//Wrap.
		wrapatomic<<<blocks, threads, 0, stream[f%nStreams]>>> (gpupos[f%nStreams], sellen, gpucenters[f%nStreams], gpuindexlist, gpuboxsize[f%nStreams], gpuinvboxsize[f%nStreams]);
		//Copy back.
		cudaMemcpyAsync(&pos[(f%nStreams)*3*mol->nAtoms], gpupos[f%nStreams], sizeof(float) * 3 *mol->nAtoms, cudaMemcpyDeviceToHost, stream[f%nStreams]);
	}
	//Cleanup. Wait for the kernels to complete.
	for (f = max(last - nStreams + 1,0); f <= last; f++) {
		cudaStreamSynchronize(stream[f%nStreams]);
		memcpy(mol->get_frame(f)->pos, &pos[(f%nStreams)*3*mol->nAtoms], sizeof(float) * 3*mol->nAtoms);
	}
	cudaHostUnregister(boxsize);
	cudaHostUnregister(center);
	cudaFree(gpuindexlist);
	for (f = 0; f< nStreams; f++) {
		cudaStreamDestroy(stream[f]);
		cudaFree(gpupos[f]);
		cudaFree(gpuboxsize[f]);
		cudaFree(gpuinvboxsize[f]);
		cudaFree(gpucenters[f]);
	}
	cudaFreeHost(pos);
	if (csel != NULL) {
		cudaFree(gpuweights);
		cudaFree(gpuweightidx);
	}
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		// print the CUDA error message and fallback to CPU
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		printf("Reverting to CPU algorithm\n");
		fpbc_exec_wrapatomic_cpu(mol, first, last, sellen, indexlist, weights, csel, center);
	}
}


void fpbc_exec_join(Molecule* mol, int first, int last, int fnum, int *compoundmap, int sellen, int* indexlist) {
	int f;
	float *pos;
	Timestep *ts;
	float *gpupos[nStreams];
	float boxsize[3];
	float *gpuboxsize[nStreams];
	float *gpuinvboxsize[nStreams];
	int *gpucompoundmap;
	int *gpuatomtofragmap;
	int *gpuindexlist;
	int blocks = (fnum + threads - 1) / threads;
	cudaStream_t stream[nStreams];
	cudaMalloc((void**) &gpuatomtofragmap, sizeof(int) * sellen);
	cudaMemset(gpuatomtofragmap, 0, sizeof(int) * sellen);
	cudaMalloc((void**) &gpucompoundmap, sizeof(int) * (fnum+1));
	cudaMalloc((void**) &gpuindexlist, sizeof(int) * sellen);
	cudaHostRegister(compoundmap, sizeof(int) * (fnum+1),0);
	cudaHostRegister(indexlist, sizeof(int) * sellen,0);
	cudaMemcpy(gpucompoundmap, compoundmap, sizeof(int) * (fnum+1), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuindexlist, indexlist, sizeof(int) * sellen, cudaMemcpyHostToDevice);
	fragmentperatom<<<blocks, threads>>>(fnum, gpucompoundmap, gpuatomtofragmap);//Setup the gpu per atom map.
	thrust::inclusive_scan(thrust::device_ptr<int>(gpuatomtofragmap), thrust::device_ptr<int>(gpuatomtofragmap + sellen), thrust::device_ptr<int>(gpuatomtofragmap));
	for (f = 0; f< nStreams; f++) {
		cudaStreamCreate(&stream[f]);
		cudaMalloc((void**) &gpupos[f], sizeof(float) * 3*mol->nAtoms);
		cudaMalloc((void**) &gpuboxsize[f], sizeof(float) * 3);
		cudaMalloc((void**) &gpuinvboxsize[f], sizeof(float) * 3);
	}
	cudaHostRegister(boxsize, sizeof(float)*3,0);
	cudaMallocHost(&pos, nStreams * sizeof(float) * 3*mol->nAtoms);
	//Make sure the gpuatomtofragmap is set before proceeding.
	cudaDeviceSynchronize();
	blocks = (3*sellen + threads - 1) / threads;
	for (f = first; f <= last; f++) {
		ts = mol->get_frame(f);
		boxsize[0] = ts->a_length;
		boxsize[1] = ts->b_length;
		boxsize[2] = ts->c_length;
		//Block here just so that I don't overwrite a buffer.
		cudaStreamSynchronize(stream[f%nStreams]);
		if (! (f-first < nStreams)) {
			memcpy(mol->get_frame(f-nStreams)->pos, &pos[(f%nStreams)*3*mol->nAtoms], sizeof(float) * 3*mol->nAtoms);
		}
		memcpy(&pos[(f%nStreams)*3*mol->nAtoms], ts->pos, sizeof(float) * 3*mol->nAtoms);
		cudaMemcpyAsync(gpupos[f%nStreams], &pos[(f%nStreams)*3*mol->nAtoms], sizeof(float) * 3*mol->nAtoms, cudaMemcpyHostToDevice, stream[f%nStreams]);
		cudaMemcpyAsync(gpuboxsize[f%nStreams], boxsize, sizeof(float)*3, cudaMemcpyHostToDevice, stream[f%nStreams]);
		//Do math here.
		inverseboxsize<<<1,4,0,stream[f%nStreams]>>>(gpuboxsize[f%nStreams], gpuinvboxsize[f%nStreams]);
		repositionfragments<<<blocks,threads, 0, stream[f%nStreams]>>>(gpupos[f%nStreams], sellen, gpuatomtofragmap,
			gpucompoundmap, gpuindexlist, gpuboxsize[f%nStreams], gpuinvboxsize[f%nStreams]);
		//Copy back.
		cudaMemcpyAsync(&pos[(f%nStreams)*3*mol->nAtoms], gpupos[f%nStreams], sizeof(float) * 3 *mol->nAtoms, cudaMemcpyDeviceToHost, stream[f%nStreams]);
	}
	for (f = max(last - nStreams + 1,0); f <= last; f++) {
		cudaStreamSynchronize(stream[f%nStreams]);
		memcpy(mol->get_frame(f)->pos, &pos[(f%nStreams)*3*mol->nAtoms], sizeof(float) * 3*mol->nAtoms);
	}
	cudaHostUnregister(boxsize);
	cudaHostUnregister(indexlist);
	cudaHostUnregister(compoundmap);
	cudaFree(gpucompoundmap);
	cudaFree(gpuindexlist);
	cudaFree(gpuatomtofragmap);
	for (f = 0; f< nStreams; f++) {
		cudaStreamDestroy(stream[f]);
		cudaFree(gpupos[f]);
		cudaFree(gpuboxsize[f]);
		cudaFree(gpuinvboxsize[f]);
	}
	cudaFreeHost(pos);
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		// print the CUDA error message and fallback to CPU
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		printf("Reverting to CPU algorithm\n");
		fpbc_exec_join(mol, first, last, fnum, compoundmap, sellen, indexlist);
	}
}


void fpbc_exec_recenter(Molecule* mol, int first, int last, int csellen, int* cindexlist, int fnum, int *compoundmap, int sellen, int* indexlist, float* weights, AtomSel* csel, float* massarr) {
	//The basic idea here is to pass the data back and forth only once while both unwrapping and rewrapping the trajectory.
	Timestep *ts;
	int f;
	float *pos;
	float *gpupos[nStreams];
	float boxsize[3];
	float *gpuboxsize[nStreams];
	float *gpucenters[nStreams];
	float *gpuinvboxsize[nStreams];
	float *gpufragcenters[nStreams];
	float *wscale;
	cudaMalloc((void**) &wscale, sizeof(float));
	float *gpuweights;
	int *gpuweightidx;
	int *gpuindexlist;
	float *gpufragweight;
	float *gpuwscan;
	int *gpuatomtofragmap;
	int *gpucompoundmap;
	cudaStream_t stream[nStreams];
	cudaEvent_t events[nStreams];
	cudaHostRegister(indexlist, sizeof(int) * sellen,0);
	cudaHostRegister(cindexlist, sizeof(int) * csellen,0);
	cudaHostRegister(boxsize, sizeof(float) * 3,0);
	int blocks = (3*sellen + threads - 1) / threads;
	int blocks_frag = (fnum + threads - 1) / threads;
	for (f = 0; f< nStreams; f++) {
		cudaStreamCreate(&stream[f]);
		cudaEventCreateWithFlags(&events[f], cudaEventDisableTiming);
		cudaMalloc((void**) &gpupos[f], sizeof(float) * 3*mol->nAtoms);
		cudaMalloc((void**) &gpuboxsize[f], sizeof(float) * 3);
		cudaMalloc((void**) &gpuinvboxsize[f], sizeof(float) * 3);
		cudaMalloc((void**) &gpucenters[f], sizeof(float) * 3);
		if (fnum) {
			cudaMalloc((void**) &gpufragcenters[f], sizeof(float) * 3 * fnum);
		}
	}
	float tmp;
	cudaMalloc((void**) &gpuindexlist, sizeof(int) * sellen);
	cudaMallocHost(&pos, nStreams * sizeof(float) * 3*mol->nAtoms);
	cudaMemcpy(gpuindexlist, indexlist, sizeof(int) * sellen, cudaMemcpyHostToDevice);
	//Deal with computing the weighted center of mass.
	cudaMalloc((void**) &gpuweightidx, sizeof(int) * csellen);
	cudaMemcpy(gpuweightidx, cindexlist, sizeof(int) * csellen, cudaMemcpyHostToDevice);
	if (fnum) {//Compound runs only.
		cudaMalloc((void**) &gpuweights, sizeof(float) * mol->nAtoms);
		cudaMalloc((void**) &gpufragweight, sizeof(float) * fnum);
		cudaHostRegister(compoundmap, sizeof(int) * (fnum+1),0);
		cudaMemcpy(gpuweights, massarr, sizeof(float) * mol->nAtoms, cudaMemcpyHostToDevice);
		cudaMalloc((void**) &gpucompoundmap, sizeof(int) * (fnum+1));
		cudaMemcpy(gpucompoundmap, compoundmap, sizeof(int) * (fnum+1), cudaMemcpyHostToDevice);
		cudaMalloc((void**) &gpuatomtofragmap, sizeof(int) * sellen);
		cudaMemset(gpuatomtofragmap, 0, sizeof(int) * sellen);
		fragmentperatom<<<blocks_frag, threads>>>(fnum, gpucompoundmap, gpuatomtofragmap);//Setup the gpu per atom map.
		thrust::inclusive_scan(thrust::device_ptr<int>(gpuatomtofragmap), thrust::device_ptr<int>(gpuatomtofragmap + sellen), thrust::device_ptr<int>(gpuatomtofragmap));
		//Get the mass per fragment (for scaling/finding center of mass for everything.)
		thrust::device_vector<float> mass (thrust::device_ptr<float>(gpuweights), thrust::device_ptr<float>(gpuweights+mol->nAtoms));
		mass.push_back(0);//This zero is here so we can scan the weights efficiently.
		cudaMalloc((void**) &gpuwscan, sizeof(float) * (mol->nAtoms+1));
		thrust::exclusive_scan(mass.begin(), mass.end(), thrust::device_ptr<float>(gpuwscan), 0, thrust::plus<float>());
		fragwscale<<<blocks_frag, threads>>>(gpufragweight, gpuwscan, fnum, gpucompoundmap);
		thrust::device_vector<int> ids (thrust::device_ptr<int>(gpuweightidx), thrust::device_ptr<int>(gpuweightidx+csel->selected));
		tmp = 1.0 / thrust::reduce(thrust::make_permutation_iterator(mass.begin(), ids.begin()),
			thrust::make_permutation_iterator(mass.end(), ids.end()), 0, thrust::plus<float>());
		cudaDeviceSynchronize();
		cudaFree(gpuwscan);
	}
	else {
		cudaMalloc((void**) &gpuweights, sizeof(float) * csel->selected);
		cudaMemcpy(gpuweights, weights, sizeof(float) * csel->selected, cudaMemcpyHostToDevice);
		tmp = 1.0 / thrust::reduce(thrust::device_ptr<float>(gpuweights), thrust::device_ptr<float>(gpuweights + csel->selected), 0, thrust::plus<float>());
	}
	cudaMemcpy(wscale, &tmp, sizeof(float), cudaMemcpyHostToDevice);
	//Setup the initial memcopies.
	f = first;
	ts = mol->get_frame(f);
	memcpy(&pos[(f%nStreams)*3*mol->nAtoms], ts->pos, sizeof(float) * 3*mol->nAtoms);
	cudaMemcpyAsync(gpupos[f%nStreams], &pos[(f%nStreams)*3*mol->nAtoms], sizeof(float) * 3*mol->nAtoms, cudaMemcpyHostToDevice, stream[f%nStreams]);
	//Do stuff
	blocks_frag = (3*fnum + threads - 1) / threads;
	for (f=first; f<=last; f++) {
		ts = mol->get_frame(f);
		boxsize[0] = ts->a_length;
		boxsize[1] = ts->b_length;
		boxsize[2] = ts->c_length;
		//Block here just so that I don't overwrite a buffer.
		cudaStreamSynchronize(stream[f%nStreams]);
		if (! (f-first < nStreams)) {
			memcpy(mol->get_frame(f-nStreams)->pos, &pos[(f%nStreams)*3*mol->nAtoms], sizeof(float) * 3*mol->nAtoms);
		}
		cudaMemcpyAsync(gpuboxsize[f%nStreams], boxsize, sizeof(float) * 3, cudaMemcpyHostToDevice, stream[f%nStreams]);
		//Do math here.
		inverseboxsize<<<1,4,0,stream[f%nStreams]>>>(gpuboxsize[f%nStreams], gpuinvboxsize[f%nStreams]);
		if (f > first) {//These are the ones that also need to be unwrapped.
			memcpy(&pos[(f%nStreams)*3*mol->nAtoms], ts->pos, sizeof(float) * 3*mol->nAtoms);
			cudaMemcpyAsync(gpupos[f%nStreams], &pos[(f%nStreams)*3*mol->nAtoms], sizeof(float) * 3*mol->nAtoms, cudaMemcpyHostToDevice, stream[f%nStreams]);
			//We must wait until the previous stream is done moving atoms around or loading. This part is inherently serial.
			cudaStreamWaitEvent(stream[f%nStreams], events[(f-1)%nStreams],0);
			unwrapatomic<<<blocks,threads,0,stream[f%nStreams]>>>(gpupos[f%nStreams], gpupos[(f-1)%nStreams], csellen, gpuweightidx, gpuboxsize[f%nStreams], gpuinvboxsize[f%nStreams]);
			cudaEventRecord(events[f%nStreams],stream[f%nStreams]);
		}
		cudaMemsetAsync(gpucenters[f%nStreams],0, 3 * sizeof(float), stream[f%nStreams]);		
		//Compounding will have a non-zero fnum.
		if (fnum) {
			measurecenter_fullmass<<<(3*csel->selected + 95) / 96, 96, 0, stream[f%nStreams]>>>(gpupos[f%nStreams], gpucenters[f%nStreams], csel->selected, gpuweights, gpuweightidx, wscale);
			//Wrap.
			wrapcompound<<<blocks, threads, 0, stream[f%nStreams]>>> (gpupos[f%nStreams], sellen, gpucenters[f%nStreams], gpuatomtofragmap, gpuindexlist, gpuboxsize[f%nStreams], gpuinvboxsize[f%nStreams], gpufragcenters[f%nStreams]);
		}
		else {
			measurecenter<<<(3*csel->selected + 95) / 96, 96, 0, stream[f%nStreams]>>>(gpupos[f%nStreams], gpucenters[f%nStreams], csel->selected, gpuweights, gpuweightidx, wscale);
			//Wrap.
			wrapatomic<<<blocks, threads, 0, stream[f%nStreams]>>> (gpupos[f%nStreams], sellen, gpucenters[f%nStreams], gpuindexlist, gpuboxsize[f%nStreams], gpuinvboxsize[f%nStreams]);
		}
		cudaEventRecord(events[f%nStreams],stream[f%nStreams]);
		//Copy out.
		cudaMemcpyAsync(&pos[(f%nStreams)*3*mol->nAtoms], gpupos[f%nStreams], sizeof(float) * 3*mol->nAtoms, cudaMemcpyDeviceToHost,stream[f%nStreams]);
	}
	//Copy back the remaining elements.
	for (f = max(last - nStreams + 1,0); f <= last; f++) {
		cudaStreamSynchronize(stream[f%nStreams]);
		memcpy(mol->get_frame(f)->pos, &pos[(f%nStreams)*3*mol->nAtoms], sizeof(float) * 3*mol->nAtoms);
	}
	//Cleanup
	for (f = 0; f< nStreams; f++) {
		cudaStreamDestroy(stream[f]);
		cudaEventDestroy(events[f]);
		cudaFree(gpupos[f]);
		cudaFree(gpuboxsize[f]);
		cudaFree(gpuinvboxsize[f]);
		cudaFree(gpucenters[f]);
		if (fnum) {
			cudaFree(gpufragcenters[f]);
		}
	}
	cudaHostUnregister(indexlist);
	cudaHostUnregister(cindexlist);
	cudaHostUnregister(boxsize);
	cudaFree(gpuindexlist);
	cudaFree(gpuweightidx);
	cudaFree(gpuweights);
	cudaFree(wscale);
	if (fnum) {
		cudaFree(gpucompoundmap);
		cudaFree(gpuatomtofragmap);
		cudaFree(gpufragweight);
		cudaHostUnregister(compoundmap);
	}
	cudaFreeHost(pos);
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		// print the CUDA error message and fallback to CPU
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		printf("Reverting to CPU algorithm\n");
		fpbc_exec_recenter_cpu(mol, first, last, csellen, cindexlist, fnum, compoundmap, sellen, indexlist, weights, csel, massarr);
	}
}

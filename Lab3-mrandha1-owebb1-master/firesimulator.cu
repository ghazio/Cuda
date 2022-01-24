/*
 * Swarthmore College, CS 87
 * Copyright (c) 2020 Swarthmore College Computer Science Department,
 * Swarthmore PA, Professor Tia Newhall
 */

// This is the simulator that contains the fireSimulator class
// Contains methods as well as CUDA kernal methods that we use

#include "firesimulator.h"
#include <handle_cuda_error.h>
#include <curand_kernel.h>
#include <curand.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/types.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <ctype.h>

/*********** constants ***********/
// NOTE: the .h file defines N and NUMITERS since they are used by main.cpp too
// C++ style constants (static const global variables...type info is useful):
//   static: limits the scope to this file
//   const: its value cannot be modified
static const int   BLOCK_SIZE   = 8;     // pick so that it evenly divides N
static const float FIREPROB     = 0.25;  // chances cell catches fire from pal
static const int   STEP         = 20;    // burn rate
static const int   UNBURNED     = 60;
static const int   TEMP_IGNIGHT = 300;
static const int   TEMP_MAX     = 1000;
static const int   LAKE         = 30;
static const int   BURNED       = 50;


/********** function prototypes  **************/
/* print usage info for calling firesimulator program: implemented for you */
static void usage(void);



/********** CUDA Kernel prototypes **************/
//       random number generator (it is almost completely written for you)
//       this should be called one time in firesimulator
//       constructor to init each thread's random number generater state
__global__ void  init_rand(curandState *rand_state, int n_cols);

// this cuda kernel performs the main fire simulation step
// This mainly takes in our board, the state of the random function,
// and the board keeper. The rest are integers that are small pieces
__global__ void  fire_sim_kernel(curandState* state,int* burned_keeper,
            int* data, int n_cols, int n_rows,float prob, int step);

// this is the animation image update kernel
// this takes in the color3 image buffer and the fire simulation
// state and colors each pixel value based the state
__global__ void  fire_to_color(color3 *img_buf, int* burned_keeper,
                                int* data, int n_cols) ;


// if neighbor cell at (x,y) is on fire or not
__device__ int  checkNeighbor(int x, int y, int* data, int n_cols);

/*Checks up to four neihbors and returns number of neighbors that are burning*/
__device__ int firedupNeighbors(int x, int y, int* data, int n_cols,int n_rows);

/***** optional global variables for CPU-side initing only ****/
// The only globals allowed are for CPU side state initialization
// (you can use malloc instead, to avoid global variables and dynamically
//  allocate CPU space to initialize the forest's state)
// DO NOT use local variables that are statically declared 2D arrays of
// this size...it is way too much space to alloate on the stack
// global to avoid huge stack space usage of a local variables (mallocing up
// heap space is fine too for this). This state is only need to init
// state and then copy to CUDA memory).  If you use globals declare them
// to be static, something like this:
//     static int  temps[N][N];

/**************************************************************/
/**************************************************************/
// Destructor that cleans up any state assoc with this object
fireSimulatorKernel::~fireSimulatorKernel() {

  printf("Total GPU Time: %fs\n", time);

  int ret = cudaFree(m_dev_grid);
  if(ret != cudaSuccess) {
    printf("cudaFree failed returned %d\n", ret);
    exit(1);
  }
  m_dev_grid = nullptr;

  ret = cudaFree(dev_random);
  if(ret != cudaSuccess) {
    printf("cudaFree failed returned %d\n", ret);
    exit(1);
  }
  dev_random = nullptr;

  ret = cudaFree(burned_keeper);
  if(ret != cudaSuccess) {
    printf("cudaFree failed returned %d\n", ret);
    exit(1);
  }
  burned_keeper = nullptr;

  printf("Everything is cleaned up!\n");
}

/**
  * Function to process command line args for the kernel
*/
void process_args(int ac, char *av[], int* iters, int* step, float* prob, int* lighting_strike, int* numb_lakes, int** lakes, char* filename){
  int c=1;
  int f=1;
  int i=1;  // p is a flag that we set if we get the -p command line option

  while(1){

    // "p:"  means -p option has an arg  "c"  -c does not
    // in this examle -p, -f and -n have arguments, -c and -h do not
    c = getopt(ac, av, "hi:d:p:f:");

    if( c == -1) {
      break; // nothing left to parse in the command line
    }

    switch(c) {  // switch stmt can be used in place of some chaining if-else if
      case 'h':
                usage();
                break;
      case 'i':
                *iters=atoi(optarg);
                i = 0;
                break;
      case 'd': *step = atoi(optarg);
                i = 0;
                break;
      case 'p': *prob = atof(optarg);
                i = 0;
                break;
      case 'f':
                filename=optarg;
                f=0;
                break;
      default: printf("optopt: %c\n", optopt);
    }
    if(f==0 && i==0){
      usage();
      exit(EXIT_FAILURE);
    }
  }
  if(f==0){
       FILE* fp;
       char* line = NULL;
       size_t len = 0;
       int read;


       fp = fopen(filename, "r");
       if (fp == NULL){
           exit(EXIT_FAILURE);
       }
       read = getline(&line, &len, fp);
       if (read < 0) {
         perror("file not read properly for iters");
         exit(EXIT_FAILURE);
       }

       *iters = atoi(line);


       read = getline(&line, &len, fp);
       if (read < 0) {
         perror("file not read properly for step");
         exit(EXIT_FAILURE);
       }

       *step = atoi(line);

       read = getline(&line, &len, fp);
       if (read < 0) {
         perror("file not read properly for prob");
         exit(EXIT_FAILURE);
       }

       *prob = atof(line);

       read = getdelim(&line, &len, 32, fp);
       if (read < 0) {
         perror("file not read properly for lightng");
         exit(EXIT_FAILURE);
       }
       lighting_strike[0] = atoi(line);
       read = getdelim(&line, &len, 10, fp);
       if (read < 0) {
         perror("file not read properly for lightn strike 2");
         exit(EXIT_FAILURE);
       }

       lighting_strike[1] = atoi(line);

       read = getdelim(&line, &len,10,fp);
       if (read < 0) {
         perror("file not read properly for numb_lakes\n");
         exit(EXIT_FAILURE);
       }

      *numb_lakes = atoi(line);
      int num_buckets=*numb_lakes*4;
      *lakes= (int*)malloc(sizeof(int)*(num_buckets));
      for (int i = 0; i < *numb_lakes; i++){
         int ascii =32;
         for (int j = 0; j < 4; j++){
          if(j==3){
            ascii=10;
          }
           read = getdelim(&line, &len, ascii, fp);
           if (read < 0) {
             perror("file not read properly for lake\n");
             exit(EXIT_FAILURE);
           }
           (*lakes)[i*4+j] = atoi(line);
        }
       }



       fclose(fp);

       if (line){
           free(line);
         }
  } else {

    // Set 4 default lakes when run via command line
    int num_buckets=*numb_lakes*4;
    *lakes=(int*) malloc(sizeof(int)*(num_buckets));
    (*lakes)[0] = N/4;
    (*lakes)[1] = N/4;
    (*lakes)[2] = N/4 + N/8;
    (*lakes)[3] = N/4 + N/8;
    (*lakes)[4] = N/2 + 20;
    (*lakes)[5] = N/2 + 20;
    (*lakes)[6] = N - N/4;
    (*lakes)[7] = N - N/4;
    (*lakes)[8] = N - N/4;
    (*lakes)[9] = N - N/4;
    (*lakes)[10] = N- N/8;
    (*lakes)[11] = N - N/8;
    (*lakes)[12] = N - N/4;
    (*lakes)[13] = N/8;
    (*lakes)[14] = N- N/8;
    (*lakes)[15] = N/4;

  }
  return;
}



/**************************************************************/
// initialize the fireSimulatorKernel, process the commanline args,
// and malloc memory in CUDA and on the CPU
fireSimulatorKernel::fireSimulatorKernel(int w, // width of world
                                         int h, // height of world
                                         int argc, // num command line args
                                         char* argv[]): // command line args
  m_dev_grid(nullptr),dev_random(nullptr),
  m_rows(h),m_cols(w)
  {
  int* lakes = nullptr;
  char* filename=nullptr;
  int ret;
  dim3 blocks(m_cols/BLOCK_SIZE, m_rows/BLOCK_SIZE, 1);
  dim3 threads_block(BLOCK_SIZE, BLOCK_SIZE, 1);
  int* cpu_data;
  int* cpu_burned;
  int numb_lakes = 4;
  float local_amt = 0;
  float tmp;
  cudaEvent_t e1, e2;

  this->time = 0;
  if(argc<=1){
    perror("Inputs not correct\n");
    usage();
    exit(1);
  }

  total_iters=NUMITERS;
  steps =STEP;
  prob=FIREPROB;



  //default lightning strike at middle of the board
  this->lighting_strike[0] = w/2;
  this->lighting_strike[1] = h/2;

  //processes the command line arguments passed in and sets main variables
  process_args(argc,argv,&total_iters,&steps,&prob,lighting_strike,
                &numb_lakes, &lakes, filename);

  // malloc on the CPU
  cpu_data =(int*)malloc(sizeof(int)*m_rows*m_cols);
  cpu_burned =(int*)malloc(sizeof(int)*m_rows*m_cols);

  // malloc on the GPU
  ret = cudaMalloc((void**)&burned_keeper,sizeof(int)*m_rows*m_cols);

  /* intialize CPU memory to Unburnt forest  */
  for (int r = 0; r < m_rows; r++) {
    for (int c = 0; c < m_cols; c++) {
      cpu_data[r * m_cols + c] = UNBURNED;
      cpu_burned[r * m_cols + c] = 0;
    }
  }

  //for each lake
  for(int i=0;i<numb_lakes*4;i=i+4){
    int up_x=lakes[i]-1;
    int up_y=lakes[i+1]-1;
    int lower_x=lakes[i+2]-1;
    int lower_y=lakes[i+3]-1;

    //initialize the lake regions
    for(int c = up_x; c < lower_x;c++){
      for(int r = up_y; r < lower_y; r++ ){
          cpu_data[c*m_cols+r]=LAKE;
        }
      }

  }
  free(lakes);
  lakes = nullptr;

  //set the fire starting point
  cpu_data[lighting_strike[1]*m_cols+lighting_strike[0]]=TEMP_IGNIGHT;

  //allocates memory in the GPU
  int bufSize = sizeof(int)*m_cols*m_rows;
  ret = cudaMalloc((void**)&m_dev_grid, bufSize);
  if(ret != cudaSuccess) {
    printf("malloc m_dev_grid of size %d failed returned %d\n", bufSize, ret);
    exit(1);
  }
  //copy the memory from cpu/host to gpu/device
  ret = cudaMemcpy(m_dev_grid, cpu_data, bufSize, cudaMemcpyHostToDevice);
  if(ret != cudaSuccess) {
    printf("cudaMemCpy failed returned %d\n", ret);
    ret = cudaFree(m_dev_grid);  // if this also fails, can't do much about it
    exit(1);
  }

  //copy the memory from cpu/host to gpu/device
  ret = cudaMemcpy(burned_keeper, cpu_burned, bufSize, cudaMemcpyHostToDevice);
  if(ret != cudaSuccess) {
    printf("cudaMemCpy failed returned %d\n", ret);
    ret = cudaFree(m_dev_grid);  // if this also fails, can't do much about it
    exit(1);
  }

  ret = cudaMalloc((void**)&dev_random, sizeof(curandState)*N*N);
  

  // Need to tim gpu time
  cudaEventCreate(&e1);
  cudaEventCreate(&e2);

  cudaEventRecord(e1,0);

  // init the random function
  init_rand<<<blocks,threads_block>>>(dev_random,m_cols);

  cudaEventRecord(e2, 0);
  cudaEventSynchronize(e2);

  cudaEventElapsedTime(&local_amt, e1, e2);

  tmp = local_amt + this->time;
  this->time = tmp;

  cudaEventDestroy(e1);
  cudaEventDestroy(e2);


  free(cpu_data);
  cpu_data = nullptr;

  free(cpu_burned);
  cpu_burned= nullptr;

}
/**************************************************************/
// implements one step of fire simulation and its animation
// calls two cuda kernels: one to do a simulation step on the 
// world, another to animate the current state of the world 
// (update image_buff)
void fireSimulatorKernel::update(ImageBuffer* img) {
  //declare cuda blocks
  dim3 blocks(m_cols/BLOCK_SIZE, m_rows/BLOCK_SIZE, 1);
  dim3 threads_block(BLOCK_SIZE, BLOCK_SIZE, 1);

  cudaEvent_t e1, e2;
  float local_amt;
  float tmp;
  // needed to time gpu time
  cudaEventCreate(&e1);
  cudaEventCreate(&e2);

  // init_rand for each thread
  cudaEventRecord(e1,0);
  //for (int i = 0; i < 90; i++){
  /* call the fire_sim_kernel kernel on world state */
  fire_sim_kernel<<<blocks, threads_block>>>(dev_random,
                          burned_keeper,m_dev_grid, m_cols,m_rows,prob,steps);
  //}

  /* call the fire_to_color kernel to update the animation image buffer */
  fire_to_color<<<blocks, threads_block>>>(img->buffer,burned_keeper,m_dev_grid,
                                          m_cols);

  cudaEventRecord(e2, 0);
  cudaEventSynchronize(e2);

  cudaEventElapsedTime(&local_amt, e1, e2);

  tmp = local_amt + this->time;
  this->time = tmp;

  cudaEventDestroy(e1);
  cudaEventDestroy(e2);

  usleep(100000);

}


/*
 * returns if cell is currently burning
*/
__device__ int  checkNeighbor(int x, int y, int* data, int n_cols){
    int index = 0;

    index = n_cols *y + x;
    if (data[index] > UNBURNED){
        return 1;
     }

  return 0;
}

/*
 *  Checks up to four neihbors and returns number of neighbors that are burning
 */

__device__ int  firedupNeighbors(int x, int y, int* data, int n_cols,
                                int n_rows)
  {
  //check each of its up to four neighbors
  int fire_neighbors;
  int n1 = 0;
  int n2 = 0;
  int n3 = 0;
  int n4 = 0;

  if (x > 0) {
    n1 =checkNeighbor((x-1),y,data,n_cols);
  }

  if (y > 0){

    n2 =checkNeighbor(x,y-1,data,n_cols);
  }
  if (x<n_cols-1){
    n3 =checkNeighbor(x+1,y,data,n_cols);
  }

  if (y <n_rows-1){
    n4 =checkNeighbor(x,y+1,data,n_cols);
  }

  fire_neighbors = n1 + n2 + n3 + n4;


  return fire_neighbors;
}




/**************************************************************/
// This updates the cell that relates to each thread for
// the iteration
__global__ void  fire_sim_kernel(curandState* dev_random,int* burned_keeper,
            int* data, int n_cols, int n_rows,float prob, int steps)
{

  //find the offset of each thread
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int offset = x + y * n_cols;
  int n_fire=0;
  float val=0;

  //we call the helper function to firedupNeighbors
  if(data[offset]!=LAKE && data[offset] != BURNED){
    //if the cell is unburned forest,
    if(data[offset]==UNBURNED){
      n_fire = firedupNeighbors(x,y,data,n_cols,n_rows);
      //if atleast one neighbor is on fire
      if(n_fire>0){
        val = curand_uniform(&(dev_random[offset]));
        // if the cell catches fire *randomly*
        if( val <= prob) {
          data[offset] = TEMP_IGNIGHT;
        }
      }

    }
    //if the cell was already burning
    else{
      if(burned_keeper[offset]==1){ //it has already approached 1000
        //decrease the fire state by the specified step
        data[offset]=data[offset]-steps;
        if(data[offset]<=UNBURNED){ // if the fire should have gone out for cell
          data[offset]=BURNED;
        }
      }
      else{
        data[offset]=data[offset]+steps;
        if(data[offset]>=TEMP_MAX){
            data[offset]=TEMP_MAX;
            burned_keeper[offset] = 1;
        }
      }
    }
  }
}

/**************************************************************/
// Will color each cell depending on the value of the cell
__global__ void  fire_to_color(color3 *optr, int* burned_keeper,int* data,
                              int n_cols)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int offset = x + y * n_cols;

  // change this pixel's color value based on temperature
  if(data[offset]==LAKE){
    optr[offset].r = 0;  // R value
    optr[offset].g = 0; // G value
    optr[offset].b =  255; // B value
  }
  else if(data[offset]==UNBURNED){
    optr[offset].r = 0;  // R value
    optr[offset].g = 185; // G value
    optr[offset].b =  0; // B value
  }
  else if(data[offset]==BURNED){
    optr[offset].r = 0;  // R value
    optr[offset].g = 0; // G value
    optr[offset].b = 0; // B value
  }
  else if (data[offset] >= TEMP_IGNIGHT && burned_keeper[offset] == 0){
    optr[offset].r = 255;  // R value
    optr[offset].g = 0; // G value
    optr[offset].b = 0; // B value
  }else if (data[offset] >= TEMP_IGNIGHT && burned_keeper[offset] == 1){
    optr[offset].r = 255;  // R value
    optr[offset].g = 119; // G value
    optr[offset].b = 0; // B value
  }else{
    optr[offset].r = 255;  // R value
    optr[offset].g = 255; // G value
    optr[offset].b = 0; // B value
  }

}
/**************************************************************/

__device__ unsigned int hash(unsigned int a) {
  a = (a+0x7ed55d16) + (a<<12);
  a = (a^0xc761c23c) ^ (a>>19);
  a = (a+0x165667b1) + (a<<5);
  a = (a+0xd3a2646c) ^ (a<<9);
  a = (a+0xfd7046c5) + (a<<3);
  a = (a^0xb55a4f09) ^ (a>>16);
  return a;
}

/*
 * initialize the random number generator state for every thread
 *
 * This is a CUDA kernel that is passed allocated curandState
 * in CUDA memory to initialize.  Each thread needs its own
 * state for generating its own sequence of random numbers
 */
__global__ void  init_rand(curandState *rand_state, int n_cols) {

  //       compute x, y, and offset values based on
  //       the block and thread layout
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int rand_offset = x + y * n_cols;
  if(y < N && x < N) {
    curand_init(hash(rand_offset), 0, 0, &(rand_state[rand_offset]));
  }
}

/**************************************************************/
/*
 * prints out firesimulator usage information
 */
static void usage(void) {

  fprintf(stderr,
      "./firesimulator {-i iters -d step -p prob | -f filename}\n"
      " -i iters   the number of iterations to run\n"
      "            0 = infinite, must be >= 0\n"
      "-d step     the rate at which a burning cell's temp increases or\n"
      "            decreases at each step\n"
      "            must be > 0\n"
      "-p prob     the prob that a cell will catch fire if one of\n"
      "            its neighbors is burning\n"
      "            must be >= 0.0, <= 1.0"
      "-f filename read in configuration info from a file\n"
      "-h print out this message\n"
      );
  exit(-1);
}

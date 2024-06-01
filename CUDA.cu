#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cstdlib>
#include <time.h>

#define THREADS 64
#define SIZE 1024
#define BLOCK_SIZE 512
#define TILE_SIZE (BLOCK_SIZE * 2)
// El numero de particulas a reducir podra ser hasta 2 * SIZE
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))
__device__ double dist(int nd, double r1, double r2, double dr[]);
void r8mat_uniform_ab(int m, int n, double a, double b, int *seed, double r[]);
__global__ void update(int np, int nd, double pos[], double vel[], double f[], double acc[], double mass, double dt);
__global__ void compute(int np, int nd, double pos[], double vel[], double mass, double f[], double pot[], double kin[]);
__global__ void reduction(double* g_data, int n, double* out);

static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[]) {
    double *d_acc = NULL;
    double dt;
    double e0;
    double *d_force = NULL;
    double mass = 1.0;
    int nd;
    int np;
    double *d_pos = NULL;
    int step_num;
    int step_print = 0;
    int step_print_index = 0;
    int step_print_num = 10;
    double *d_vel = NULL;



    struct timeval tv_start, tv_end;
    double run_time;

    printf("\nMD\n");
    printf("  CUDA version\n");
    printf("  A molecular dynamics program.\n");

    if (argc > 1)
        nd = atoi(argv[1]);
    else {
        printf("\nEnter ND, the spatial dimension (2 or 3).\n");
        scanf("%d", &nd);
    }

    if (argc > 2)
        np = atoi(argv[2]);
    else {
        printf("\nEnter NP, the number of particles (500, for instance).\n");
        scanf("%d", &np);
    }

    if (argc > 3)
        step_num = atoi(argv[3]);
    else {
        printf("\nEnter ND, the number of time steps (500 or 1000, for instance).\n");
        scanf("%d", &step_num);
    }

    if (argc > 4)
        dt = atof(argv[4]);
    else {
        printf("\nEnter DT, the size of the time step (0.1, for instance).\n");
        scanf("%lf", &dt);
    }

    printf("\nND, the spatial dimension, is %d\n", nd);
    printf("NP, the number of particles in the simulation, is %d\n", np);
    printf("STEP_NUM, the number of time steps, is %d\n", step_num);
    printf("DT, the size of each time step, is %lf\n", dt);

   double* h_acc;
   double* h_force;
   double* h_pos;
   double* h_vel;
   double* h_ken;
   double* h_pen;
   double* d_ken = NULL;
   double* d_pen = NULL;
   double ken_res;
   double pen_res;

   unsigned int size = nd * np * sizeof(double);
   unsigned int size_part = np * sizeof(double);

    h_acc = (double *)malloc(size);
    h_force = (double *)malloc(size);
    h_pos = (double *)malloc(size);
    h_vel = (double *)malloc(size);
    h_ken = (double *)malloc(size_part);
    h_pen = (double *)malloc(size_part);

    cudaMalloc(&d_acc, nd * np * sizeof(double));
    cudaMalloc(&d_force, nd * np * sizeof(double));
    cudaMalloc(&d_pos, nd * np * sizeof(double));
    cudaMalloc(&d_vel, nd * np * sizeof(double));

    printf("\nAt each step, we report the potential and kinetic energies.\n");
    printf("The sum of these energies should be a constant.\n");
    printf("As an accuracy check, we also print the relative error\n");
    printf("in the total energy.\n");
    printf("\n      Step      Potential       Kinetic        (P+K-E0)/E0\n");
    printf("                Energy P        Energy K       Relative Energy Error\n");
    printf("\n");

    step_print = 0;
    step_print_index = 0;
    step_print_num = 10;

    gettimeofday(&tv_start, NULL);

    int j;
    int seed = 123456789;
    r8mat_uniform_ab(nd, np, 0.0, 10.0, &seed, h_pos);

    for (int j = 0; j < np; j++) {
        for (int i = 0; i < nd; i++) {
            h_vel[i + j * nd] = 0.0;
            h_acc[i + j * nd] = 0.0;
        }
    }

    for(j = 0 ; j < np ; j++){
        h_ken[j] = 0;
        h_pen[j] = 0;
    }

    cudaMalloc(&d_acc, size);
    cudaMalloc(&d_force, size);
    cudaMalloc(&d_pos, size);
    cudaMalloc(&d_vel, size);
    cudaMalloc(&d_ken, size_part);
    cudaMalloc(&d_pen, size_part);

    cudaMemcpy(d_acc, h_acc, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos, h_pos, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel, h_vel, size, cudaMemcpyHostToDevice);
    cudaMemset(d_force, 0, size);
    cudaMemset(d_ken, 0, size_part);
    cudaMemset(d_pen, 0, size_part);


    int bDim = THREADS;   // numThreads = 64
    int gDim = np / bDim; // numBlocks
    int shared_memory_size = THREADS * nd * sizeof(double);


    for (int step = 0; step <= step_num; step++) {
        if (step > 0) {
            update<<<gDim, THREADS>>>(np, nd, d_pos, d_vel, d_force, d_acc, mass, dt);
        }
        compute<<<gDim, bDim, shared_memory_size>>>(np, nd, d_pos, d_vel, mass, d_force, d_pen, d_ken);



        if (step == step_num / 10 * (step / (step_num / 10))){
            ken_res = 0.0;
            pen_res = 0.0;

            double* d_pen_res;
            double* d_ken_res;
            cudaMalloc((void**)&d_pen_res, sizeof(double));
            cudaMalloc((void**)&d_ken_res, sizeof(double));
        
            cudaMemcpy(d_pen_res, &pen_res, sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_ken_res, &ken_res, sizeof(double), cudaMemcpyHostToDevice);
        
            int blocks = ceil(np / (double)TILE_SIZE);

            reduction <<<blocks, BLOCK_SIZE>>>(d_pen, np, d_pen_res);
            reduction <<<blocks, BLOCK_SIZE>>>(d_ken, np, d_ken_res);

            cudaMemcpy(&pen_res, d_pen_res, sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(&ken_res, d_ken_res, sizeof(double), cudaMemcpyDeviceToHost);
            cudaFree(d_pen_res);
            cudaFree(d_ken_res);
            
            if (step == 0) {
                e0 = ken_res + pen_res;
            }


            printf("  %8d  %14f  %14f  %14e\n", step, pen_res, ken_res, (pen_res + ken_res - e0) / e0);
            step_print += 2;
            step_print_index = step_print_index + 1;
            step_print = (step_print_index * step_num) / step_print_num;
        }
    }


    gettimeofday(&tv_end, NULL);
    run_time = (tv_end.tv_sec - tv_start.tv_sec) * 1000000 +
               (tv_end.tv_usec - tv_start.tv_usec); // en us
    run_time = run_time / 1000000;                  // en s

    printf("\n Tiempo version CUDA = %lg s\n", run_time);

    free(h_acc);
    free(h_force);
    free(h_pos);
    free(h_vel);
    free(h_pen);   
    free(h_ken);

    cudaFree(d_acc);
    cudaFree(d_force);
    cudaFree(d_pos);
    cudaFree(d_vel);
    cudaFree(d_ken);
    cudaFree(d_pen);

    printf("\nMD\n");
    printf("  Normal end of execution.\n");
    printf("\n");

    return 0;
}
__device__ double dist(int nd, double *r1, double *r2, double *dr) {
    double d = 0.0;
    for (int i = 0; i < nd; i++) {
        dr[i] = r1[i] - r2[i];
        d += dr[i] * dr[i];
    }
    return sqrt(d);
}

void r8mat_uniform_ab(int m, int n, double a, double b, int *seed, double r[]) {
    const int i4_huge = 2147483647;

    if (*seed == 0) {
        fprintf(stderr, "\nR8MAT_UNIFORM_AB - Fatal error!\n");
        fprintf(stderr, "  Input value of SEED = 0.\n");
        exit(1);
    }

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            int k = *seed / 127773;
            *seed = 16807 * (*seed - k * 127773) - k * 2836;

            if (*seed < 0) {
                *seed += i4_huge;
            }

            r[i + j * m] = a + (b - a) * (double)(*seed) * 4.656612875E-10;
        }
    }
} 

__global__ void update(int np, int nd, double *pos, double *vel, double *f, double *acc, double mass, double dt) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= np) return;

    double rmass = 1.0 / mass;
    for (int i = 0; i < nd; i++) {
        pos[i + k * nd] = pos[i + k * nd] + vel[i + k * nd] * dt + 0.5 * acc[i + k * nd] * dt * dt;
        vel[i + k * nd] = vel[i + k * nd] + 0.5 * dt * (f[i + k * nd] * rmass + acc[i + k * nd]);
        acc[i + k * nd] = f[i + k * nd] * rmass;
    }
}

__global__ void compute(int np, int nd, double pos[], double vel[], double mass, double f[], double pot[], double kin[]) {
    extern __shared__ double shared_pos[];

    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (k >= np) return;

    double ke = 0.0;
    double pe = 0.0;
    double PI2 = 1.570796327; // 3.141592653589793 / 2.0
    double rij[3];

    // Load particle positions into shared memory
    for (int i = 0; i < nd; i++) {
        shared_pos[tid * nd + i] = pos[k * nd + i];
    }
    __syncthreads();

    // Initialize force to zero
    for (int i = 0; i < nd; i++) {
        f[i + k * nd] = 0.0;
    }

    // Compute forces and potential energy
    for (int j = 0; j < np; j++) {
        if (k != j) {
            double *pos_k = &shared_pos[tid * nd];
            double *pos_j = &pos[j * nd];

            double d_squared = 0.0;
            for (int i = 0; i < nd; i++) {
                rij[i] = pos_k[i] - pos_j[i];
                d_squared += rij[i] * rij[i];
            }

            if (d_squared > 0.0) {
                double d = sqrt(d_squared);
                double d2 = (d < PI2) ? d : PI2;
                double sin_d2 = sin(d2);
                double sin_2d2 = sin(2.0 * d2);

                pe += 0.5 * sin_d2 * sin_d2;

                for (int i = 0; i < nd; i++) {
                    f[i + k * nd] -= rij[i] * sin_2d2 / d;
                }
            }
        }
    }

    // Compute kinetic energy
    for (int i = 0; i < nd; i++) {
        ke += vel[k * nd + i] * vel[k * nd + i];
    }
    ke *= 0.5 * mass;

    // Store potential and kinetic energies
    pot[k] = pe;
    kin[k] = ke;
}


__global__ void reduction(double *g_data, int n, double* out) {
    int tile = TILE_SIZE * blockIdx.x;
    __shared__ double data[BLOCK_SIZE];

    int my_idx = tile + threadIdx.x;
    data[threadIdx.x] = (my_idx < n) ? g_data[my_idx] : 0.0;

    int next_idx = my_idx + blockDim.x;
    if (next_idx < n) {
        data[threadIdx.x] += g_data[next_idx];
    }

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            data[threadIdx.x] += data[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(out, data[0]);
    }
}

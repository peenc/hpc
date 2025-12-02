/**
 * @file main-multi.c
 * @brief Matrix operations: loading, saving, printing, and multiplication.
 */

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>
#include <omp.h>
#include <getopt.h>
#include <assert.h>
#include <stdint.h>
#include <openssl/evp.h>
#include <sys/stat.h>

#define STRING_SIZE 64*1024
#ifdef __OPTIMIZE__
#define PROJECT_NAME "MAT-MULT-TN712-OPTIMIZE"
#else
#define PROJECT_NAME "MAT-MULT-TN712-NON-OPTIMIZE"
#endif

#define PAPAI_EVENTS_NUMBER 2

#define NO_PAPI

typedef struct {
    uint32_t m, n;
    double *v;
} tpMatrix;

double get_wall_time(void);
void print_matrix(const tpMatrix *A);
void load_binary(tpMatrix *A, char * filename);
void save_binary(tpMatrix *A, char * filename);

void matrix_multi(long long **p_values,
                  tpMatrix *  C,
                  const tpMatrix *  A,
                  const tpMatrix *  B);

uint32_t md5_from_memory(uint8_t *md_value, const unsigned char *data, size_t len);
void save_anwser(const int threads, 
                 const double elapsedtime, 
                 const uint64_t mem, 
                 const long long **p_values, 
                 const uint8_t *md5, 
                 const uint32_t md_len);
void help(void);

int main (int ac, char **av){

    double elapsedtime = 0.0;
    uint64_t mem = 0;
    int show_matrix = 0;
    int option_index = 0;
    int threads =  omp_get_num_procs();
    int input_opt = 0;
    tpMatrix A, B, C;

    char filename_matrix_A[STRING_SIZE],
         filename_matrix_B[STRING_SIZE],
         filename_matrix_C[STRING_SIZE];

    filename_matrix_A[0] = 0;
    filename_matrix_B[0] = 0;
    filename_matrix_C[0] = 0;

    long long *p_values[PAPAI_EVENTS_NUMBER];

    if (ac == 1) help();

    struct option long_options[] =
    {
        {"help",     no_argument,  0, 'h'},
        {"answer-matrix-c",   optional_argument, 0, 'c'},
        {"matrix-a",   required_argument, 0, 'a'},
        {"matrix-b",   required_argument, 0, 'b'},
        {"threads",   optional_argument, 0, 't'},
        {"show",   no_argument, 0, 's'},
        {0, 0, 0, 0}
    };

    while ((input_opt = getopt_long (ac, av, "hc:a:b:t:s", long_options, &option_index)) != EOF){
        switch (input_opt)
        {
            case 'h': help(); break;
            case 'c': strcpy(filename_matrix_C, optarg); break;
            case 'a': strcpy(filename_matrix_A, optarg); break;
            case 'b': strcpy(filename_matrix_B, optarg); break;
            case 't': threads = atoi(optarg); break;
            case 's': show_matrix = 1; break; 
            default: help(); break;
        }
    }

    assert(filename_matrix_A[0] != 0);
    assert(filename_matrix_B[0] != 0);

    printf("Matrix multiplication\n\n");
    printf(" - Matrix A: [%s]\n", filename_matrix_A);
    printf(" - Matrix B: [%s]\n", filename_matrix_B);

    if (filename_matrix_C[0] != 0)
        printf(" - Matrix C: [%s]\n", filename_matrix_C);

    load_binary(&A, filename_matrix_A);
    load_binary(&B, filename_matrix_B);

    C.m = A.m;
    C.n = B.n;
    C.v = (double*) malloc(sizeof(double) * C.m * C.n);

    mem = sizeof(double) * ( (A.m * A.n) + (B.m * B.n) + (C.m * C.n) );

    memset(C.v, 0x00, C.m * C.n * sizeof(double));

    omp_set_num_threads(threads);
    printf("\t - Threads used: %d\n", threads);
    printf("\t - Memory used: %lu bytes\n", mem);

    for (uint64_t i = 0; i < PAPAI_EVENTS_NUMBER; i++){
        p_values[i] = (long long *) malloc(sizeof(long long) * threads);
        memset(p_values[i], 0, sizeof(long long) * threads);
        assert(p_values[i] != NULL);
    }

    // --- SEM PAPI ---
    elapsedtime = get_wall_time();
    matrix_multi(p_values, &C, &A, &B);
    elapsedtime = get_wall_time() - elapsedtime;
    // -----------------

    if (show_matrix){
        printf("\n-------------------------------------------------------------------------------\n");
        print_matrix(&A);
        printf("\n-------------------------------------------------------------------------------\n");
        print_matrix(&B);
        printf("\n-------------------------------------------------------------------------------\n");
        print_matrix(&C);
    }

    unsigned char md5_value[EVP_MAX_MD_SIZE];
    uint32_t len = md5_from_memory(&md5_value[0],
                                   (const unsigned char*)C.v,
                                   (C.m * C.n * sizeof(double)));

    save_anwser(threads, elapsedtime, mem,
                (const long long **)p_values,
                &md5_value[0], len);

    for (uint64_t i = 0; i < PAPAI_EVENTS_NUMBER; i++){
        free(p_values[i]);
    }

    free(A.v);
    free(B.v);
    free(C.v);

    return EXIT_SUCCESS;
}


double get_wall_time(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == -1) {
        perror("clock_gettime");
        return 0.0;
    }
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

void print_matrix(const tpMatrix *A){
    printf("\t Print matrix (%u, %u)\n", A->m, A->n);
    for (uint32_t j = 0; j < A->m; j++){
        for (uint32_t i = 0; i < A->n; i++){
            printf("% 15.8lf;", A->v[j * A->n + i]);
        }
        printf("\n");
    }
}

void load_binary(tpMatrix *A, char * filename){
    FILE *input = fopen(filename, "rb");
    uint64_t bytesRead = 0;
    assert(input != NULL);

    bytesRead  = fread(&A->m, sizeof(uint32_t), 1, input);
    bytesRead += fread(&A->n, sizeof(uint32_t), 1, input);
    bytesRead *= sizeof(uint32_t);

    A->v = (double*) malloc(sizeof(double) * A->m * A->n);
    bytesRead += fread(A->v, sizeof(double), A->m * A->n, input) * sizeof(double);

    printf("\t load_binary - bytes read [%lu]\n", bytesRead);
    fclose(input);
}

void save_binary(tpMatrix *A, char * filename){
    FILE *output = fopen(filename, "wb+");
    uint64_t bytesWrite = 0;
    assert(output != NULL);

    bytesWrite  = fwrite(&A->m, sizeof(uint32_t), 1, output);
    bytesWrite += fwrite(&A->n, sizeof(uint32_t), 1, output);
    bytesWrite *= sizeof(uint32_t);

    bytesWrite += fwrite(A->v, sizeof(double), A->m * A->n, output) * sizeof(double);

    printf("\t save_binary - bytes written [%lu]\n", bytesWrite);
    fclose(output);
}


void matrix_multi(long long **p_values,
                  tpMatrix *  C,
                  const tpMatrix *  A,
                  const tpMatrix *  B){

    #pragma omp parallel for
    for (uint32_t j = 0; j < C->m; j++){
        for (uint32_t i = 0; i < C->n; i++){
            double c = 0.0f;
            for (uint32_t jA = 0; jA < A->n; jA++){
                uint32_t ak = j * A->n + jA;
                uint32_t bk = jA * B->n + i;
                c += A->v[ak] * B->v[bk];
            }
            C->v[j * C->n + i] = c;
        }
    }
}


/* -------------------------------------------------------------------------- */

uint32_t md5_from_memory(uint8_t *md_value, const unsigned char *data, size_t len) {
    uint32_t md_len;
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(ctx, EVP_md5(), NULL);
    EVP_DigestUpdate(ctx, data, len);
    EVP_DigestFinal_ex(ctx, md_value, &md_len);
    EVP_MD_CTX_free(ctx);
    return md_len;
}

void save_anwser(const int threads, 
                 const double elapsedtime, 
                 const uint64_t mem, 
                 const long long **p_values, 
                 const uint8_t *md5, 
                 const uint32_t md_len){

    char file_name[STRING_SIZE];
    struct stat st;

    sprintf(file_name, "%s-%03d.csv", PROJECT_NAME, threads);
    printf("\t - Saving [%s]", file_name);

    FILE *ptr = NULL;

    if (!stat(file_name, &st) == 0) {
        ptr = fopen(file_name, "w+");
        assert(ptr != NULL);
        fprintf(ptr, "threads;elapsedtime;mem;md5_anwser");
        for (int i = 0; i < threads; i++){
            fprintf(ptr, ";t-%d-TOT_INS;t-%d-TOT_CYC", i, i);
        }
        fprintf(ptr, "\n");
    } else {
        ptr = fopen(file_name, "a+");
        assert(ptr != NULL);
    }

    fprintf(ptr, "%d;%lf;%u;", threads, elapsedtime, mem);
    for (uint8_t i = 0; i < md_len; i++)
        fprintf(ptr, "%02x", md5[i]);

    for (int i = 0; i < threads; i++){
        // valores zerados (sem PAPI)
        fprintf(ptr, ";0;0");
    }
    fprintf(ptr, "\n");

    fclose(ptr);
    printf("\t OK\n");
}

void help(void){
    fprintf(stdout, "\nMatrix multiplication\n");
    fprintf(stdout, "Usage: ../m_mult.exec [ -c matrix C.bin ] < -a matrix A.bin > < -b matrix B.bin > [ -t threads ] [ -s ]\n");
    exit(EXIT_FAILURE);
}

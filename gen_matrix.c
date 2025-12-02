#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Uso: ./gen_matrix <m> <n> <arquivo.bin>\n");
        return 1;
    }

    uint32_t m = atoi(argv[1]);
    uint32_t n = atoi(argv[2]);
    char* filename = argv[3];

    FILE* f = fopen(filename, "wb");
    if (!f) {
        perror("Erro ao abrir arquivo");
        return 1;
    }

    fwrite(&m, sizeof(uint32_t), 1, f);
    fwrite(&n, sizeof(uint32_t), 1, f);

    srand(time(NULL));

    for (uint64_t i = 0; i < (uint64_t)m * n; i++) {
        double v = (double)(rand() % 10);
        fwrite(&v, sizeof(double), 1, f);
    }

    fclose(f);

    printf("Matriz %ux%u salva em %s\n", m, n, filename);
    return 0;
}

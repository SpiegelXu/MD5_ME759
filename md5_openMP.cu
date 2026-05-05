#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <cuda_runtime.h>
#include <omp.h> 

#define BATCH_SIZE 2097152   // large Batch
#define WORDS_PER_MSG 16     
#define NUM_STREAMS 2        
#define A0 0x67452301
#define B0 0xefcdab89
#define C0 0x98badcfe
#define D0 0x10325476

// MD5 
#define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z)))
#define ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

#define FF(a, b, c, d, x, s, ac) { (a) += F((b), (c), (d)) + (x) + (uint32_t)(ac); (a) = ROTATE_LEFT((a), (s)); (a) += (b); }
#define GG(a, b, c, d, x, s, ac) { (a) += G((b), (c), (d)) + (x) + (uint32_t)(ac); (a) = ROTATE_LEFT((a), (s)); (a) += (b); }
#define HH(a, b, c, d, x, s, ac) { (a) += H((b), (c), (d)) + (x) + (uint32_t)(ac); (a) = ROTATE_LEFT((a), (s)); (a) += (b); }
#define II(a, b, c, d, x, s, ac) { (a) += I((b), (c), (d)) + (x) + (uint32_t)(ac); (a) = ROTATE_LEFT((a), (s)); (a) += (b); }

struct md5_ctx { uint32_t a, b, c, d; };

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void md5_kernel_soa(const uint32_t* __restrict__ d_data, md5_ctx target, int* match_idx, int total) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= total) return;

    uint32_t in[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) in[i] = d_data[i * total + idx];

    uint32_t a = A0, b = B0, c = C0, d = D0;

    FF(a,b,c,d,in[0],7,0xd76aa478); FF(d,a,b,c,in[1],12,0xe8c7b756); FF(c,d,a,b,in[2],17,0x242070db); FF(b,c,d,a,in[3],22,0xc1bdceee);
    FF(a,b,c,d,in[4],7,0xf57c0faf); FF(d,a,b,c,in[5],12,0x4787c62a); FF(c,d,a,b,in[6],17,0xa8304613); FF(b,c,d,a,in[7],22,0xfd469501);
    FF(a,b,c,d,in[8],7,0x698098d8); FF(d,a,b,c,in[9],12,0x8b44f7af); FF(c,d,a,b,in[10],17,0xffff5bb1); FF(b,c,d,a,in[11],22,0x895cd7be);
    FF(a,b,c,d,in[12],7,0x6b901122); FF(d,a,b,c,in[13],12,0xfd987193); FF(c,d,a,b,in[14],17,0xa679438e); FF(b,c,d,a,in[15],22,0x49b40821);

    GG(a,b,c,d,in[1],5,0xf61e2562); GG(d,a,b,c,in[6],9,0xc040b340); GG(c,d,a,b,in[11],14,0x265e5a51); GG(b,c,d,a,in[0],20,0xe9b6c7aa);
    GG(a,b,c,d,in[5],5,0xd62f105d); GG(d,a,b,c,in[10],9,0x02441453); GG(c,d,a,b,in[15],14,0xd8a1e681); GG(b,c,d,a,in[4],20,0xe7d3fbc8);
    GG(a,b,c,d,in[9],5,0x21e1cde6); GG(d,a,b,c,in[14],9,0xc33707d6); GG(c,d,a,b,in[3],14,0xf4d50d87); GG(b,c,d,a,in[8],20,0x455a14ed);
    GG(a,b,c,d,in[13],5,0xa9e3e905); GG(d,a,b,c,in[2],9,0xfcefa3f8); GG(c,d,a,b,in[7],14,0x676f02d9); GG(b,c,d,a,in[12],20,0x8d2a4c8a);

    HH(a,b,c,d,in[5],4,0xfffa3942); HH(d,a,b,c,in[8],11,0x8771f681); HH(c,d,a,b,in[11],16,0x6d9d6122); HH(b,c,d,a,in[14],23,0xfde5380c);
    HH(a,b,c,d,in[1],4,0xa4beea44); HH(d,a,b,c,in[4],11,0x4bdecfa9); HH(c,d,a,b,in[7],16,0xf6bb4b60); HH(b,c,d,a,in[10],23,0xbebfbc70);
    HH(a,b,c,d,in[13],4,0x289b7ec6); HH(d,a,b,c,in[0],11,0xeaa127fa); HH(c,d,a,b,in[3],16,0xd4ef3085); HH(b,c,d,a,in[6],23,0x04881d05);
    HH(a,b,c,d,in[9],4,0xd9d4d039); HH(d,a,b,c,in[12],11,0xe6db99e5); HH(c,d,a,b,in[15],16,0x1fa27cf8); HH(b,c,d,a,in[2],23,0xc4ac5665);

    II(a,b,c,d,in[0],6,0xf4292244); II(d,a,b,c,in[7],10,0x432aff97); II(c,d,a,b,in[14],15,0xab9423a7); II(b,c,d,a,in[5],21,0xfc93a039);
    II(a,b,c,d,in[12],6,0x655b59c3); II(d,a,b,c,in[3],10,0x8f0ccc92); II(c,d,a,b,in[10],15,0xffeff47d); II(b,c,d,a,in[1],21,0x85845dd1);
    II(a,b,c,d,in[8],6,0x6fa87e4f); II(d,a,b,c,in[15],10,0xfe2ce6e0); II(c,d,a,b,in[6],15,0xa3014314); II(b,c,d,a,in[13],21,0x4e0811a1);
    II(a,b,c,d,in[4],6,0xf7537e82); II(d,a,b,c,in[11],10,0xbd3af235); II(c,d,a,b,in[2],15,0x2ad7d2bb); II(b,c,d,a,in[9],21,0xeb86d391);

    if ((a + A0) == target.a && (b + B0) == target.b && (c + C0) == target.c && (d + D0) == target.d) {
        atomicExch(match_idx, idx);
    }
}

bool parse_hex(const std::string& hex, md5_ctx& ctx) {
    if (hex.length() != 32) return false;
    uint32_t r[4];
    for (int i = 0; i < 4; i++) {
        uint32_t v; sscanf(hex.substr(i*8, 8).c_str(), "%08x", &v);
        r[i] = ((v & 0xFF) << 24) | ((v & 0xFF00) << 8) | ((v & 0xFF0000) >> 8) | ((v >> 24) & 0xFF);
    }
    ctx = {r[0], r[1], r[2], r[3]}; return true;
}

int main(int argc, char** argv) {
    if (argc < 3) return 1;
    md5_ctx target; parse_hex(argv[2], target);

    std::ifstream file(argv[1]);
    std::vector<std::string> wordlist;
    std::string line;
    while (std::getline(file, line)) if (!line.empty()) wordlist.push_back(line);

    uint32_t *h_data[NUM_STREAMS], *d_data[NUM_STREAMS];
    int *h_match[NUM_STREAMS], *d_match[NUM_STREAMS];
    cudaStream_t streams[NUM_STREAMS];
    size_t batch_base[NUM_STREAMS];

    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaHostAlloc(&h_data[i], BATCH_SIZE * 16 * 4, cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&h_match[i], 4, cudaHostAllocDefault));
        CHECK_CUDA(cudaMalloc(&d_data[i], BATCH_SIZE * 16 * 4));
        CHECK_CUDA(cudaMalloc(&d_match[i], 4));
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
        *h_match[i] = -1;
    }

    std::cout << "Starting RTX 4090 [SoA + OpenMP + Async] Crack...\n";
    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    bool found = false; std::string result;
    for (size_t i = 0; i < wordlist.size(); i += BATCH_SIZE) {
        int s = (i / BATCH_SIZE) % NUM_STREAMS;
        int cur = (int)std::min((size_t)BATCH_SIZE, wordlist.size() - i);

        CHECK_CUDA(cudaStreamSynchronize(streams[s]));
        if (*h_match[s] != -1) { result = wordlist[batch_base[s] + *h_match[s]]; found = true; break; }

        batch_base[s] = i; *h_match[s] = -1;

        // --- OpenMP  ---
        #pragma omp parallel for
        for (int j = 0; j < cur; j++) {
            const std::string& str = wordlist[i + j];
            uint32_t temp[16] = {0};
            size_t len = std::min(str.length(), (size_t)55);
            memcpy(temp, str.data(), len);
            ((uint8_t*)temp)[len] = 0x80;
            temp[14] = (uint32_t)(len * 8);
            for (int w = 0; w < 16; w++) h_data[s][w * cur + j] = temp[w];
        }

        CHECK_CUDA(cudaMemcpyAsync(d_data[s], h_data[s], cur * 16 * 4, cudaMemcpyHostToDevice, streams[s]));
        CHECK_CUDA(cudaMemcpyAsync(d_match[s], h_match[s], 4, cudaMemcpyHostToDevice, streams[s]));
        md5_kernel_soa<<<(cur+255)/256, 256, 0, streams[s]>>>(d_data[s], target, d_match[s], cur);
        CHECK_CUDA(cudaMemcpyAsync(h_match[s], d_match[s], 4, cudaMemcpyDeviceToHost, streams[s]));
        
        if (i % (BATCH_SIZE * 5) == 0) printf("\rProgress: %.2f%%", (float)i / wordlist.size() * 100), fflush(stdout);
    }

    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms; cudaEventElapsedTime(&ms, start, stop);
    
    if (found) std::cout << "\n[+] Cracked: " << result << "\n";
    else std::cout << "\n[-] Not found.\n";
    std::cout << "[*] Speed: " << (wordlist.size() / (ms / 1000.0)) / 1e6 << " MH/s\n";

    return 0;
}
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <cuda_runtime.h>
#include <stdint.h>

#define BATCH_SIZE 1048576  
#define WORDS_PER_MSG 16    
#define NUM_STREAMS 2       
#define A0 0x67452301
#define B0 0xefcdab89
#define C0 0x98badcfe
#define D0 0x10325476

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

// SoA 
__global__ void md5_crack_kernel_soa(const uint32_t* __restrict__ d_data, md5_ctx target, int* match_idx, int total) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= total || *match_idx != -1) return;

    uint32_t in[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        in[i] = d_data[i * total + idx];
    }

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
        atomicCAS(match_idx, -1, idx);
    }
}

bool parse_hex(const std::string& hex_str, uint32_t* a, uint32_t* b, uint32_t* c, uint32_t* d) {
    if (hex_str.length() != 32) return false;
    uint32_t raw[4];
    for (int i = 0; i < 4; i++) {
        std::string part = hex_str.substr(i * 8, 8);
        uint32_t val; sscanf(part.c_str(), "%08x", &val);
        raw[i] = ((val & 0xFF) << 24) | ((val & 0xFF00) << 8) | ((val & 0xFF0000) >> 8) | ((val >> 24) & 0xFF);
    }
    *a = raw[0]; *b = raw[1]; *c = raw[2]; *d = raw[3];
    return true;
}

int main(int argc, char** argv) {
    if (argc != 3) { std::cerr << "Usage: " << argv[0] << " <wordlist> <hash>\n"; return 1; }

    md5_ctx target_ctx;
    if (!parse_hex(argv[2], &target_ctx.a, &target_ctx.b, &target_ctx.c, &target_ctx.d)) return 1;

    std::ifstream file(argv[1]);
    if (!file) { std::cerr << "Cannot open wordlist.\n"; return 1; }
    std::vector<std::string> wordlist;
    std::string line;
    while (std::getline(file, line)) if (!line.empty()) wordlist.push_back(line);
    file.close();

    // initilaizer
    uint32_t *h_data[NUM_STREAMS];
    int *h_match[NUM_STREAMS];
    uint32_t *d_data[NUM_STREAMS];
    int *d_match[NUM_STREAMS];
    size_t batch_base_idx[NUM_STREAMS]; 
    cudaStream_t streams[NUM_STREAMS];

    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaHostAlloc(&h_data[i], BATCH_SIZE * WORDS_PER_MSG * sizeof(uint32_t), cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&h_match[i], sizeof(int), cudaHostAllocDefault));
        CHECK_CUDA(cudaMalloc(&d_data[i], BATCH_SIZE * WORDS_PER_MSG * sizeof(uint32_t)));
        CHECK_CUDA(cudaMalloc(&d_match[i], sizeof(int)));
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
        *h_match[i] = -1;
        batch_base_idx[i] = 0;
    }

    std::cout << "Loaded " << wordlist.size() << " words. Starting RTX 4090 Streamed Crack...\n";

    cudaEvent_t start_ev, stop_ev;
    cudaEventCreate(&start_ev); cudaEventCreate(&stop_ev);
    cudaEventRecord(start_ev);

    bool found = false;
    std::string result = "";

  
    for (size_t i = 0; i < wordlist.size(); i += BATCH_SIZE) {
        int s_idx = (i / BATCH_SIZE) % NUM_STREAMS;
        int current_batch_size = (int)std::min((size_t)BATCH_SIZE, wordlist.size() - i);

        CHECK_CUDA(cudaStreamSynchronize(streams[s_idx]));
        if (*h_match[s_idx] != -1) {
            result = wordlist[batch_base_idx[s_idx] + *h_match[s_idx]];
            found = true; break;
        }

        batch_base_idx[s_idx] = i; 
        *h_match[s_idx] = -1;
        memset(h_data[s_idx], 0, BATCH_SIZE * WORDS_PER_MSG * sizeof(uint32_t));

        for (int j = 0; j < current_batch_size; j++) {
            const std::string& s = wordlist[i + j];
            size_t len = std::min(s.length(), (size_t)55);
            uint8_t temp[64] = {0};
            memcpy(temp, s.data(), len);
            temp[len] = 0x80;
            uint32_t bit_len = (uint32_t)(len * 8);
            memcpy(temp + 56, &bit_len, 4);

            uint32_t* temp32 = (uint32_t*)temp;
            for (int w = 0; w < 16; w++) {
                h_data[s_idx][w * current_batch_size + j] = temp32[w];
            }
        }

        CHECK_CUDA(cudaMemcpyAsync(d_data[s_idx], h_data[s_idx], WORDS_PER_MSG * current_batch_size * sizeof(uint32_t), cudaMemcpyHostToDevice, streams[s_idx]));
        CHECK_CUDA(cudaMemcpyAsync(d_match[s_idx], h_match[s_idx], sizeof(int), cudaMemcpyHostToDevice, streams[s_idx]));

        int tpb = 256;
        int bpg = (current_batch_size + tpb - 1) / tpb;
        md5_crack_kernel_soa<<<bpg, tpb, 0, streams[s_idx]>>>(d_data[s_idx], target_ctx, d_match[s_idx], current_batch_size);

        CHECK_CUDA(cudaMemcpyAsync(h_match[s_idx], d_match[s_idx], sizeof(int), cudaMemcpyDeviceToHost, streams[s_idx]));

        if (i % (BATCH_SIZE * 5) == 0) {
            printf("\rProgress: %.2f%%", (float)i / wordlist.size() * 100);
            fflush(stdout);
        }
    }

    if (!found) {
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamSynchronize(streams[i]);
            if (*h_match[i] != -1) {
                result = wordlist[batch_base_idx[i] + *h_match[i]];
                found = true; break;
            }
        }
    }

    cudaEventRecord(stop_ev); cudaEventSynchronize(stop_ev);
    float ms; cudaEventElapsedTime(&ms, start_ev, stop_ev);

    if (found) {
        std::cout << "\n\n[+] Cracked! Result: " << result << "\n";
    } else {
        std::cout << "\n\n[-] Hash not found in wordlist.\n";
    }
    std::cout << "[*] Speed: " << (wordlist.size() / (ms / 1000.0)) / 1e6 << " MH/s\n";

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaFreeHost(h_data[i]); cudaFreeHost(h_match[i]);
        cudaFree(d_data[i]); cudaFree(d_match[i]);
        cudaStreamDestroy(streams[i]);
    }
    return 0;
}
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <cuda_runtime.h>
#include <iomanip>

#define A0 (0x67452301)
#define B0 (0xefcdab89)
#define C0 (0x98badcfe)
#define D0 (0x10325476)
#define DIGEST_SIZE (16)
#define CHUNK_SIZE (64)
#define WORD_SIZE (4)
#define BATCH_SIZE (4096)
#define CEIL(x) ((x) == (int)(x) ? (int)(x) : ((x) > 0 ? (int)(x) + 1 : (int)(x)))

// MD5 循环左移位数
__constant__ uint32_t shift_amts[64] = {
    7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,
    5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,
    4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,
    6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21,
};

// MD5 常量表
__constant__ uint32_t k_table[64] = {
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
    0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
    0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
    0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
    0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
    0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
    0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
    0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
    0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
};

struct md5_ctx { uint32_t a, b, c, d; };
struct FfiVector { uint8_t *data; size_t len; };

md5_ctx init_ctxs[BATCH_SIZE];

__device__ uint32_t leftrotate(uint32_t x, uint32_t amt) {
    return (x << (amt % 32)) | (x >> (32 - (amt % 32)));
}

__global__ void md5_preprocess_batched(uint8_t *pre_processed_msgs, size_t *pre_processed_sizes, size_t *orig_sizes, size_t *culmn_sizes) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < BATCH_SIZE) {
        int n = orig_sizes[idx];
        int size_in_bits = 8 * n;
        int pre_processed_size = pre_processed_sizes[idx];
        pre_processed_msgs[culmn_sizes[idx] + n] = 0x80;
        for (int i = pre_processed_size - 8; i < pre_processed_size; i++) {
            int offset = i - (pre_processed_size - 8);
            pre_processed_msgs[culmn_sizes[idx] + (pre_processed_size - 8) + ((pre_processed_size - i) - 1)] = (size_in_bits >> ((7 - offset) * 8)) & 0xff;
        }
    }
}

__global__ void md5_compute_batched(md5_ctx *ctxs, uint8_t *pre_processed_msgs, size_t *pre_processed_sizes, size_t *culmn_sizes) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < BATCH_SIZE) {
        int pre_processed_size = pre_processed_sizes[idx];
        uint8_t *pre_processed_msg = pre_processed_msgs + culmn_sizes[idx];
        md5_ctx ctx = ctxs[idx];

        for (uint8_t *chunk = pre_processed_msg; chunk < pre_processed_msg + pre_processed_size; chunk += CHUNK_SIZE) {
            uint32_t words[CHUNK_SIZE / WORD_SIZE] = {0};
            for (int word_idx = 0; word_idx < CHUNK_SIZE; word_idx += WORD_SIZE) {
                words[word_idx / WORD_SIZE] = chunk[word_idx] + (chunk[word_idx + 1] << 8) + (chunk[word_idx + 2] << 16) + (chunk[word_idx + 3] << 24);
            }
            uint32_t a = ctx.a, b = ctx.b, c = ctx.c, d = ctx.d;
            for (int i = 0; i < 64; i++) {
                uint32_t f, g;
                if (i <= 15) { f = ((b & c) | ((~b) & d)); g = i; } 
                else if (i <= 31) { f = ((d & b) | ((~d) & c)); g = (5*i + 1) % 16; } 
                else if (i <= 47) { f = (b ^ c ^ d); g = (3*i + 5) % 16; } 
                else { f = (c ^ (b | (~d))); g = (7*i) % 16; }
                f = (f + a + k_table[i] + words[g]);
                a = d; d = c; c = b;
                b = (b + leftrotate(f, shift_amts[i]));
            }
            ctxs[idx].a += a; ctxs[idx].b += b; ctxs[idx].c += c; ctxs[idx].d += d;
        }
    }
}

__global__ void md5_compare_ctx_batched(md5_ctx *ctxs, md5_ctx *target_ctx, int *match_idx) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < BATCH_SIZE) {
        md5_ctx ctx = ctxs[idx];
        if (ctx.a == target_ctx->a && ctx.b == target_ctx->b && ctx.c == target_ctx->c && ctx.d == target_ctx->d) {
            *match_idx = idx;
        }
    }
}

int md5_target_batched(FfiVector *msgs, md5_ctx *h_target_ctx) {
    uint8_t *h_pre_processed_msgs, *d_pre_processed_msgs;
    size_t h_pre_processed_sizes[BATCH_SIZE], *d_pre_processed_sizes;
    size_t h_orig_sizes[BATCH_SIZE], *d_orig_sizes;
    size_t h_culmn_sizes[BATCH_SIZE], *d_culmn_sizes;
    md5_ctx *d_ctxs, *d_target_ctx;
    int *d_match_idx; int h_match_idx = -1;
    int total_size = 0;

    for (int i = 0; i < BATCH_SIZE; i++) {
        int pre_processed_size = CEIL((float)(msgs[i].len + 9) / 64.0) * 64;
        h_pre_processed_sizes[i] = pre_processed_size;
        h_orig_sizes[i] = msgs[i].len;
        h_culmn_sizes[i] = (i == 0 ? 0 : h_culmn_sizes[i - 1] + h_pre_processed_sizes[i-1]);
        total_size += pre_processed_size;
    }

    h_pre_processed_msgs = new uint8_t[total_size];
    memset(h_pre_processed_msgs, 0, total_size);
    for (int i = 0; i < BATCH_SIZE; i++) {
        if (msgs[i].len > 0) memcpy(h_pre_processed_msgs + h_culmn_sizes[i], msgs[i].data, msgs[i].len);
    }

    cudaMalloc(&d_pre_processed_msgs, total_size);
    cudaMalloc(&d_culmn_sizes, sizeof(size_t) * BATCH_SIZE);
    cudaMalloc(&d_pre_processed_sizes, sizeof(size_t) * BATCH_SIZE);
    cudaMalloc(&d_orig_sizes, sizeof(size_t) * BATCH_SIZE);
    cudaMalloc(&d_ctxs, sizeof(md5_ctx) * BATCH_SIZE);
    cudaMalloc(&d_target_ctx, sizeof(md5_ctx));
    cudaMalloc(&d_match_idx, sizeof(int));

    cudaMemcpy(d_pre_processed_msgs, h_pre_processed_msgs, total_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_culmn_sizes, h_culmn_sizes, sizeof(size_t) * BATCH_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pre_processed_sizes, h_pre_processed_sizes, sizeof(size_t) * BATCH_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_orig_sizes, h_orig_sizes, sizeof(size_t) * BATCH_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ctxs, &init_ctxs, sizeof(md5_ctx) * BATCH_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_target_ctx, h_target_ctx, sizeof(md5_ctx), cudaMemcpyHostToDevice);
    cudaMemcpy(d_match_idx, &h_match_idx, sizeof(int), cudaMemcpyHostToDevice);

    const int tpb = 256;
    const int bpg = (BATCH_SIZE + tpb - 1) / tpb;

    md5_preprocess_batched<<<bpg, tpb>>>(d_pre_processed_msgs, d_pre_processed_sizes, d_orig_sizes, d_culmn_sizes);
    md5_compute_batched<<<bpg, tpb>>>(d_ctxs, d_pre_processed_msgs, d_pre_processed_sizes, d_culmn_sizes);
    md5_compare_ctx_batched<<<bpg, tpb>>>(d_ctxs, d_target_ctx, d_match_idx);
    
    cudaMemcpy(&h_match_idx, d_match_idx, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_culmn_sizes); cudaFree(d_pre_processed_msgs); cudaFree(d_pre_processed_sizes);
    cudaFree(d_orig_sizes); cudaFree(d_ctxs); cudaFree(d_target_ctx); cudaFree(d_match_idx);
    delete[] h_pre_processed_msgs;

    return h_match_idx;
}

bool parse_hex_to_ctx(const std::string& hex, md5_ctx& ctx) {
    if (hex.length() != 32) return false;
    uint32_t raw[4];
    for (int i = 0; i < 4; i++) {
        uint32_t v; sscanf(hex.substr(i * 8, 8).c_str(), "%08x", &v);
        raw[i] = ((v & 0xFF) << 24) | ((v & 0xFF00) << 8) | ((v & 0xFF0000) >> 8) | ((v >> 24) & 0xFF);
    }
    ctx.a = raw[0]; ctx.b = raw[1]; ctx.c = raw[2]; ctx.d = raw[3];
    return true;
}

int main(int argc, char** argv) {
    if (argc != 3) return 1;
    md5_ctx target_ctx;
    if (!parse_hex_to_ctx(argv[2], target_ctx)) return 1;

    for (int i = 0; i < BATCH_SIZE; i++) {
        init_ctxs[i] = {A0, B0, C0, D0};
    }

    std::ifstream file(argv[1]);
    std::vector<std::string> wordlist;
    std::string line;
    while (std::getline(file, line)) if (!line.empty()) wordlist.push_back(line);

    std::cout << "Loaded " << wordlist.size() << " words. Starting Old-Style Crack...\n";

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    bool found = false;
    for (size_t i = 0; i < wordlist.size(); i += BATCH_SIZE) {
        size_t cur_size = std::min((size_t)BATCH_SIZE, wordlist.size() - i);
        FfiVector batch[BATCH_SIZE];
        for (size_t j = 0; j < BATCH_SIZE; j++) {
            if (j < cur_size) {
                batch[j].data = (uint8_t*)wordlist[i+j].data();
                batch[j].len = wordlist[i+j].length();
            } else {
                batch[j].data = (uint8_t*)""; batch[j].len = 0;
            }
        }

        int match = md5_target_batched(batch, &target_ctx);
        if (match != -1) {
            std::cout << "\n[+] Cracked: " << wordlist[i + match] << "\n";
            found = true; break;
        }

        if (i % (BATCH_SIZE * 10) == 0) {
            printf("\rProgress: %.2f%%", (float)i / wordlist.size() * 100);
            fflush(stdout);
        }
    }

    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms; cudaEventElapsedTime(&ms, start, stop);

    if (!found) std::cout << "\n[-] Not found.\n";
    std::cout << "[*] Speed: " << (wordlist.size() / (ms / 1000.0)) / 1e6 << " MH/s\n";

    return 0;
}
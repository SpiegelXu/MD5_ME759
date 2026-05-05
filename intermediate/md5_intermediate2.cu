#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <cuda_runtime.h>
#include <stdint.h>

#define A0 0x67452301
#define B0 0xefcdab89
#define C0 0x98badcfe
#define D0 0x10325476
#define BATCH_SIZE 1048576
#define MY_CEIL(n, d) (((n) + (d) - 1) / (d))

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


struct md5_ctx {
    uint32_t a, b, c, d;
};

struct FfiVector {
    uint8_t *data;
    size_t len;
};

// --- CUDA kernal ---
__global__ void md5_crack_kernel(
    uint8_t *raw_msgs, 
    size_t *culmn_offsets, 
    size_t *orig_sizes, 
    md5_ctx target, 
    int *match_idx, 
    int total_threads
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= total_threads) return;

    // --- early exist---
    if (*match_idx != -1) return; 

    uint32_t a_total = A0, b_total = B0, c_total = C0, d_total = D0;
    size_t len = orig_sizes[idx];
    uint8_t *my_msg = raw_msgs + culmn_offsets[idx];

    // padding length
    int pre_processed_size = MY_CEIL(len + 9, 64) * 64;

    for (int offset = 0; offset < pre_processed_size; offset += 64) {
        uint32_t in[16] = {0}; 
        
        #pragma unroll
        for (int i = 0; i < 64; i++) {
            int current_pos = offset + i;
            if (current_pos < len) {
                ((uint8_t*)in)[i] = my_msg[current_pos];
            } else if (current_pos == len) {
                ((uint8_t*)in)[i] = 0x80; 
            } 
        }

        if (offset + 64 == pre_processed_size) {
            in[14] = (uint32_t)(len * 8); 
        }

        uint32_t a = a_total, b = b_total, c = c_total, d = d_total;

        // round 1
        FF(a,b,c,d,in[0],7,0xd76aa478); FF(d,a,b,c,in[1],12,0xe8c7b756); FF(c,d,a,b,in[2],17,0x242070db); FF(b,c,d,a,in[3],22,0xc1bdceee);
        FF(a,b,c,d,in[4],7,0xf57c0faf); FF(d,a,b,c,in[5],12,0x4787c62a); FF(c,d,a,b,in[6],17,0xa8304613); FF(b,c,d,a,in[7],22,0xfd469501);
        FF(a,b,c,d,in[8],7,0x698098d8); FF(d,a,b,c,in[9],12,0x8b44f7af); FF(c,d,a,b,in[10],17,0xffff5bb1); FF(b,c,d,a,in[11],22,0x895cd7be);
        FF(a,b,c,d,in[12],7,0x6b901122); FF(d,a,b,c,in[13],12,0xfd987193); FF(c,d,a,b,in[14],17,0xa679438e); FF(b,c,d,a,in[15],22,0x49b40821);
        // round 2
        GG(a,b,c,d,in[1],5,0xf61e2562); GG(d,a,b,c,in[6],9,0xc040b340); GG(c,d,a,b,in[11],14,0x265e5a51); GG(b,c,d,a,in[0],20,0xe9b6c7aa);
        GG(a,b,c,d,in[5],5,0xd62f105d); GG(d,a,b,c,in[10],9,0x02441453); GG(c,d,a,b,in[15],14,0xd8a1e681); GG(b,c,d,a,in[4],20,0xe7d3fbc8);
        GG(a,b,c,d,in[9],5,0x21e1cde6); GG(d,a,b,c,in[14],9,0xc33707d6); GG(c,d,a,b,in[3],14,0xf4d50d87); GG(b,c,d,a,in[8],20,0x455a14ed);
        GG(a,b,c,d,in[13],5,0xa9e3e905); GG(d,a,b,c,in[2],9,0xfcefa3f8); GG(c,d,a,b,in[7],14,0x676f02d9); GG(b,c,d,a,in[12],20,0x8d2a4c8a);
        // round 3
        HH(a,b,c,d,in[5],4,0xfffa3942); HH(d,a,b,c,in[8],11,0x8771f681); HH(c,d,a,b,in[11],16,0x6d9d6122); HH(b,c,d,a,in[14],23,0xfde5380c);
        HH(a,b,c,d,in[1],4,0xa4beea44); HH(d,a,b,c,in[4],11,0x4bdecfa9); HH(c,d,a,b,in[7],16,0xf6bb4b60); HH(b,c,d,a,in[10],23,0xbebfbc70);
        HH(a,b,c,d,in[13],4,0x289b7ec6); HH(d,a,b,c,in[0],11,0xeaa127fa); HH(c,d,a,b,in[3],16,0xd4ef3085); HH(b,c,d,a,in[6],23,0x04881d05);
        HH(a,b,c,d,in[9],4,0xd9d4d039); HH(d,a,b,c,in[12],11,0xe6db99e5); HH(c,d,a,b,in[15],16,0x1fa27cf8); HH(b,c,d,a,in[2],23,0xc4ac5665);
        // round 4
        II(a,b,c,d,in[0],6,0xf4292244); II(d,a,b,c,in[7],10,0x432aff97); II(c,d,a,b,in[14],15,0xab9423a7); II(b,c,d,a,in[5],21,0xfc93a039);
        II(a,b,c,d,in[12],6,0x655b59c3); II(d,a,b,c,in[3],10,0x8f0ccc92); II(c,d,a,b,in[10],15,0xffeff47d); II(b,c,d,a,in[1],21,0x85845dd1);
        II(a,b,c,d,in[8],6,0x6fa87e4f); II(d,a,b,c,in[15],10,0xfe2ce6e0); II(c,d,a,b,in[6],15,0xa3014314); II(b,c,d,a,in[13],21,0x4e0811a1);
        II(a,b,c,d,in[4],6,0xf7537e82); II(d,a,b,c,in[11],10,0xbd3af235); II(c,d,a,b,in[2],15,0x2ad7d2bb); II(b,c,d,a,in[9],21,0xeb86d391);

        a_total += a; b_total += b; c_total += c; d_total += d;
    }

    if (a_total == target.a && b_total == target.b && c_total == target.c && d_total == target.d) {
        atomicCAS(match_idx, -1, idx); 
    }
}
// --- Host Function ---
int md5_target_batched(FfiVector *msgs, md5_ctx *h_target_ctx) {
    size_t h_culmn_offsets[BATCH_SIZE];
    size_t h_orig_sizes[BATCH_SIZE];
    size_t total_raw_size = 0;

    for (int i = 0; i < BATCH_SIZE; i++) {
        h_culmn_offsets[i] = total_raw_size;
        h_orig_sizes[i] = msgs[i].len;
        total_raw_size += msgs[i].len;
    }

    uint8_t *h_raw_msgs = new uint8_t[total_raw_size];
    for (int i = 0; i < BATCH_SIZE; i++) {
        memcpy(h_raw_msgs + h_culmn_offsets[i], msgs[i].data, msgs[i].len);
    }

    uint8_t *d_raw_msgs;
    size_t *d_culmn_offsets, *d_orig_sizes;
    int *d_match_idx;
    int h_match_idx = -1;

    cudaMalloc(&d_raw_msgs, total_raw_size);
    cudaMalloc(&d_culmn_offsets, sizeof(size_t) * BATCH_SIZE);
    cudaMalloc(&d_orig_sizes, sizeof(size_t) * BATCH_SIZE);
    cudaMalloc(&d_match_idx, sizeof(int));

    cudaMemcpy(d_raw_msgs, h_raw_msgs, total_raw_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_culmn_offsets, h_culmn_offsets, sizeof(size_t) * BATCH_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_orig_sizes, h_orig_sizes, sizeof(size_t) * BATCH_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_match_idx, &h_match_idx, sizeof(int), cudaMemcpyHostToDevice);

    const int threads_per_block = 256;
    const int blocks_per_grid = (BATCH_SIZE + threads_per_block - 1) / threads_per_block;

    md5_crack_kernel<<<blocks_per_grid, threads_per_block>>>(d_raw_msgs, d_culmn_offsets, d_orig_sizes, *h_target_ctx, d_match_idx, BATCH_SIZE);

    cudaMemcpy(&h_match_idx, d_match_idx, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_raw_msgs); cudaFree(d_culmn_offsets); cudaFree(d_orig_sizes); cudaFree(d_match_idx);
    delete[] h_raw_msgs;

    return h_match_idx;
}

bool parse_hex(const std::string& hex_str, uint8_t* output) {
    if (hex_str.length() != 32) return false;
    for (int i = 0; i < 16; i++) {
        std::string byteString = hex_str.substr(i * 2, 2);
        output[i] = (uint8_t)strtol(byteString.c_str(), NULL, 16);
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Method: " << argv[0] << " <wordlist.txt> <md5_hash>\n";
        return 1;
    }

    std::string wordlist_path = argv[1];
    std::string target_hash_hex = argv[2];

    uint8_t target_bytes[16];
    if (!parse_hex(target_hash_hex, target_bytes)) {
        std::cerr << "error, the harsh value is not the multiple of 16\n";
        return 1;
    }

    md5_ctx target_ctx;
    memcpy(&target_ctx, target_bytes, 16);

    std::ifstream file(wordlist_path);
    if (!file.is_open()) {
        std::cerr << "error can't open the file " << wordlist_path << "\n";
        return 1;
    }

    std::vector<std::string> wordlist;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) wordlist.push_back(line);
    }
    file.close();


    bool found = false;
    size_t total_words = wordlist.size();

    for (size_t i = 0; i < total_words; i += BATCH_SIZE) {
        int current_batch_size = (int)std::min((size_t)BATCH_SIZE, total_words - i);
        FfiVector batch_msgs[BATCH_SIZE]; 

        for (int j = 0; j < BATCH_SIZE; j++) {
            if (j < current_batch_size) {
                batch_msgs[j].data = (uint8_t*)wordlist[i + j].data();
                batch_msgs[j].len = wordlist[i + j].length();
            } else {
                batch_msgs[j].data = (uint8_t*)"";
                batch_msgs[j].len = 0;
            }
        }

        int match_idx = md5_target_batched(batch_msgs, &target_ctx);

        if (match_idx != -1 && match_idx < current_batch_size) {
            std::cout << "\n[+] success! \n";
            std::cout << "    word: " << wordlist[i + match_idx] << "\n";
            std::cout << "    hash: " << target_hash_hex << "\n";
            found = true;
            break;
        }

        if (i % (BATCH_SIZE * 10) == 0) {
            printf("\rprocess: %.2f%%", (float)i / total_words * 100);
            fflush(stdout);
        }
    }

    if (!found) std::cout << "\n\rprocess: 100% \n[-] Failure\n";
    return 0;
}
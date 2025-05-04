#include <iostream>
#include <unordered_map>
#include <queue>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <chrono>  // For timing measurements

#define ASCII_RANGE 256
#define MAX_CODE_LENGTH 32  // Maximum length of a Huffman code

using namespace std;
using namespace std::chrono;

// Structure for Huffman Tree Node
struct HuffmanNode {
    char data;
    int freq;
    HuffmanNode *left, *right;
    HuffmanNode(char d, int f) {
        data = d;
        freq = f;
        left = right = nullptr;
    }
};

// Comparator for the priority queue
struct Compare {
    bool operator()(HuffmanNode* l, HuffmanNode* r) {
        return l->freq > r->freq;
    }
};

// Recursively assign codes to characters
void assignCodes(HuffmanNode* root, string str, unordered_map<char, string>& huffCodes) {
    if (!root) return;
    if (!root->left && !root->right) {
        huffCodes[root->data] = str;
    }
    assignCodes(root->left, str + "0", huffCodes);
    assignCodes(root->right, str + "1", huffCodes);
}

// CUDA Kernel: Compute frequency of each character in parallel
__global__ void computeFrequency(char* input, int* freq, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        atomicAdd(&freq[(unsigned char)input[idx]], 1);
    }
}

// CUDA Kernel: Encode characters using Huffman codes
__global__ void encodeKernel(char* input, int* codeLengths, char* codeData, char* encoded, int size, int* pos) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        unsigned char ch = (unsigned char)input[idx];
        int codeLen = codeLengths[ch];
        
        if (codeLen > 0) {
            // Get the starting position of this character's code in the codeData array
            int codeOffset = 0;
            for (int i = 0; i < ch; i++) {
                codeOffset += codeLengths[i];
            }
            
            // Copy code into output buffer with atomic operation to handle concurrency
            int offset = atomicAdd(pos, codeLen);
            for (int i = 0; i < codeLen; i++) {
                encoded[offset + i] = codeData[codeOffset + i];
            }
        }
    }
}

// Function to recursively free the Huffman tree
void freeHuffmanTree(HuffmanNode* root) {
    if (root == nullptr) return;
    
    freeHuffmanTree(root->left);
    freeHuffmanTree(root->right);
    
    delete root;
}

int main() {
    // Start timing the entire execution
    auto programStartTime = high_resolution_clock::now();
    
    // Variables to track CPU and GPU time
    long long cpuTime = 0;
    long long gpuTime = 0;
    auto startTime = high_resolution_clock::now();
    auto endTime = high_resolution_clock::now();
    
    // CPU: Input handling
    startTime = high_resolution_clock::now();
    string input = "";
    cout << "Enter the input string: ";
    getline(cin, input);  // Using getline to handle spaces in the input
    int size = input.size();
    endTime = high_resolution_clock::now();
    cpuTime += duration_cast<milliseconds>(endTime - startTime).count();
    
    // GPU: Memory allocation and input transfer
    startTime = high_resolution_clock::now();
    char* d_input;
    int* d_freq;
    
    // Allocate device memory
    cudaMalloc((void**)&d_input, size * sizeof(char));
    cudaMalloc((void**)&d_freq, ASCII_RANGE * sizeof(int));
    cudaMemset(d_freq, 0, ASCII_RANGE * sizeof(int));
    
    // Copy input string to device
    cudaMemcpy(d_input, input.c_str(), size * sizeof(char), cudaMemcpyHostToDevice);
    endTime = high_resolution_clock::now();
    gpuTime += duration_cast<milliseconds>(endTime - startTime).count();
    
    // GPU: Compute character frequencies
    startTime = high_resolution_clock::now();
    computeFrequency<<<(size + 255)/256, 256>>>(d_input, d_freq, size);
    cudaDeviceSynchronize();
    endTime = high_resolution_clock::now();
    gpuTime += duration_cast<milliseconds>(endTime - startTime).count();
    
    // GPU to CPU: Copy frequency data
    startTime = high_resolution_clock::now();
    int h_freq[ASCII_RANGE] = {0};
    cudaMemcpy(h_freq, d_freq, ASCII_RANGE * sizeof(int), cudaMemcpyDeviceToHost);
    endTime = high_resolution_clock::now();
    gpuTime += duration_cast<milliseconds>(endTime - startTime).count();
    
    // CPU: Build Huffman Tree
    startTime = high_resolution_clock::now();
    priority_queue<HuffmanNode*, vector<HuffmanNode*>, Compare> pq;
    for (int i = 0; i < ASCII_RANGE; i++) {
        if (h_freq[i]) {
            pq.push(new HuffmanNode((char)i, h_freq[i]));
        }
    }
    
    while (pq.size() > 1) {
        HuffmanNode* left = pq.top(); pq.pop();
        HuffmanNode* right = pq.top(); pq.pop();
        HuffmanNode* merged = new HuffmanNode('\0', left->freq + right->freq);
        merged->left = left;
        merged->right = right;
        pq.push(merged);
    }
    
    HuffmanNode* root = pq.top();
    
    // Assign codes to characters
    unordered_map<char, string> huffCodes;
    assignCodes(root, "", huffCodes);
    
    // Print the Huffman Codes
    cout << "Huffman Codes:\n";
    for (auto& pair : huffCodes) {
        cout << pair.first << ": " << pair.second << "\n";
    }
    endTime = high_resolution_clock::now();
    cpuTime += duration_cast<milliseconds>(endTime - startTime).count();
    
    // CPU: Create data structures for GPU encoding
    startTime = high_resolution_clock::now();
    int codeLengths[ASCII_RANGE] = {0};
    int totalCodeLength = 0;
    
    // Calculate total length of all codes and individual lengths
    for (auto& pair : huffCodes) {
        codeLengths[(unsigned char)pair.first] = pair.second.length();
        totalCodeLength += pair.second.length();
    }
    
    // Create a flat array to store all code data
    char* codeData = new char[totalCodeLength];
    int codeOffset = 0;
    
    // Fill the code data array
    for (int i = 0; i < ASCII_RANGE; i++) {
        if (codeLengths[i] > 0) {
            string code = huffCodes[(char)i];
            for (size_t j = 0; j < code.length(); j++) {
                codeData[codeOffset + j] = code[j];
            }
            codeOffset += code.length();
        }
    }
    endTime = high_resolution_clock::now();
    cpuTime += duration_cast<milliseconds>(endTime - startTime).count();
    
    // GPU: Allocate memory for encoding
    startTime = high_resolution_clock::now();
    int* d_codeLengths;
    char* d_codeData;
    char* d_encoded;
    int* d_pos;
    
    cudaMalloc((void**)&d_codeLengths, ASCII_RANGE * sizeof(int));
    cudaMalloc((void**)&d_codeData, totalCodeLength * sizeof(char));
    
    // Calculate maximum length of encoded string (worst case: all codes are the maximum length)
    int maxEncodedLength = size * MAX_CODE_LENGTH;
    cudaMalloc((void**)&d_encoded, maxEncodedLength * sizeof(char));
    cudaMalloc((void**)&d_pos, sizeof(int));
    
    // Initialize position counter
    int h_pos = 0;
    cudaMemcpy(d_pos, &h_pos, sizeof(int), cudaMemcpyHostToDevice);
    
    // Copy code information to device
    cudaMemcpy(d_codeLengths, codeLengths, ASCII_RANGE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_codeData, codeData, totalCodeLength * sizeof(char), cudaMemcpyHostToDevice);
    endTime = high_resolution_clock::now();
    gpuTime += duration_cast<milliseconds>(endTime - startTime).count();
    
    // GPU: Launch encoding kernel
    startTime = high_resolution_clock::now();
    encodeKernel<<<(size + 255)/256, 256>>>(d_input, d_codeLengths, d_codeData, d_encoded, size, d_pos);
    cudaDeviceSynchronize();
    endTime = high_resolution_clock::now();
    gpuTime += duration_cast<milliseconds>(endTime - startTime).count();
    
    // GPU to CPU: Get encoded data
    startTime = high_resolution_clock::now();
    cudaMemcpy(&h_pos, d_pos, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Copy encoded result back to host
    char* h_encoded = new char[h_pos + 1];
    cudaMemcpy(h_encoded, d_encoded, h_pos * sizeof(char), cudaMemcpyDeviceToHost);
    h_encoded[h_pos] = '\0';  // Add null terminator
    endTime = high_resolution_clock::now();
    gpuTime += duration_cast<milliseconds>(endTime - startTime).count();
    
    // CPU: Verify and print results
    startTime = high_resolution_clock::now();
    // Verify encoding by comparing with CPU encoding
    string cpuEncodedStr = "";
    for (char ch : input) {
        cpuEncodedStr += huffCodes[ch];
    }

    // Print encoded result
    cout << "CPU encoded string: ";
    if (cpuEncodedStr.length() > 100) {
        cout << cpuEncodedStr.substr(0, 100) << "..." << endl;
    } else {
        cout << cpuEncodedStr << endl;
    }
    
    cout << "GPU encoded string: ";
    int printLen = (h_pos > 100) ? 100 : h_pos;
    for (int i = 0; i < printLen; i++) {
        cout << h_encoded[i];
    }
    if (h_pos > 100) {
        cout << "...";
    }
    cout << endl;
    
    // cout << "Encoding verification: " << (string(h_encoded) == cpuEncodedStr ? "PASSED" : "FAILED") << endl;
    endTime = high_resolution_clock::now();
    cpuTime += duration_cast<milliseconds>(endTime - startTime).count();
    
    // Clean up
    startTime = high_resolution_clock::now();
    
    // GPU cleanup
    cudaFree(d_input);
    cudaFree(d_freq);
    cudaFree(d_codeLengths);
    cudaFree(d_codeData);
    cudaFree(d_encoded);
    cudaFree(d_pos);
    
    // CPU cleanup
    delete[] codeData;
    delete[] h_encoded;
    
    // Free Huffman tree nodes
    freeHuffmanTree(root);
    
    endTime = high_resolution_clock::now();
    // Split cleanup time evenly between CPU and GPU
    long long cleanupTime = duration_cast<milliseconds>(endTime - startTime).count();
    cpuTime += cleanupTime / 2;
    gpuTime += cleanupTime / 2;
    
    // Calculate and print total execution time
    auto programEndTime = high_resolution_clock::now();
    auto totalDurationMs = duration_cast<milliseconds>(programEndTime - programStartTime).count();
    double totalDurationSec = totalDurationMs / 1000.0;
    
    cout << "\n===== Execution Time Summary =====\n";
    cout << "CPU processing time: " << cpuTime << " milliseconds\n";
    cout << "GPU processing time: " << gpuTime << " milliseconds\n";
    cout << "Total execution time: " << totalDurationSec << " seconds\n";
    
    return 0;
}
/*
To run this code follow the steps
1. Compile using : nvcc huffman.cu
2. Run using     : ./a.out

Major Operations Performed on CPU
1. Huffman Tree Construction
2. Code Assignment to each character

Operations Performed on GPU
1. Character Frequency Counting
2. String Encoding
*/


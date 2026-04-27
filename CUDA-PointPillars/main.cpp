/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <dirent.h>
#include <chrono>

#include "cuda_runtime.h"

#include "./params.h"
#include "./pointpillar.h"

#define checkCudaErrors(status)                                   \
{                                                                 \
  if (status != 0)                                                \
  {                                                               \
    std::cout << "Cuda failure: " << cudaGetErrorString(status)   \
              << " at line " << __LINE__                          \
              << " in file " << __FILE__                          \
              << " error status: " << status                      \
              << std::endl;                                       \
              abort();                                            \
    }                                                             \
}

// std::string Data_File = "../data/";   // original: hardcoded data path
// std::string Save_Dir = "../eval/kitti/object/pred_velo/";  // original: hardcoded save path
std::string Model_File = "../model/pointpillar.onnx";

std::vector<std::string> list_bin_files(const std::string& dir) {
  std::vector<std::string> files;
  DIR* dp = opendir(dir.c_str());
  if (!dp) { std::cerr << "Cannot open data dir: " << dir << std::endl; return files; }
  struct dirent* ep;
  while ((ep = readdir(dp))) {
    std::string name = ep->d_name;
    if (name.size() > 4 && name.substr(name.size() - 4) == ".bin")
      files.push_back(dir + "/" + name);
  }
  closedir(dp);
  std::sort(files.begin(), files.end());
  return files;
}

void Getinfo(void)
{
  cudaDeviceProp prop;

  int count = 0;
  cudaGetDeviceCount(&count);
  printf("\nGPU has cuda devices: %d\n", count);
  for (int i = 0; i < count; ++i) {
    cudaGetDeviceProperties(&prop, i);
    printf("----device id: %d info----\n", i);
    printf("  GPU : %s \n", prop.name);
    printf("  Capbility: %d.%d\n", prop.major, prop.minor);
    printf("  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
    printf("  Const memory: %luKB\n", prop.totalConstMem  >> 10);
    printf("  SM in a block: %luKB\n", prop.sharedMemPerBlock >> 10);
    printf("  warp size: %d\n", prop.warpSize);
    printf("  threads in a block: %d\n", prop.maxThreadsPerBlock);
    printf("  block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
  }
  printf("\n");
}

int loadData(const char *file, void **data, unsigned int *length)
{
  std::fstream dataFile(file, std::ifstream::in);

  if (!dataFile.is_open())
  {
	  std::cout << "Can't open files: "<< file<<std::endl;
	  return -1;
  }

  //get length of file:
  unsigned int len = 0;
  dataFile.seekg (0, dataFile.end);
  len = dataFile.tellg();
  dataFile.seekg (0, dataFile.beg);

  //allocate memory:
  char *buffer = new char[len];
  if(buffer==NULL) {
	  std::cout << "Can't malloc buffer."<<std::endl;
    dataFile.close();
	  exit(-1);
  }

  //read data as a block:
  dataFile.read(buffer, len);
  dataFile.close();

  *data = (void*)buffer;
  *length = len;
  return 0;  
}

void SaveBoxPred(std::vector<Bndbox> boxes, std::string file_name)
{
    std::ofstream ofs;
    ofs.open(file_name, std::ios::out);
    if (ofs.is_open()) {
        for (const auto box : boxes) {
          ofs << box.x << " ";
          ofs << box.y << " ";
          ofs << box.z << " ";
          ofs << box.w << " ";
          ofs << box.l << " ";
          ofs << box.h << " ";
          ofs << box.rt << " ";
          ofs << box.id << " ";
          ofs << box.score << " ";
          ofs << "\n";
        }
    }
    else {
      std::cerr << "Output file cannot be opened!" << std::endl;
    }
    ofs.close();
    std::cout << "Saved prediction in: " << file_name << std::endl;
    return;
};

// ---- original main (hardcoded 10 frames, single subprocess) ----
// int main(int argc, const char **argv)
// {
//   Getinfo();
//   cudaEvent_t start, stop;
//   float elapsedTime = 0.0f;
//   cudaStream_t stream = NULL;
//   checkCudaErrors(cudaEventCreate(&start));
//   checkCudaErrors(cudaEventCreate(&stop));
//   checkCudaErrors(cudaStreamCreate(&stream));
//   Params params_;
//   std::vector<Bndbox> nms_pred;
//   nms_pred.reserve(100);
//   PointPillar pointpillar(Model_File, stream);
//   for (int i = 0; i < 10; i++) {
//     std::string dataFile = Data_File;
//     std::stringstream ss; ss << i;
//     int n_zero = 6; std::string _str = ss.str();
//     std::string index_str = std::string(n_zero - _str.length(), '0') + _str;
//     dataFile += index_str; dataFile += ".bin";
//     std::cout << "<<<<<<<<<<<" << std::endl;
//     std::cout << "load file: " << dataFile << std::endl;
//     unsigned int length = 0; void *data = NULL;
//     std::shared_ptr<char> buffer((char *)data, std::default_delete<char[]>());
//     loadData(dataFile.data(), &data, &length);
//     buffer.reset((char *)data);
//     float* points = (float*)buffer.get();
//     size_t points_size = length/sizeof(float)/4;
//     std::cout << "find points num: " << points_size << std::endl;
//     float *points_data = nullptr;
//     unsigned int points_data_size = points_size * 4 * sizeof(float);
//     checkCudaErrors(cudaMallocManaged((void **)&points_data, points_data_size));
//     checkCudaErrors(cudaMemcpy(points_data, points, points_data_size, cudaMemcpyDefault));
//     checkCudaErrors(cudaDeviceSynchronize());
//     cudaEventRecord(start, stream);
//     pointpillar.doinfer(points_data, points_size, nms_pred);
//     cudaEventRecord(stop, stream);
//     cudaEventSynchronize(stop);
//     cudaEventElapsedTime(&elapsedTime, start, stop);
//     std::cout << "TIME: pointpillar: " << elapsedTime << " ms." << std::endl;
//     checkCudaErrors(cudaFree(points_data));
//     std::cout << "Bndbox objs: " << nms_pred.size() << std::endl;
//     std::string save_file_name = Save_Dir + index_str + ".txt";
//     SaveBoxPred(nms_pred, save_file_name);
//     nms_pred.clear();
//     std::cout << ">>>>>>>>>>>" << std::endl;
//   }
//   checkCudaErrors(cudaEventDestroy(start));
//   checkCudaErrors(cudaEventDestroy(stop));
//   checkCudaErrors(cudaStreamDestroy(stream));
//   return 0;
// }
// ---- end original main ----

// New main: persistent process with --data-dir, --warmup, --repeat, --save-preds
// --warmup N and --repeat N are individual FRAME counts (same as profile_suite.py steps).
// Files are cycled if needed (e.g. 10 bundled files × 50 cycles = 500 frames).
// Usage:
//   Latency:  ./demo --data-dir /mnt/kitti/training/velodyne --warmup 500 --repeat 500
//   Accuracy: ./demo --data-dir /mnt/kitti/training/velodyne --warmup 0 --repeat 7481 --save-preds --save-dir /mnt/preds
int main(int argc, const char **argv)
{
  std::string data_dir  = "../data";
  std::string save_dir  = "../eval/kitti/object/pred_velo/";
  int warmup_frames = 500;
  int repeat_frames = 500;
  bool save_preds = false;

  for (int a = 1; a < argc; a++) {
    std::string arg = argv[a];
    if      (arg == "--data-dir"  && a+1 < argc) data_dir      = argv[++a];
    else if (arg == "--save-dir"  && a+1 < argc) save_dir      = argv[++a];
    else if (arg == "--warmup"    && a+1 < argc) warmup_frames = atoi(argv[++a]);
    else if (arg == "--repeat"    && a+1 < argc) repeat_frames = atoi(argv[++a]);
    else if (arg == "--save-preds") save_preds = true;
  }

  auto bin_files = list_bin_files(data_dir);
  if (bin_files.empty()) {
    std::cerr << "No .bin files found in: " << data_dir << std::endl;
    return -1;
  }
  int n_files = (int)bin_files.size();
  int total_frames = warmup_frames + repeat_frames;

  std::cout << "[info] data_dir=" << data_dir << "  files=" << n_files
            << "  warmup_frames=" << warmup_frames
            << "  repeat_frames=" << repeat_frames << std::endl;

  Getinfo();

  cudaEvent_t start, stop;
  float elapsedTime = 0.0f;
  cudaStream_t stream = NULL;

  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaStreamCreate(&stream));

  std::vector<Bndbox> nms_pred;
  nms_pred.reserve(100);

  // Load TRT engine once — matches teammate's persistent Python process
  PointPillar pointpillar(Model_File, stream);

  // Iterate frame-by-frame; cycle through files if repeat > n_files
  for (int frame = 0; frame < total_frames; frame++) {
    bool is_warmup = (frame < warmup_frames);
    const std::string& file_path = bin_files[frame % n_files];

    unsigned int length = 0;
    void *data = NULL;
    auto t_read0 = std::chrono::high_resolution_clock::now();
    std::shared_ptr<char> buffer((char *)data, std::default_delete<char[]>());
    loadData(file_path.c_str(), &data, &length);
    buffer.reset((char *)data);
    auto t_read1 = std::chrono::high_resolution_clock::now();
    float read_ms = std::chrono::duration<float, std::milli>(t_read1 - t_read0).count();

    float* points = (float*)buffer.get();
    size_t points_size = length/sizeof(float)/4;

    float *points_data = nullptr;
    unsigned int points_data_size = points_size * 4 * sizeof(float);
    checkCudaErrors(cudaMallocManaged((void **)&points_data, points_data_size));
    checkCudaErrors(cudaMemcpy(points_data, points, points_data_size, cudaMemcpyDefault));
    checkCudaErrors(cudaDeviceSynchronize());

    cudaEventRecord(start, stream);
    pointpillar.doinfer(points_data, points_size, nms_pred);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    if (!is_warmup) {
      std::cout << "TIME: read_points: " << read_ms << " ms." << std::endl;
      std::cout << "TIME: pointpillar: " << elapsedTime << " ms." << std::endl;
    }

    if (save_preds && !is_warmup) {
      std::string stem = file_path.substr(file_path.rfind('/') + 1);
      stem = stem.substr(0, stem.size() - 4);
      SaveBoxPred(nms_pred, save_dir + stem + ".txt");
    }

    checkCudaErrors(cudaFree(points_data));
    nms_pred.clear();
  }

  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  checkCudaErrors(cudaStreamDestroy(stream));

  return 0;
}

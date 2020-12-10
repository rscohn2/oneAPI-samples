#include <iomanip>
#include <iostream>
#include <fstream>

#include "stencil.hpp"

int halo_size = 1;
// Data with halo
int data_size;

int num_points = 100;
int num_steps = 1000;
int num_gpus = 1;

float *input;
float *reference_output;

void Usage(const char *programName) {
  std::cout << " Usage: " << programName << " [points] [steps] [GPUs]\n";
}

void serial_stencil() {
  // Allocate and initialize 2 arrays to ping/pong computation
  float *data[2];
  for (int i = 0; i < 2; i++) {
    data[i] = new float[data_size];
    for (int j = 0; j < data_size; j++) {
      data[i][j] = input[j];
    }
  }
  
  // Perform stencil
  float *in = data[0];
  float *out = data[1];
  for (int step = 0; step < num_steps; step++) {
    for (int j = 1; j <= num_points; j++) {
      compute_point(out+j, in+j);
    }
    std::swap(in, out);
  }

  // Copy results to reference
  for (int j = 0; j < data_size; j++) {
    reference_output[j] = in[j];
  }

  delete [] data[0];
  delete [] data[1];
}

void initialize() {
      data_size = num_points + 2 * halo_size;
    std::cout << "Configuration:\n"
	      << "  points: " << num_points << "\n"
	      << "  steps: " << num_steps << "\n"
	      << "  GPUs: " << num_gpus << "\n";

  // Initialize input including halo
  input = new float[data_size];
  for (int i = 1; i < data_size; i++)
    input[i] = 0.0;
  input[0] = 100.0;

  // Reference output
  reference_output = new float[data_size];
  serial_stencil();

}

void deinitialize() {
  delete [] input;
  delete [] reference_output;
}

void parse_args(int argc, char *argv[]) {
  try {
    if (argc > 1)
      num_points = std::stoi(argv[1]);
    if (argc > 2)
      num_steps = std::stoi(argv[2]);
    if (argc > 3)
      num_gpus = std::stoi(argv[3]);
    if (num_points > 0
	&& num_steps > 0
	&& num_gpus > 0
	&& argc <= 4)
      return;
  } catch (...) {
    ;
  }

  Usage(argv[0]);
  exit(1);
}



std::chrono::steady_clock::time_point time_begin(std::string name) {
  std::cout << name << "\n";
  return std::chrono::steady_clock::now();
}

void time_end(std::chrono::steady_clock::time_point start) {
  auto now = std::chrono::steady_clock::now();
  double duration =
    std::chrono::duration_cast<std::chrono::duration<double>>(now - start).count();
  std::cout << "  elapsed: " << duration << "\n";
}

void check(std::string name, float *output) {
  int errors = 0;
  float error_threshold = .001;
  
  std::ofstream results(name + ".out");
  results << name << "\n"
	  << "           reference          test       delta\n";
  results << std::fixed;
  for (int i = 0; i < data_size; i++) {
    results << std::setw(8) << i
	    << std::setw(12) << reference_output[i]
	    << "  " << std::setw(12) << output[i];
    float delta = reference_output[i] - output[i];
    if (delta > error_threshold) {
      results << std::setw(12) << delta << "\n";
      errors += 1;
    } else {
      results << "\n";
    }
  }

  std::cout << "  " << (errors ? "Fail\n" : "Pass\n");
}

std::chrono::steady_clock::time_point begin(std::string name) {
  std::cout << name << "\n";
  return std::chrono::steady_clock::now();
}


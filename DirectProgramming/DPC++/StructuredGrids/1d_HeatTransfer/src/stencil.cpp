#include <stdlib.h>
#include <string>

#include <CL/sycl.hpp>

#include "stencil.hpp"

void USM_stencil() {
  std::string name("USM_stencil");
  
  sycl::queue q;

  // Allocate and initialize 2 arrays to ping/pong computation
  float *data[2];
  for (int i = 0; i < 2; i++) {
    data[i] = sycl::malloc_shared<float>(data_size, q);
    for (int j = 0; j < data_size; j++) {
      data[i][j] = input[j];
    }
  }
  
  auto begin = time_begin(name);

  // Perform stencil
  float *in = data[0];
  float *out = data[1];
  for (int step = 0; step < num_steps; step++) {
    q.parallel_for(num_points,
		   [=](auto id) {
		     int j = id + 1;
		     compute_point(out+j, in+j);
		   });
    q.wait();
    std::swap(in, out);
  }
  time_end(begin);

  check(name, in);
  
  for (int j = 0; j < data_size; j++) {
    reference_output[j] = in[j];
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);
  initialize();
  
  USM_stencil();

  deinitialize();
}

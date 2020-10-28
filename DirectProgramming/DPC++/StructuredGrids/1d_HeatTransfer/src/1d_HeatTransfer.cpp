//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
//
// 1D HEAT TRANSFER: Using Intel® oneAPI DPC++ Language to simulate 1D Heat
// Transfer.
//
// The code sample simulates the heat propagation according to the following
// equation (case where there is no heat generation):
//
//    dU/dt = k * d2U/dx2
//    (u(x,t+DT) - u(x,t)) / DT = k * (u(x+DX,t)- 2u(x,t) + u(x-DX,t)) / DX2
//    U(i) = C * (U(i+1) - 2 * U(i) + U(i-1)) + U(i)
//
// where constant C = k * dt / (dx * dx)
//
// For comprehensive instructions regarding DPC++ Programming, go to
// https://software.intel.com/en-us/oneapi-programming-guide
// and search based on relevant terms noted in the comments.
//
// DPC++ material used in this code sample:
//
// Basic structures of DPC++:
//   DPC++ Queues (including device selectors and exception handlers)
//   DPC++ Buffers and accessors (communicate data between the host and the
//   device)
//   DPC++ Kernels (including parallel_for function and range<1> objects)
//
//******************************************************************************
// Content: (version 1.1)
//   1d_HeatTransfer.cpp
//
//******************************************************************************
#include <CL/sycl.hpp>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/<version>/include/dpc_common.hpp
#include "dpc_common.hpp"

#define FAKE_GPUS 7

using namespace sycl;
using namespace std;

constexpr float dt = 0.002f;
constexpr float dx = 0.01f;
constexpr float k = 0.025f;
constexpr float initial_temperature = 100.0f; // Initial temperature.

int failures = 0;

//
// Display input parameters used for this sample.
//
void Usage(const string &programName) {
  cout << " Incorrect parameters \n";
  cout << " Usage: ";
  cout << programName << " <n> <i>\n\n";
  cout << " n : Number of points to simulate \n";
  cout << " i : Number of timesteps \n";
}

//
// Initialize the temperature arrays
//
void Initialize(float *arr, float *arr_next, size_t num) {
  for (size_t i = 1; i < num; i++)
    arr[i] = arr_next[i] = 0.0f;
  arr[0] = arr_next[0] = initial_temperature;
}

//
// Compare host and device results
//
void CompareResults(string prefix, float *device_results, float *host_results,
                    size_t num_point, float C) {
  string path = prefix + "_error_diff.txt";
  float delta = 0.001f;
  float difference = 0.00f;
  double norm2 = 0;
  bool err = false;

  ofstream err_file;
  err_file.open(path);

  err_file << " \t idx\theat[i]\t\theat_CPU[i] \n";

  for (size_t i = 0; i < num_point + 2; i++) {
    err_file << "\n RESULT: " << i << "\t" << std::setw(12) << std::left
             << device_results[i] << "\t" << host_results[i];

    difference = fabsf(host_results[i] - device_results[i]);
    norm2 += difference * difference;

    if (difference > delta) {
      err = true;
      err_file << ", diff: " << difference;
    }
  }

  if (err) {
    cout << "  FAIL! Please check " << path << "\n";
    failures++;
  } else
    cout << "  PASSED!\n";
}

//
// Compute heat on the device using DPC++ buffer
//
void ComputeHeatBuffer(float C, size_t num_p, size_t num_iter, float *arr_CPU) {
  // Define device selector as 'default'
  default_selector device_selector;

  // Create a device queue using DPC++ class queue
  queue q(device_selector, dpc_common::exception_handler);
  cout << "Using buffers\n";
  cout << "  Kernel runs on " << q.get_device().get_info<info::device::name>()
       << "\n";

  // Temperatures of the current and next iteration
  float *arr_host = new float[num_p + 2];
  float *arr_host_next = new float[num_p + 2];

  Initialize(arr_host, arr_host_next, num_p + 2);

  // Start timer
  dpc_common::TimeInterval t_par;

  auto *arr_buf = new buffer<float>(arr_host, range(num_p + 2));
  auto *arr_buf_next = new buffer<float>(arr_host_next, range(num_p + 2));

  // Iterate over timesteps
  for (size_t i = 0; i < num_iter; i++) {
    auto cg = [&](auto &h) {
      accessor arr(*arr_buf, h);
      accessor arr_next(*arr_buf_next, h);
      auto step = [=](id<1> idx) {
        size_t k = idx + 1;

        if (k == num_p + 1) {
          arr_next[k] = arr[k - 1];
        } else {
          arr_next[k] = C * (arr[k + 1] - 2 * arr[k] + arr[k - 1]) + arr[k];
        }
      };

      h.parallel_for(range{num_p + 1}, step);
    };
    q.submit(cg);

    // Swap arrays for next step
    swap(arr_buf, arr_buf_next);
  }

  // Deleting will wait for tasks to complete and write data back to host
  // Write back is not needed for arr_buf_next
  arr_buf_next->set_write_back(false);
  delete arr_buf;
  delete arr_buf_next;

  // Display time used to process all time steps
  cout << "  Elapsed time: " << t_par.Elapsed() << " sec\n";

  CompareResults("buffer", ((num_iter % 2) == 0) ? arr_host : arr_host_next,
                 arr_CPU, num_p, C);

  delete[] arr_host;
  delete[] arr_host_next;
}

//
// Compute heat on the device using USM
//
void ComputeHeatUSM(float C, size_t num_p, size_t num_iter, float *arr_CPU) {
  // Timesteps depend on each other, so make the queue inorder
  property_list properties{property::queue::in_order()};
  // Define device selector as 'default'
  default_selector device_selector;

  // Create a device queue using DPC++ class queue
  queue q(device_selector, dpc_common::exception_handler, properties);
  cout << "Using USM\n";
  cout << "  Kernel runs on " << q.get_device().get_info<info::device::name>()
       << "\n";

  // Temperatures of the current and next iteration
  float *arr = malloc_shared<float>(num_p + 2, q);
  float *arr_next = malloc_shared<float>(num_p + 2, q);

  Initialize(arr, arr_next, num_p + 2);

  // Start timer
  dpc_common::TimeInterval time;

  // for each timesteps
  for (size_t i = 0; i < num_iter; i++) {
    auto step = [=](id<1> idx) {
      size_t k = idx + 1;
      if (k == num_p + 1)
        arr_next[k] = arr[k - 1];
      else
        arr_next[k] = C * (arr[k + 1] - 2 * arr[k] + arr[k - 1]) + arr[k];
    };

    q.parallel_for(range{num_p + 1}, step);

    // Swap arrays for next step
    swap(arr, arr_next);
  }

  // Wait for all the timesteps to complete
  q.wait_and_throw();

  // Display time used to process all time steps
  cout << "  Elapsed time: " << time.Elapsed() << " sec\n";

  CompareResults("usm", arr, arr_CPU, num_p, C);

  free(arr, q);
  free(arr_next, q);
}

//
// Returns a vector of SYCL devices including the most capable device
// and all devices on the same platform with the same compute power
//
vector<device> GetDevices() {
  vector<device> devices;

  // Let the runtime pick the most capable device
  device d(default_selector{});

  auto p = d.get_platform();
  cout << "  Platform: " << p.get_info<info::platform::name>() << "\n";
  auto compute_units = d.get_info<info::device::max_compute_units>();
  for (auto &d : p.get_devices()) {
    // Add all the devices from the same platform that match in compute power
    if (d.get_info<info::device::max_compute_units>() == compute_units) {
      devices.push_back(d);
      cout << "    " << d.get_info<info::device::name>() << "\n";
      //#define GPU_LIMIT 1
#if defined(GPU_LIMIT)
      if (devices.size() == GPU_LIMIT)
        break;
#endif
    }
  }

#if defined(FAKE_GPUS) && FAKE_GPUS > 0
  // Simulate a parallel system by duplicating the same device
  // device
  for (int i = 0; i < FAKE_GPUS; i++)
    devices.push_back(d);
#endif

  cout << "  Number of Devices: " << devices.size() << "\n";
  if (devices.size() == 0) {
    cout << "  No devices available.\n";
  }

  return devices;
}

//
// We ping-pong between 2 copies of data so we don't overwrite our
// inputs
//
class InOut {
public:
  float *data; // temperature array
  event step;  // event for this time step
};

//
// Holds information for each device
//
class Node {
public:
  int index;
  Node *left;
  Node *right;
  queue queue;
  InOut inout[2];     // input and output data
  InOut *in;          // in data for this timestep
  InOut *out;         // out data for this timestep
  size_t num_p;       // number of points for this device
  size_t host_offset; // offset into host data for this node
};

//
// Compute heat on the device using multiple devices
//
void ComputeHeatMultiDevice(float C, size_t num_p, size_t num_iter,
                            float *arr_CPU) {
  cout << "Using multiple devices\n";

  // Create the initial temperature on host
  float *arr_host[2];
  arr_host[0] = new float[num_p + 2];
  arr_host[1] = new float[num_p + 2];
  Initialize(arr_host[0], arr_host[1], num_p + 2);

  //
  // Create a vector of nodes, one for each device.  Allocate queues,
  // memory, and initialize the memory from the host.
  //

  vector<device> devices = GetDevices();
  int num_devices = devices.size();

  // Divide points evenly among devices. Distribute remainder starting
  // from node 0
  size_t device_p = num_p / num_devices;
  size_t remainder_p = num_p % num_devices;
  vector<Node> nodes(num_devices);
  property_list q_prop{property::queue::in_order()};
  size_t host_offset = 1;
  for (int i = 0; i < num_devices; i++) {
    Node &n = nodes[i];
    device &d = devices[i];
    n.left = (i == 0 ? nullptr : &nodes[i - 1]);
    n.right = (i == num_devices-1 ? nullptr : &nodes[i + 1]);
    
    if (i != num_devices - 1)
      n.right = &nodes[i + 1];
    n.num_p = device_p + (i < remainder_p);
    n.queue = queue(d, dpc_common::exception_handler, q_prop);
    n.host_offset = host_offset;
    n.in = &n.inout[0];
    n.out = &n.inout[1];
    n.index = i;
    for (int i = 0; i < 2; i++) {
      float *data = malloc_device<float>(n.num_p + 2, n.queue);
      n.inout[i].data = data;
      event e = n.queue.memcpy(data, arr_host[i] + host_offset - 1,
			       sizeof(float) * (n.num_p + 2));
      e.wait();
    }

    host_offset += n.num_p;
  }
  
  //
  // Computation
  //

  // Start timer
  dpc_common::TimeInterval time;

  // for each timestep
  for (size_t i = 0; i < num_iter; i++) {
    // for each device
    for (auto &n : nodes) {
#if FAKE_GPUS > 0
      auto &q = nodes[0].queue;
#else
      auto &q = n.queue;
#endif
      auto in = n.in->data;
      auto out = n.out->data;

      // Update the temperature
      auto step = [=](id<1> idx) {
        size_t k = idx + 1;
        out[k] = C * (in[k + 1] - 2 * in[k] + in[k - 1]) + in[k];
      };
      auto cg = [=](handler &h) {
		  if (n.left) {
		    h.depends_on(n.left->in->step);
		  }
		  if (n.right) {
		    h.depends_on(n.right->in->step);
		  }
        h.parallel_for(range{n.num_p}, step);
      };
      n.out->step = q.submit(cg);

      // Update the halo
      if (n.left) {
        // Send halo left
        Node &l = *n.left;
        n.out->step =
            q.memcpy(&(l.out->data[l.num_p + 1]), &out[1], sizeof(float));
      }
      if (n.right) {
        // Send halo right
        Node &r = *n.right;
        n.out->step = q.memcpy(&(r.out->data[0]), &out[n.num_p], sizeof(float));
      } else {
        // no device on the right, update my halo
        n.out->step = q.memcpy(&out[n.num_p + 1], &in[n.num_p], sizeof(float));
      }
    }

    // Swap in/out for next step
    for (auto &n : nodes) {
      swap(n.in, n.out);
    }
  }

  // Wait for all devices to finish
  for (auto &n : nodes) {
    n.queue.wait_and_throw();
  }

  // Display time used to process all time steps
  cout << "  Elapsed time: " << time.Elapsed() << " sec\n";

  // Copy back to host
  for (auto &n : nodes) {
    event e = n.queue.memcpy(arr_host[0] + n.host_offset - 1, n.in->data,
                             sizeof(float) * (n.num_p + 2));
    e.wait();
    free(n.in->data, n.queue);
    free(n.out->data, n.queue);
  }

  CompareResults("multi-device", arr_host[0], arr_CPU, num_p, C);

  delete[] arr_host[0];
  delete[] arr_host[1];
}

//
// Compute heat serially on the host
//
float *ComputeHeatHostSerial(float *arr, float *arr_next, float C, size_t num_p,
                             size_t num_iter) {
  size_t i, k;

  // Set initial condition
  Initialize(arr, arr_next, num_p + 2);

  // Iterate over timesteps
  for (i = 0; i < num_iter; i++) {
    for (k = 1; k <= num_p; k++)
      arr_next[k] = C * (arr[k + 1] - 2 * arr[k] + arr[k - 1]) + arr[k];

    arr_next[num_p + 1] = arr[num_p];

    // Swap the buffers for the next step
    swap(arr, arr_next);
  }

  return arr;
}

int main(int argc, char *argv[]) {
  size_t n_point; // The number of points in 1D space
  size_t
      n_iteration; // The number of iterations to simulate the heat propagation

  // Read input parameters
  try {
    int np = stoi(argv[1]);
    int ni = stoi(argv[2]);
    if (np < 0 || ni < 0) {
      Usage(argv[0]);
      return -1;
    }
    n_point = np;
    n_iteration = ni;
  } catch (...) {
    Usage(argv[0]);
    return (-1);
  }

  cout << "Number of points: " << n_point << "\n";
  cout << "Number of iterations: " << n_iteration << "\n";

  // Temperatures of the current and next iteration
  float *heat_CPU = new float[n_point + 2];
  float *heat_CPU_next = new float[n_point + 2];

  // Constant used in the simulation
  float C = (k * dt) / (dx * dx);

  // Compute heat serially on CPU for comparision
  float *final_CPU = final_CPU =
      ComputeHeatHostSerial(heat_CPU, heat_CPU_next, C, n_point, n_iteration);

  try {
    ComputeHeatBuffer(C, n_point, n_iteration, final_CPU);
    ComputeHeatUSM(C, n_point, n_iteration, final_CPU);
    ComputeHeatMultiDevice(C, n_point, n_iteration, final_CPU);
  } catch (sycl::exception e) {
    cout << "SYCL exception caught: " << e.what() << "\n";
    failures++;
  }

  delete[] heat_CPU;
  delete[] heat_CPU_next;

  return failures;
}

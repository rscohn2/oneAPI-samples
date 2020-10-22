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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <algorithm>
// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/<version>/include/dpc_common.hpp
#include "dpc_common.hpp"

using namespace sycl;
using namespace std;

constexpr float dt = 0.002f;
constexpr float dx = 0.01f;
constexpr float k = 0.025f;
constexpr float initial_temperature = 100.0f;  // Initial temperature.

int failures = 0;

//
// Display input parameters used for this sample.
//
void Usage(string programName) {
  cout << " Incorrect parameters \n";
  cout << " Usage: ";
  cout << programName << " <n> <i>\n\n";
  cout << " n : Number of points to simulate \n";
  cout << " i : Number of timesteps \n";
}

//
// Initialize the temperature arrays
//
void Initialize(float* arr, float* arr_next, size_t num) {
  for (size_t i = 1; i < num; i++)
    arr[i] = arr_next[i] = 0.0f;
  arr[0] = arr_next[0] = initial_temperature;
}

//
// Compare host and device results
//
void CompareResults(string prefix, float* device_results, float* host_results,
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
void ComputeHeatBuffer(float C, size_t num_p, size_t num_iter,
		       float* arr_CPU) {
  // Define device selector as 'default'
  default_selector device_selector;

  // Create a device queue using DPC++ class queue
  queue q(device_selector, dpc_common::exception_handler);
  cout << "Using buffers\n";
  cout << "  Kernel runs on " << q.get_device().get_info<info::device::name>() << "\n";

  // Temperatures of the current and next iteration
  float* arr_host = new float[num_p + 2];
  float* arr_host_next = new float[num_p + 2];

  Initialize(arr_host, arr_host_next, num_p + 2);

  // Start timer
  dpc_common::TimeInterval t_par;

  auto* arr_buf = new buffer<float>(arr_host, range(num_p + 2));
  auto* arr_buf_next = new buffer<float>(arr_host_next, range(num_p + 2));

  // Iterate over timesteps
  for (size_t i = 0; i < num_iter; i++) {
    auto handler =
      [&](auto& h) {
	accessor arr(*arr_buf, h);
	accessor arr_next(*arr_buf_next, h);
	auto step =
	  [=](id<1> idx) {
	    size_t k = idx + 1;

	    if (k == num_p + 1) {
	      arr_next[k] = arr[k - 1];
	    } else {
	      arr_next[k] =
		C * (arr[k + 1] - 2 * arr[k] + arr[k - 1]) + arr[k];
	      }
	  };
      
	h.parallel_for(range{num_p + 1}, step);
      };
    q.submit(handler);

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

  CompareResults("buffer", ((num_iter % 2) == 0) ? arr_host : arr_host_next, arr_CPU, num_p, C);

  delete [] arr_host;
  delete [] arr_host_next;
}

//
// Compute heat on the device using USM
//
void ComputeHeatUSM(float C, size_t num_p, size_t num_iter,
		    float* arr_CPU) {
  // Timesteps depend on each other, so make the queue inorder
  property_list properties{property::queue::in_order()};
  // Define device selector as 'default'
  default_selector device_selector;

  // Create a device queue using DPC++ class queue
  queue q(device_selector, dpc_common::exception_handler, properties);
  cout << "Using USM\n";
  cout << "  Kernel runs on " << q.get_device().get_info<info::device::name>() << "\n";

  // Temperatures of the current and next iteration
  float* arr = malloc_shared<float>(num_p + 2, q);
  float* arr_next = malloc_shared<float>(num_p + 2, q);

  Initialize(arr, arr_next, num_p + 2);

  // Start timer
  dpc_common::TimeInterval time;

  // for each timestep
  for (size_t i = 0; i < num_iter; i++) {
    auto step =
      [=](id<1> idx) {
	size_t k = idx + 1;
	if (k == num_p + 1)
	  arr_next[k] = arr[k - 1];
	else
	  arr_next[k] =
	    C * (arr[k + 1] - 2 * arr[k] + arr[k - 1]) + arr[k];
      };

    q.parallel_for(range{num_p+1}, step);

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
// Contains SYCL device + extra info
//
class Device {
public:
  queue queue;
  device sycl_device;
  float* arr;
  float* arr_next;
  size_t num_points;
  size_t host_offset;
  event device_event;
  event host_event;
};

vector<Device>* GetDevices(size_t num_points) {
  // Select a platform with a GPU
  auto p = sycl::platform(sycl::gpu_selector());
  cout << "  Platform: " << p.get_info<info::platform::name>() << "\n";
  //vector<device> sycl_devices = p.get_devices(info::device_type::gpu);
  vector<device> sycl_devices = p.get_devices();
  //#define FAKE_GPUS 7
#if defined(FAKE_GPUS) && FAKE_GPUS > 0
  for (int i = 0; i < FAKE_GPUS; i++)
    sycl_devices.push_back(sycl_devices[0]);
#endif
  int num_devices = sycl_devices.size();
  cout << "  Number of GPUs: " << num_devices << "\n";
  if (num_devices < 1) {
    cout << "  No devices available.\n";
    return 0;
  }

  // Create queues for each device
  auto* devices = new vector<Device>(num_devices);
  property_list q_prop{property::queue::in_order()};
  int slice_size = num_points / num_devices;
  int slice_bytes = sizeof(float) * slice_size;
  for (int i = 0; i < num_devices; i++) {
    auto& d = (*devices)[i];
    d.sycl_device = sycl_devices[i];
    d.queue = queue(d.sycl_device, dpc_common::exception_handler, q_prop);
  }

  return devices;
}

//
// Compute heat on the device using multiple devices
//
// Slice the 1d temperature array into contiguous chunks and
// distribute among the devices. After each device computes the new
// temperature for the points in its slide, we must exchange the
// boundaries between devices for the next step.
//
void ComputeHeatMultiDevice(float C, size_t num_points, size_t num_iter,
			    float* arr_CPU) {
  cout << "Using multi-device\n";

  auto& devices = *GetDevices(num_points);
  int num_devices = devices.size();

  // Create the initial temperature on host
  float* arr_host = new float[num_points + 2];
  float* arr_host_next = new float[num_points + 2];
  Initialize(arr_host, arr_host_next, num_points + 2);

  // Allocate memory on the devices and initialize from host.
  size_t device_points = num_points/num_devices;
  size_t remainder_points = num_points % num_devices;
  // offset into host array, +1 because host array contains extra
  // point on the left
  size_t host_offset = 1;
  for (int i = 0; i < num_devices; i++) {
    auto& d = devices[i];
    // Divide points among devices equally, distribute remainder among
    // first devices
    d.num_points = device_points + (i < remainder_points);
    d.arr = malloc_shared<float>(d.num_points + 2, d.queue);
    d.arr_next = malloc_shared<float>(d.num_points + 2, d.queue);

    // Initialize device memory from host array. Each device has extra
    // point on left and right that overlaps with neighbor
    d.host_offset = host_offset;
    // size is + 2 and offset is -1 because of the overlap
    size_t s = sizeof(*d.arr) * (d.num_points + 2);
    memcpy(d.arr, arr_host + d.host_offset - 1, s);
    memcpy(d.arr_next, arr_host_next + d.host_offset - 1, s);
    host_offset += d.num_points;
  }

  // Start timer
  dpc_common::TimeInterval time;

  // for each timestep
  for (size_t i = 0; i < num_iter; i++) {

    // queue the device steps that compute the temperature
    for (int i = 0; i < num_devices; i++) {
      auto& d = devices[i];
      float* d_arr = d.arr;
      float* d_arr_next = d.arr_next;
      auto handler =
	[&](sycl::handler &h) {
	  auto step =
	    [=](id<1> idx) {
	      size_t k = idx + 1;
	      d_arr_next[k] =
		C * (d_arr[k + 1] - 2 * d_arr[k] + d_arr[k - 1]) + d_arr[k];
	    };
		   
	  // Wait for the host step on left and this one to finish
	  vector<event> events{d.host_event};
	  if (i != 0)
	    events.push_back(devices[i-1].host_event);
	  event::wait_and_throw(events);
	  h.parallel_for(range{d.num_points}, step);
	};
      d.device_event = d.queue.submit(handler);
    }

    // queue the host steps that swaps the boundaries between devices
    for (int i = 0; i < num_devices; i++) {
      auto& d = devices[i];
      auto handler = 
	[&](auto& h) {
	  auto step = 
	    [=]() {
	      // swap boundaries with the neighbor on the right
	      if (i == num_devices-1) {
		d.arr_next[d.num_points+1] = d.arr[d.num_points];
	      } else {
		auto& d_right = devices[i+1];
		// The device computes 1..num_points
		// 0 & num_points+1 are copies of the neighbor
		d.arr_next[d.num_points+1] = d_right.arr_next[1];
		d_right.arr_next[0] = d.arr_next[d.num_points];
	      }
	    };
      
	  // Wait for this device and neighbor on right to finish
	  vector<event> events{d.device_event};
	  if (i != num_devices-1)
	    events.push_back(devices[i+1].device_event);
	  event::wait_and_throw(events);
	  h.codeplay_host_task(step);
	};
	d.host_event = d.queue.submit(handler);
    }

    // Swap arrays for next step
    for (auto& d : devices)
      swap(d.arr, d.arr_next);
  }

  // Wait for all the timesteps to complete
  for (auto& d : devices)
    d.host_event.wait_and_throw();
    
  // Display time used to process all time steps
  cout << "  Elapsed time: " << time.Elapsed() << " sec\n";

  // Copy from devices to host
  for (auto& d : devices)
    memcpy(arr_host + d.host_offset - 1, d.arr, sizeof(float) * (d.num_points + 2));
  
  CompareResults("multi-device", arr_host, arr_CPU, num_points, C);

  free(arr_host);
  free(arr_host_next);
}

//
// Compute heat serially on the host
//
float* ComputeHeatHostSerial(float* arr, float* arr_next,
			     float C, size_t num_p,
			     size_t num_iter) {
  size_t i, k;

  // Set initial condition
  Initialize(arr, arr_next, num_p + 2);

  // Iterate over timesteps
  for (i = 0; i < num_iter; i++) {
    for (k = 1; k <= num_p; k++)
      arr_next[k] =
	C * (arr[k + 1] - 2 * arr[k] + arr[k - 1]) + arr[k];

    arr_next[num_p + 1] = arr[num_p];

    // Swap the buffers for the next step
    swap(arr, arr_next);
  }

  return arr;
}


int main(int argc, char* argv[]) {
  size_t n_point;      // The number of points in 1D space
  size_t n_iteration;  // The number of iterations to simulate the heat propagation

  // Read input parameters
  try {
    n_point = stoi(argv[1]);
    n_iteration = stoi(argv[2]);
  } catch (...) {
    Usage(argv[0]);
    return (-1);
  }

  cout << "Number of points: " << n_point << "\n";
  cout << "Number of iterations: " << n_iteration << "\n";

  // Temperatures of the current and next iteration
  float* heat_CPU = new float[n_point + 2];
  float* heat_CPU_next = new float[n_point + 2];

  // Constant used in the simulation
  float C = (k * dt) / (dx * dx);

  // Compute heat serially on CPU for comparision
  float* final_CPU = 
    final_CPU = ComputeHeatHostSerial(heat_CPU, heat_CPU_next, C, n_point, n_iteration);

  try {
    ComputeHeatBuffer(C, n_point, n_iteration, final_CPU);
    ComputeHeatUSM(C, n_point, n_iteration, final_CPU);
    ComputeHeatMultiDevice(C, n_point, n_iteration, final_CPU);
  } catch (sycl::exception e) {
    cout << "SYCL exception caught: " << e.what() << "\n";
    failures++;
  }

  delete [] heat_CPU;
  delete [] heat_CPU_next;
  
  return failures;
}

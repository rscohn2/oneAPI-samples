#include <CL/sycl.hpp>

using namespace sycl;
using namespace std;

int main() {
  property_list q_prop{property::queue::in_order()};
  int size = 10;

  float *arr_host = new float[size];

  platform p = device(gpu_selector{}).get_platform();
  cout << "Platform: " << p.get_info<info::platform::name>() << std::endl;
  vector<device> devices = p.get_devices();
  size_t num_devices = devices.size();

  vector<queue> queues(num_devices);
  float* data[10];
  //context ctxt(devices);
  
  for (size_t i = 0; i < num_devices; i++) {
    device &d = devices[i];
    cout << "  device: " << d.get_info<info::device::name>() << std::endl;
    queues[i] = queue(d, q_prop);
    data[i] = malloc_host<float>(size, queues[i]);
    event e = queues[i].memcpy(data[i], arr_host,
			     sizeof(float) * size);
  }

  auto step01 = [=]() {
		  data[0][0] = data[1][0];
		};
  auto step10 = [=]() {
		  data[1][0] = data[0][0];
		};
  queues[0].wait();
  queues[1].wait();
  cout << "Step 1" << std::endl;
  queues[0].single_task(step01);
  queues[0].wait();
  cout << "Step 2" << std::endl;
  queues[1].single_task(step10);
  queues[1].wait();
  cout << "Step 3" << std::endl;
}


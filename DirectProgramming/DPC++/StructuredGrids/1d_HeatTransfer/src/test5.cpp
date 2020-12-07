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
  vector<float*> data(num_devices);
  //context ctxt(devices);
  
  for (size_t i = 0; i < num_devices; i++) {
    device &d = devices[i];
    cout << "  device: " << d.get_info<info::device::name>() << std::endl;
    queues[i] = queue(d, q_prop);
    data[i] = malloc_device<float>(size, queues[i]);
    event e = queues[i].memcpy(data[i], arr_host,
			     sizeof(float) * size);
    e.wait();
  }

  cout << "device 0: copy device 1 to device 0" << std::endl;
  queues[1].memcpy(arr_host, data[1], sizeof(float) * size);
  queues[0].memcpy(data[0], arr_host, sizeof(float) * size);

  cout << "device 0: copy device 0 to device 1" << std::endl;
  queues[0].memcpy(arr_host, data[0], sizeof(float) * size);
  queues[0].memcpy(data[1], arr_host, sizeof(float) * size);
}


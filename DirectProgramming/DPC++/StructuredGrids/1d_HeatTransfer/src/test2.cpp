#include <CL/sycl.hpp>

using namespace sycl;
using namespace std;

int main() {
  int size = 10;

  float *arr_host = new float[size];

  platform p = device(gpu_selector{}).get_platform();
  cout << "Platform: " << p.get_info<info::platform::name>() << std::endl;
  vector<device> devices = p.get_devices();
  size_t num_devices = devices.size();

  vector<queue> queues(num_devices);
  context ctxt(devices);
  
  for (size_t i = 0; i < num_devices; i++) {
    device &d = devices[i];
    cout << "  device: " << d.get_info<info::device::name>() << std::endl;
    queues[i] = queue(ctxt, d);
    float *data = malloc_device<float>(size, queues[i]);
    event e = queues[i].memcpy(data, arr_host,
			     sizeof(float) * size);
    e.wait();
  }
  

}


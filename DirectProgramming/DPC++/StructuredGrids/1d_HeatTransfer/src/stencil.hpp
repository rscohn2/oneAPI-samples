#include <chrono>

extern float *input;
extern float *reference_output;
extern int data_size;
extern int num_steps;
extern int num_points;

void parse_args(int argc, char *argv[]);
void initialize();
void deinitialize();
std::chrono::steady_clock::time_point time_begin(std::string name);
void time_end(std::chrono::steady_clock::time_point start);
void check(std::string, float*);

static void compute_point(float *out, float *in) {
  out[0] = (in[-1] + in[0] + in[1])/3;
}

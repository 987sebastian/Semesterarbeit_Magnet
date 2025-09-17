#include "model.hpp"
#include <iostream>
int main(){
  std::vector<double> sensors = {0.03,0.01,0.02, 0.04,0.02,0.03};
  std::array<double,3> c = {0.01,0.01,0.012};
  std::array<double,3> u = {0.0,0.0,1.0};
  auto res = model::compute_B_and_J(sensors, c, u, 1.0);
  std::cout << "B: " << res.B[0] << "," << res.B[1] << "," << res.B[2] << std::endl;
}

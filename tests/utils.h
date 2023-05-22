#include <random>

double GetRandomDouble(double lower_bound, double upper_bound) {
    double value = (double)rand() / RAND_MAX;
    return lower_bound + value * (upper_bound - lower_bound);
}
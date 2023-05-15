#include <string>
#include <vector>
#include <iostream>
#include <fstream>

namespace NeuralNetworkApp {

class MNISTReader {
public:
    MNISTReader(const std::string& images_filepath, const std::string& labels_filepath)
        : images_filepath_(images_filepath), labels_filepath_(labels_filepath) {
    }

    std::vector<std::vector<double>> GetImagesData() const;
    std::vector<double> GetLabelsData() const;

private:
    std::string images_filepath_;
    std::string labels_filepath_;
    static constexpr uint32_t labels_magic_number_ = 2049;
    static constexpr uint32_t images_magic_number = 2051;

    uint32_t GetUIntFromHighEndian(unsigned char* buff) const;
};

}  // namespace NeuralNetworkApp
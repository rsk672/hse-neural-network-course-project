#include "mnist_reader.h"

namespace NeuralNetworkApp {

uint32_t MNISTReader::GetUIntFromHighEndian(unsigned char* buff) const {
    uint32_t result = static_cast<uint32_t>(buff[0]) << 24;
    result |= static_cast<uint32_t>(buff[1]) << 16;
    result |= static_cast<uint32_t>(buff[2]) << 8;
    result |= static_cast<int32_t>(buff[3]);

    return result;
}

std::vector<std::vector<double>> MNISTReader::GetImagesData() const {
    std::ifstream f;

    f.open(images_filepath_, std::ios::in | std::ios::binary);

    if (!f) {
        throw std::ios_base::failure("Unable to open the file");
    }

    unsigned char magic_number_buff[4];
    f.read(reinterpret_cast<char*>(magic_number_buff), 4);
    uint32_t magic_number = GetUIntFromHighEndian(magic_number_buff);

    if (magic_number != images_magic_number) {
        throw std::ios_base::failure("Incorrect file");
    }

    unsigned char image_count_buff[4];
    f.read(reinterpret_cast<char*>(image_count_buff), 4);
    uint32_t image_count = GetUIntFromHighEndian(image_count_buff);

    unsigned char row_count_buff[4];
    f.read(reinterpret_cast<char*>(row_count_buff), 4);
    uint32_t row_count = GetUIntFromHighEndian(row_count_buff);

    unsigned char col_count_buff[4];
    f.read(reinterpret_cast<char*>(col_count_buff), 4);
    uint32_t col_count = GetUIntFromHighEndian(col_count_buff);

    std::vector<std::vector<double>> images(image_count);
    for (uint32_t k = 0; k < image_count; ++k) {
        for (uint32_t i = 0; i < row_count; ++i) {
            for (uint32_t j = 0; j < col_count; ++j) {
                unsigned char pixel_buff = 0;
                f.read(reinterpret_cast<char*>(&pixel_buff), 1);
                images[k].push_back(pixel_buff);
            }
        }
    }

    f.close();

    return images;
}

std::vector<double> MNISTReader::GetLabelsData() const {
    std::ifstream f;

    f.open(labels_filepath_, std::ios::in | std::ios::binary);

    if (!f) {
        throw std::ios_base::failure("Unable to open the file");
    }

    unsigned char magic_number_buff[4];
    f.read(reinterpret_cast<char*>(magic_number_buff), 4);
    uint32_t magic_number = GetUIntFromHighEndian(magic_number_buff);

    if (magic_number != labels_magic_number_) {
        throw std::ios_base::failure("Incorrect file");
    }

    unsigned char label_count_buff[4];
    f.read(reinterpret_cast<char*>(label_count_buff), 4);
    uint32_t label_count = GetUIntFromHighEndian(label_count_buff);

    std::vector<double> labels;
    for (uint32_t i = 0; i < label_count; ++i) {
        unsigned char label_buff = 0;
        f.read(reinterpret_cast<char*>(&label_buff), 1);
        labels.push_back(label_buff);
    }

    f.close();

    return labels;
}
}  // namespace NeuralNetworkApp
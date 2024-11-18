#include <filesystem>
#include <fstream>
#include <iostream>

namespace Eacpp {

enum class CsvLineEnd { None, Comma, Newline };

inline std::ifstream OpenInputFile(const std::filesystem::path& filePath, std::ios_base::openmode mode = std::ios_base::in) {
    std::ifstream fileStream(filePath, mode);
    if (!fileStream) {
        std::cerr << "Error: Could not open " << filePath << std::endl;
        std::exit(1);
    }
    return fileStream;
}

inline std::ofstream OpenOutputFile(const std::filesystem::path& filePath, std::ios_base::openmode mode = std::ios_base::out) {
    std::ofstream fileStream(filePath, mode);
    if (!fileStream) {
        std::cerr << "Error: Could not create or open " << filePath << std::endl;
        std::exit(1);
    }
    return fileStream;
}

template <std::ranges::range T>
void WriteCsvLine(std::ofstream& file, const T& data, CsvLineEnd lineEnding = CsvLineEnd::Newline) {
    bool first = true;
    for (const auto& item : data) {
        if (!first) {
            file << ",";
        }
        file << item;
        first = false;
    }

    switch (lineEnding) {
        case CsvLineEnd::None:
            break;
        case CsvLineEnd::Comma:
            file << ",";
            break;
        case CsvLineEnd::Newline:
            file << std::endl;
            break;
    }
}

template <std::ranges::range T>
void WriteCsv(std::ofstream& file, const std::vector<T>& data, std::vector<std::string> header = {}) {
    if (!header.empty()) {
        WriteCsvLine(file, header);
    }

    for (const auto& row : data) {
        WriteCsvLine(file, row);
    }
}

template <typename T, std::ranges::range U>
void WriteCsv(std::ofstream& file, const std::vector<std::pair<T, U>>& data, std::vector<std::string> header = {}) {
    if (!header.empty()) {
        WriteCsvLine(file, header);
    }

    for (const auto& [key, row] : data) {
        file << key << ",";
        WriteCsvLine(file, row);
    }
}

template <std::ranges::range T, std::ranges::range U>
void WriteCsv(std::ofstream& file, const std::vector<std::pair<T, U>>& data, std::vector<std::string> header = {}) {
    if (!header.empty()) {
        WriteCsvLine(file, header);
    }

    for (const auto& [keys, row] : data) {
        WriteCsvLine(file, keys, CsvLineEnd::Comma);
        WriteCsvLine(file, row);
    }
}

template <std::ranges::range T>
void WriteCsv(std::ofstream& file, const T& data, int step, std::vector<std::string> header = {}) {
    if (!header.empty()) {
        WriteCsvLine(file, header);
    }

    int count = 0;
    for (const auto& elem : data) {
        file << elem;
        count++;
        if (count % step == 0) {
            file << std::endl;
        } else {
            file << ",";
        }
    }
    if (count % step != 0) {
        file << std::endl;
    }
}

}  // namespace Eacpp
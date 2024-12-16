#include <filesystem>
#include <fstream>
#include <iostream>
#include <ranges>
#include <sstream>
#include <string>
#include <vector>

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

template <std::ranges::range T, std::ranges::range U>
void WriteCsv(std::ofstream& file, const std::vector<T>& data, const U& header = {}) {
    if (!header.empty()) {
        WriteCsvLine(file, header);
    }

    for (const auto& row : data) {
        WriteCsvLine(file, row);
    }
}

template <typename T, typename U, std::ranges::range V>
void WriteCsv(std::ofstream& file, const std::vector<std::pair<T, U>>& data, const V& header = {}) {
    if (!header.empty()) {
        WriteCsvLine(file, header);
    }

    for (const auto& [key, row] : data) {
        file << key << ",";
        file << row << std::endl;
    }
}

template <typename T, std::ranges::range U, std::ranges::range V>
void WriteCsv(std::ofstream& file, const std::vector<std::pair<T, U>>& data, const V& header = {}) {
    if (!header.empty()) {
        WriteCsvLine(file, header);
    }

    for (const auto& [key, row] : data) {
        file << key << ",";
        WriteCsvLine(file, row);
    }
}

template <std::ranges::range T, std::ranges::range U, std::ranges::range V>
void WriteCsv(std::ofstream& file, const std::vector<std::pair<T, U>>& data, const V& header = {}) {
    if (!header.empty()) {
        WriteCsvLine(file, header);
    }

    for (const auto& [keys, row] : data) {
        WriteCsvLine(file, keys, CsvLineEnd::Comma);
        WriteCsvLine(file, row);
    }
}

template <std::ranges::range T, std::ranges::range U>
void WriteCsv(std::ofstream& file, const T& data, int step, const U& header = {}) {
    if (!header.empty()) {
        WriteCsvLine(file, header);
    }

    int count = 0;
    std::vector<std::vector<decltype(*std::begin(data))>> data2d;
    std::vector<decltype(*std::begin(data))> currentRow;

    for (const auto& elem : data) {
        currentRow.emplace_back(elem);
        count++;
        if (count % step == 0) {
            data2d.emplace_back(currentRow);
            currentRow.clear();
        }
    }
    if (!currentRow.empty()) {
        data2d.emplace_back(currentRow);
    }

    for (const auto& row : data2d) {
        WriteCsvLine(file, row);
    }
}

template <typename T>
std::vector<std::vector<T>> ReadCsv(std::ifstream& file, bool hasHeader = false) {
    std::vector<std::vector<T>> data;
    std::string line;
    if (hasHeader) {
        std::getline(file, line);
    }

    while (std::getline(file, line)) {
        std::vector<T> row;
        std::istringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            row.push_back(static_cast<T>(std::stod(cell)));
        }

        data.push_back(row);
    }

    return data;
}

}  // namespace Eacpp
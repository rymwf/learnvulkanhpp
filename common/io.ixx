export module common.io;

import std;

export std::vector<std::uint32_t> readSpirvFile(const std::filesystem::path &path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error(std::format("Failed to open SPIR-V: {}", path.string()));
    }

    const auto end = file.tellg();
    if (end <= std::streampos{}) {
        throw std::runtime_error(std::format("Invalid SPIR-V size for {}", path.string()));
    }

    const auto size = static_cast<std::size_t>(end);
    if (size % sizeof(std::uint32_t) != 0) {
        throw std::runtime_error(std::format("Invalid SPIR-V size for {}", path.string()));
    }

    file.seekg(0);
    std::vector<std::uint32_t> buffer(size / sizeof(std::uint32_t));
    file.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(size));
    if (!file) {
        throw std::runtime_error(std::format("Failed to read SPIR-V: {}", path.string()));
    }
    return buffer;
}

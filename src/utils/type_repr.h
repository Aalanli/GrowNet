#include <string>

namespace type_repr {

template <typename T>
std::string to_std_type_str() {
    throw std::runtime_error("no default conversion from type to str");
}

template <>
std::string to_std_type_str<double>() {
    return std::string("d");
}
template <>
std::string to_std_type_str<float>() {
    return std::string("f");
}
template <>
std::string to_std_type_str<int>() {
    return std::string("i");
}
template <>
std::string to_std_type_str<long long>() {
    return std::string("l");
}
template <>
std::string to_std_type_str<unsigned int>() {
    return std::string("ui");
}

}
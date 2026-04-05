export module common.debug;

import std;

export constexpr bool debug_mode =
#ifndef NDEBUG
    true;
#else
    false;
#endif

export template <class... Args>
inline void debug_throw_if(bool condition, std::format_string<Args...> fmt, Args&&... args) {
    if constexpr (debug_mode) {
        if (condition) {
            throw std::runtime_error(std::format(fmt, std::forward<Args>(args)...));
        }
    }
}
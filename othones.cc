
#include <iostream>
#include <vector>
#include <source_location>
#include <cassert>

#include <wayland-client-core.h>
#include <wayland-client-protocol.h>

#include "ext-tuple-like.hpp"
#include "aux-wayland.hpp"


inline auto lamed() {
    return [](auto...) { };
}
inline auto lamed(auto&& closure) {
    static auto cache = closure;
    return [](auto... args) { return cache(args...); };
}

int main() {
    using namespace aux;
    try {
        auto display = wrapper(wl_display_connect(nullptr));

        wrapper<wl_compositor> compositor;
        wrapper<wl_shm> shm;
        wrapper<zxdg_shell_v6> shell;
        bool format_checked = false;
        std::vector<wrapper<wl_output>> outputs;

        auto registry = wrapper(
            wl_display_get_registry(display),
            [&](auto registry, uint32_t name, std::string_view interface, uint32_t version) {
                if (interface == interface_name<wl_compositor>) {
                    compositor = wrapper(wl_registry_bind<wl_compositor>(registry, name, version));
                }
                else if (interface == interface_name<wl_shm>) {
                    shm = wrapper(
                        wl_registry_bind<wl_shm>(registry, name, version),
                        [&](auto, uint32_t format) {
                            if (format == WL_SHM_FORMAT_XRGB8888) {
                                format_checked = true;
                            }
                        });
                }
                else if (interface == interface_name<zxdg_shell_v6>) {
                    shell = wrapper(wl_registry_bind<zxdg_shell_v6>(registry, name, version));
                }
                else if (interface == interface_name<wl_output>) {
                    // static wl_output_listener listener = {
                    //     .geometry = [](auto... args) noexcept {
                    //         std::cout << std::tuple{args...} << std::endl;
                    //     },
                    //     .mode = [](auto... args) noexcept {
                    //         std::cout << std::tuple{args...} << std::endl;
                    //     },
                    //     .done = [](auto... args) noexcept {
                    //         std::cout << std::tuple{args...} << std::endl;
                    //     },
                    //     .scale = [](auto... args) noexcept {
                    //         std::cout << std::tuple{args...} << std::endl;
                    //     },
                    //     .name = [](auto... args) noexcept {
                    //         std::cout << std::tuple{args...} << std::endl;
                    //     },
                    //     .description = [](auto... args) noexcept {
                    //         std::cout << std::tuple{args...} << std::endl;
                    //     },
                    // };
                    // outputs.emplace_back(wl_registry_bind<wl_output>(registry, name, version));
                    // wl_output_add_listener(outputs.back(), &listener, nullptr);

                    outputs.emplace_back(wl_registry_bind<wl_output>(registry, name, version),
                                         [](auto... args) noexcept {
                                             std::cout << std::tuple{args...} << std::endl;
                                         },
                                         [](auto... args) noexcept {
                                             std::cout << std::tuple{args...} << std::endl;
                                         },
                                         [](auto... args) noexcept {
                                             std::cout << std::tuple{args...} << std::endl;
                                         },
                                         [](auto... args) noexcept {
                                             std::cout << std::tuple{args...} << std::endl;
                                         },
                                         [](auto... args) noexcept {
                                             std::cout << std::tuple{args...} << std::endl;
                                         },
                                         [](auto... args) noexcept {
                                             std::cout << std::tuple{args...} << std::endl;
                                         });
                }
            },
            [](auto...) { });
        wl_display_roundtrip(display);

        assert(compositor);
        assert(shm);
        assert(shell);
        assert(outputs.empty() == false);
        // for (auto& output : outputs) {
        //     output.add(wl_output_listener {
        //             .geometry = lamed([](auto... args) noexcept {
        //                 std::cout << std::tuple{args...} << std::endl;
        //             }),
        //             .mode = lamed([](auto... args) noexcept {
        //                 std::cout << std::tuple{args...} << std::endl;
        //             }),
        //             .done = lamed([](auto... args) noexcept {
        //                 std::cout << std::tuple{args...} << std::endl;
        //             }),
        //             .scale = lamed([](auto... args) noexcept {
        //                 std::cout << std::tuple{args...} << std::endl;
        //             }),
        //             .name = lamed([](auto... args) noexcept {
        //                 std::cout << std::tuple{args...} << std::endl;
        //             }),
        //             .description = lamed([](auto... args) noexcept {
        //                 std::cout << std::tuple{args...} << std::endl;
        //             }),
        //         });
        // }

        wl_display_roundtrip(display);
        assert(format_checked);
    }
    catch (std::exception& ex) {
        std::cout << "An exceeption occured: " << ex.what() << std::endl;
    }
    return 0;
}

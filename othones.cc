/////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <vector>
#include <source_location>
#include <filesystem>
#include <cstring>
#include <cassert>

#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

#include <sycl/sycl.hpp>

#include <wayland-client-core.h>
#include <wayland-client-protocol.h>


/////////////////////////////////////////////////////////////////////////////
#include <concepts>
#include <type_traits>
#include <tuple>
#include <iosfwd>


namespace std::inline ext
{
    template <class T, size_t I>
    concept has_tuple_element = requires(T t) {
        typename std::tuple_element_t<I, std::remove_const_t<T>>;
        { get<I>(t) } -> std::convertible_to<std::tuple_element_t<I, T> const&>;
    };
    template <class T>
    concept tuple_like = !std::is_reference_v<T> && requires(T) {
        std::tuple_size<T>::value;
        requires std::derived_from<std::tuple_size<T>, std::integral_constant<size_t, std::tuple_size_v<T>>>;
    } && []<size_t... I>(std::index_sequence<I...>) noexcept {
        return (has_tuple_element<T, I>&& ...);
    }(std::make_index_sequence<std::tuple_size_v<T>>());

    template <class Ch, tuple_like T>
    auto& operator<<(std::basic_ostream<Ch>& output, T const& t) noexcept {
        output.put('(');
        [&]<size_t ...I>(std::index_sequence<I...>) noexcept {
            (void) (int[]) {(output << (I==0 ? "" : " ") << get<I>(t), 0)...};
        }(std::make_index_sequence<std::tuple_size_v<T>>());
        return output.put(')');
    }
} // ::std::ext

/////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <stdexcept>
#include <utility>
#include <memory>
#include <string_view>

#include <wayland-client.h>
#include <wayland-client.h>

#include "xdg-shell-v6-client.h"
#include "zwp-tablet-v2-client.h"


namespace aux::inline wayland
{
    struct empty_type { };
    template <class> constexpr wl_interface const *const interface_ptr = nullptr;

    template <class T> concept client_like = (interface_ptr<T> != nullptr);

    template <client_like T> constexpr void (*deleter)(T*) = [](auto) { static_assert("unknown deleter"); };

    template <client_like T> struct listener_type { };
#define INTERN_CLIENT_LIKE_CONCEPT(CLIENT, DELETER, LISTENER)          \
    template <> constexpr wl_interface const *const interface_ptr<CLIENT> = &CLIENT##_interface; \
    template <> constexpr void (*deleter<CLIENT>)(CLIENT*) = DELETER;   \
    template <> struct listener_type<CLIENT> : LISTENER { };
    INTERN_CLIENT_LIKE_CONCEPT(wl_display,            wl_display_disconnect,         empty_type)
    INTERN_CLIENT_LIKE_CONCEPT(wl_registry,           wl_registry_destroy,           wl_registry_listener)
    INTERN_CLIENT_LIKE_CONCEPT(wl_compositor,         wl_compositor_destroy,         empty_type)
    INTERN_CLIENT_LIKE_CONCEPT(wl_output,             wl_output_destroy,             wl_output_listener)
    INTERN_CLIENT_LIKE_CONCEPT(wl_shm,                wl_shm_destroy,                wl_shm_listener)
    INTERN_CLIENT_LIKE_CONCEPT(wl_seat,               wl_seat_destroy,               wl_seat_listener)
    INTERN_CLIENT_LIKE_CONCEPT(wl_surface,            wl_surface_destroy,            wl_surface_listener)
    INTERN_CLIENT_LIKE_CONCEPT(wl_shm_pool,           wl_shm_pool_destroy,           empty_type)
    INTERN_CLIENT_LIKE_CONCEPT(wl_buffer,             wl_buffer_destroy,             wl_buffer_listener)
    INTERN_CLIENT_LIKE_CONCEPT(wl_keyboard,           wl_keyboard_destroy,           wl_keyboard_listener)
    INTERN_CLIENT_LIKE_CONCEPT(wl_pointer,            wl_pointer_destroy,            wl_pointer_listener)
    INTERN_CLIENT_LIKE_CONCEPT(wl_touch,              wl_touch_destroy,              wl_touch_listener)
    INTERN_CLIENT_LIKE_CONCEPT(zxdg_shell_v6,         zxdg_shell_v6_destroy,         zxdg_shell_v6_listener)
    INTERN_CLIENT_LIKE_CONCEPT(zxdg_surface_v6,       zxdg_surface_v6_destroy,       zxdg_surface_v6_listener)
    INTERN_CLIENT_LIKE_CONCEPT(zxdg_toplevel_v6,      zxdg_toplevel_v6_destroy,      zxdg_toplevel_v6_listener)
    INTERN_CLIENT_LIKE_CONCEPT(zwp_tablet_manager_v2, zwp_tablet_manager_v2_destroy, empty_type)
#undef INTERN_CLIENT_LIKE_CONCEPT

    template <class T>
    concept client_like_with_listener = client_like<T> && !std::is_base_of_v<empty_type, listener_type<T>>;

    template <client_like T>
    auto make_unique(T* raw = nullptr) noexcept {
        auto del = [](auto p) noexcept {
            std::cerr << "deleting... " << p << ':' << interface_ptr<T>->name << std::endl;
            deleter<T>(p);
        };
        return std::unique_ptr<T, decltype (del)>(raw, del);
    }
    template <client_like T>
    using unique_ptr_type = decltype (make_unique<T>());

    template <class> class wrapper;
    template <class T> wrapper(T*) -> wrapper<T>;

    template <client_like T>
    class wrapper<T> {
    public:
        wrapper(T* raw = nullptr) : ptr{make_unique(raw)}
            {
            }
        operator T*() const { return this->ptr.get(); }

    private:
        unique_ptr_type<T> ptr;
    };

    template <client_like_with_listener T>
    class wrapper<T> {
    private:
        static constexpr auto new_default_listener() {
            static constexpr auto N = sizeof (listener_type<T>) / sizeof (void*);
            return new listener_type<T>{
                []<size_t... I>(std::index_sequence<I...>) noexcept {
                    return listener_type<T> {
                        ([](auto... args) noexcept {
                            (void) I;
                        })...
                    };
                }(std::make_index_sequence<N>())};
        }

    public:
        wrapper(T* raw = nullptr)
            : ptr{make_unique(raw)}
            , listener{new_default_listener()}
            {
                if (ptr != nullptr) {
                    if (0 != wl_proxy_add_listener(reinterpret_cast<wl_proxy*>(operator T*()),
                                                   reinterpret_cast<void(**)(void)>(this->listener.get()),
                                                   nullptr))
                    {
                        throw std::runtime_error("failed to add listener...");
                    }
                }
            }
        operator T*() const { return this->ptr.get(); }
        listener_type<T>* operator->() const { return this->listener.get(); }

    private:
        unique_ptr_type<T> ptr;
        std::unique_ptr<listener_type<T>> listener;
    };

    template <client_like T>
    auto registry_bind(wl_registry* registry, uint32_t name, uint32_t version) noexcept {
        return static_cast<T*>(::wl_registry_bind(registry, name, interface_ptr<T>, version));
    }

    template <class T = uint32_t, wl_shm_format format = WL_SHM_FORMAT_XRGB8888, size_t bypp = 4>
    inline auto shm_allocate_buffer(wl_shm* shm, size_t cx, size_t cy) {
        std::string_view xdg_runtime_dir = std::getenv("XDG_RUNTIME_DIR");
        if (xdg_runtime_dir.empty() || !std::filesystem::exists(xdg_runtime_dir)) {
            throw std::runtime_error("No XDG_RUNTIME_DIR settings...");
        }
        std::string tmp_path(xdg_runtime_dir);
        tmp_path += "/weston-shared-XXXXXX";
        int fd = mkostemp(tmp_path.data(), O_CLOEXEC);
        if (fd >= 0) {
            unlink(tmp_path.c_str());
        }
        else {
            throw std::runtime_error("mkostemp failed...");
        }
        if (ftruncate(fd, bypp*cx*cy) < 0) {
            close(fd);
            throw std::runtime_error("ftruncate failed...");
        }
        void* data = mmap(nullptr, bypp*cx*cy, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (data == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("mmap failed...");
        }
        return std::tuple{
            wrapper(wl_shm_pool_create_buffer(wrapper(wl_shm_create_pool(shm, fd, bypp*cx*cy)),
                                             0, cx, cy, bypp * cx, format)),
            static_cast<T*>(data),
        };
    }

} // ::aux::wayland

inline auto lamed() noexcept {
    return [](auto...) noexcept { };
}

inline auto lamed(auto&& closure) noexcept {
    static auto cache = closure;
    return [](auto... args) {
        return cache(args...);
    };
}

int main() {
    using namespace aux;

    auto display = wrapper{wl_display_connect(nullptr)};
    auto registry = wrapper{wl_display_get_registry(display)};

    wrapper<wl_compositor> compositor;
    wrapper<wl_shm> shm;

    wrapper<wl_seat> seat;
    wrapper<wl_pointer> pointer;
    wrapper<wl_keyboard> keyboard;
    wrapper<wl_touch> touch;
    bool escaped = false;

    wrapper<zxdg_shell_v6> shell;

    size_t cx = std::numeric_limits<size_t>::max();
    size_t cy = std::numeric_limits<size_t>::max();
    std::vector<wrapper<wl_output>> outputs;

    registry->global = lamed([&](auto, auto registry, uint32_t name, std::string_view interface, uint32_t version) {
        if (interface == interface_ptr<wl_compositor>->name) {
            compositor = wrapper{registry_bind<wl_compositor>(registry, name, version)};
        }
        else if (interface == interface_ptr<wl_shm>->name) {
            shm = wrapper{registry_bind<wl_shm>(registry, name, version)};
        }
        else if (interface == interface_ptr<wl_seat>->name) {
            seat = wrapper{registry_bind<wl_seat>(registry, name, version)};
            seat->capabilities = lamed([&](auto, auto seat, auto capabilities) {
                if (capabilities & WL_SEAT_CAPABILITY_KEYBOARD) {
                    keyboard = wrapper{wl_seat_get_keyboard(seat)};
                    keyboard->key = lamed([&](auto, auto, auto, auto, auto k, auto s) {
                        if ((k == 1 || k == 16) && s == 0) {
                            escaped = true;
                        }
                    });
                }
                if (capabilities & WL_SEAT_CAPABILITY_POINTER) {
                    pointer = wrapper{wl_seat_get_pointer(seat)};
                }
                if (capabilities & WL_SEAT_CAPABILITY_TOUCH) {
                    touch = wrapper{wl_seat_get_touch(seat)};
                }
            });
        }
        else if (interface == interface_ptr<zxdg_shell_v6>->name) {
            shell = wrapper{registry_bind<zxdg_shell_v6>(registry, name, version)};
            shell->ping = [](auto, auto shell, auto serial) noexcept {
                zxdg_shell_v6_pong(shell, serial);
            };
        }
        else if (interface == interface_ptr<wl_output>->name) {
            outputs.emplace_back(wrapper{registry_bind<wl_output>(registry, name, version)});
            outputs.back()->mode = lamed([&](auto, auto, auto, int32_t width, int32_t height, auto) noexcept {
                cx = std::min<size_t>(cx, width) / 4;
                cy = std::min<size_t>(cy, height)/ 4;
            });
        }
    });
    wl_display_roundtrip(display);
    wl_display_roundtrip(display);

    auto surface = wrapper{wl_compositor_create_surface(compositor)};
    auto xsurface = wrapper{zxdg_shell_v6_get_xdg_surface(shell, surface)};
    xsurface->configure = [](auto, auto xsurface, auto serial) noexcept {
        zxdg_surface_v6_ack_configure(xsurface, serial);
    };

    auto [buffer, pixels] = shm_allocate_buffer(shm, cx, cy);
    auto toplevel = wrapper{zxdg_surface_v6_get_toplevel(xsurface)};
    toplevel->configure = lamed([&](auto, auto, auto w, auto h, auto) {
        std::cout << std::tuple{w, h} << std::endl;
        cx = w;
        cy = h;
        if (cx && cy) {
            std::cout << std::tuple{cx, cy} << std::endl;
            auto [b, p] = shm_allocate_buffer(shm, cx, cy);
            std::cout << b << std::endl;
            std::cout << p << std::endl;
            buffer = std::move(b);
            pixels = p;
        }
    });
    wl_surface_commit(surface);

    auto que = sycl::queue();
    std::cout << que.get_device().get_info<sycl::info::device::name>() << std::endl;
    std::cout << que.get_device().get_info<sycl::info::device::vendor>() << std::endl;

    while (wl_display_dispatch(display) != -1) {
        if (escaped) break;
        if (cx && cy) {
            auto pv = sycl::buffer<uint32_t, 2>{pixels, {cy, cx}};
            que.submit([&](auto& h) noexcept {
                auto apv = pv.get_access<sycl::access::mode::write>(h);
                h.parallel_for({cy, cx}, [=](auto idx) noexcept {
                    apv[idx] = 0;
                });
            });
        }
        wl_surface_damage(surface, 0, 0, cx, cy);
        wl_surface_attach(surface, buffer, 0, 0);
        wl_surface_commit(surface);
        wl_display_flush(display);
    }

    return 0;
}

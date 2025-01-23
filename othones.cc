/////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <vector>
#include <source_location>
#include <filesystem>
#include <complex>
#include <cstring>
#include <cassert>

#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

#include <sycl/sycl.hpp>

#include <wayland-client-core.h>
#include <wayland-client-protocol.h>


/////////////////////////////////////////////////////////////////////////////

namespace aux::inline algebra
{
    template <class T, size_t N>
    struct versor : versor<T, N-1> {
    public:
        using base_type = versor<T, N-1>;
        using value_type = typename base_type::value_type;
        using iterator = typename base_type::iterator;
        using const_iterator = typename base_type::const_iterator;
        using reference = typename base_type::reference;
        using const_reference = typename base_type::const_reference;

    public:
        value_type last;

    public:
        constexpr versor(auto... args) noexcept
        : versor({static_cast<T>(args)...}, std::make_index_sequence<N-1>())
            {
                static_assert(sizeof... (args) <= N);
            }

    private:
        constexpr static auto at(std::initializer_list<T> const& args, size_t i) noexcept {
            return (i < args.size()) ? *(args.begin() + i) : T();
        }
        template <size_t ...I>
        constexpr versor(std::initializer_list<T>&& args, std::index_sequence<I...>) noexcept
            : base_type{at(args, I)...}, last{at(args, N-1)}
            {
            }

    public:
        template <size_t I>
        constexpr auto get() const noexcept {
            static_assert(I < N);
            return static_cast<versor<T, I+1> const*>(this)->last;
        }
        template <size_t I>
        constexpr auto& get() noexcept {
            static_assert(I < N);
            return static_cast<versor<T, I+1>*>(this)->last;
        }
        template <size_t I>
        constexpr friend auto get(versor const& v) noexcept { return v.get<I>(); }
        template <size_t I>
        constexpr friend auto& get(versor& v) noexcept { return v.get<I>(); }

    public:
        constexpr auto size() const noexcept { return N; }

        constexpr auto begin() const noexcept { return &static_cast<versor<T, 1> const*>(this)->last; }
        constexpr auto begin()       noexcept { return &static_cast<versor<T, 1>      *>(this)->last; }
        constexpr auto end  () const noexcept { return &this->last + 1; }
        constexpr auto end  ()       noexcept { return &this->last + 1; }

        constexpr auto  front() const noexcept { return *(this->begin()); }
        constexpr auto& front()       noexcept { return *(this->begin()); }
        constexpr auto  back () const noexcept { return this->last; }
        constexpr auto& back ()       noexcept { return this->last; }

        constexpr auto  operator[](size_t i) const noexcept { return *(this->begin() + i); }
        constexpr auto& operator[](size_t i)       noexcept { return *(this->begin() + i); }

    public:
        template <class Func, class ...Rest>
        constexpr auto& apply(Func&& func, Rest&& ...rest) noexcept {
            this->last = func(this->last, rest.last...);
            if constexpr ((std::is_rvalue_reference_v<Rest> && ...)) {
                base_type::apply(std::forward<Func>(func), std::move(static_cast<base_type&&>(rest))...);
            }
            else {
                base_type::apply(std::forward<Func>(func), static_cast<base_type const&>(rest)...);
            }
            return *this;
        }

    public:
        constexpr auto operator<=>(versor const& rhs) const noexcept = default;

        constexpr auto operator+() const noexcept { return *this; }

        constexpr auto& negate() noexcept { return apply(std::negate<T>()); }
        constexpr auto operator-() const noexcept { return (+(*this)).negate(); }
        constexpr auto& lognot() noexcept { return apply(std::bit_not<T>()); }
        constexpr auto operator~() const noexcept { return (+(*this)).lognot(); }

        constexpr auto& operator+=(auto&& rhs) noexcept { return apply(std::plus<T>(), rhs); }
        constexpr auto& operator-=(auto&& rhs) noexcept { return apply(std::minus<T>(), rhs); }

        constexpr auto operator+(auto&& rhs) const noexcept { return (+(*this)) += rhs; }
        constexpr auto operator-(auto&& rhs) const noexcept { return (+(*this)) -= rhs; }

        constexpr auto& operator*=(value_type s) noexcept {
            return apply([s](value_type x) noexcept {
                return x * s;
            });
        }
        constexpr auto operator*(value_type s) const noexcept { return (+(*this)) *= s; }
        constexpr friend auto operator*(value_type s, versor v) noexcept { return v * s; }

        constexpr auto& operator/=(value_type d) noexcept {
            return apply([d](value_type x) noexcept {
                return x / d;
            });
        }
        constexpr auto operator/(value_type d) const noexcept { return (+(*this)) /= d; }

        constexpr friend auto inner(versor const& a, versor const& b) noexcept {
            return a.last * b.last + inner(static_cast<base_type const&>(a), static_cast<base_type const&>(b));
        }
    };

    template <class T>
    struct versor<T, 0> {
    public:
        using value_type = T;
        using iterator = value_type*;
        using const_iterator = value_type const*;
        using reference = value_type&;
        using const_reference = value_type const&;
        using size_type = size_t;

        constexpr auto operator<=>(versor const& rhs) const noexcept = default;

    protected:
        constexpr auto& apply(auto&&...) noexcept { return *this; }

        constexpr friend T inner(versor const&, versor const&) noexcept { return T(); }
    };

    template <class T>
    constexpr auto cross(versor<T, 3> lhs, versor<T, 3> rhs) noexcept {
        return versor<T, 3>{
            lhs[1] * rhs[2] - lhs[2] * rhs[1],
            lhs[2] * rhs[0] - lhs[0] * rhs[2],
            lhs[0] * rhs[1] - lhs[1] * rhs[0],
        };
    }
    template <class T>
    constexpr auto det(versor<T, 3> a, versor<T, 3> b, versor<T, 3> c) noexcept {
        return inner(cross(a, b), c);
    }

    using vec2s = versor<short,  2>;
    using vec2i = versor<int,    2>;
    using vec2f = versor<float,  2>;
    using vec2d = versor<double, 2>;
    using vec3s = versor<short,  3>;
    using vec3i = versor<int,    3>;
    using vec3f = versor<float,  3>;
    using vec3d = versor<double, 3>;
    using vec4s = versor<short,  4>;
    using vec4i = versor<int,    4>;
    using vec4f = versor<float,  4>;
    using vec4d = versor<double, 4>;

} // ::aux::algebra

/////////////////////////////////////////////////////////////////////////////
#include <concepts>
#include <type_traits>
#include <tuple>
#include <iosfwd>


namespace std
{
    using namespace aux::algebra;

    template <class T, size_t N>
    struct tuple_size<versor<T, N>> { static constexpr auto value = N; };
    template <class T, size_t N>
    constexpr size_t tuple_size_v<versor<T, N>> = N;
    template <size_t I, class T, size_t N>
    struct tuple_element<I, versor<T, N>> { using type = T; };

    template <class T, size_t I>
    concept has_tuple_element = requires(T t) {
        typename std::tuple_element_t<I, std::remove_const_t<T>>;
        { get<I>(t) } -> std::convertible_to<std::tuple_element_t<I, T> const&>;
    };
    template <class T>
    concept tuple_like = !std::is_reference_v<T> && requires(T) {
        std::tuple_size<T>::value;
        //requires std::derived_from<std::tuple_size<T>, std::integral_constant<size_t, tuple_size_v<T>>>;
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

} // ::std


/////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <stdexcept>
#include <utility>
#include <memory>
#include <string_view>

#include <wayland-client.h>
#include <wayland-client.h>

#include "xdg-shell-client.h"
#include "zwp-tablet-v2-client.h"


namespace aux::inline wayland
{
    struct empty_type { };
    template <class> constexpr wl_interface const *const interface_ptr = nullptr;

    template <class T> concept client_like = (interface_ptr<T> != nullptr);

    template <client_like T> void (*deleter)(T*) = [](auto) { static_assert("unknown deleter"); };

    template <client_like T> struct listener_type { };
#define INTERN_CLIENT_LIKE_CONCEPT(CLIENT, DELETER, LISTENER)          \
    template <> constexpr wl_interface const *const interface_ptr<CLIENT> = &CLIENT##_interface; \
    template <> void (*deleter<CLIENT>)(CLIENT*) = DELETER;   \
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
    INTERN_CLIENT_LIKE_CONCEPT(xdg_wm_base,           xdg_wm_base_destroy,           xdg_wm_base_listener)
    INTERN_CLIENT_LIKE_CONCEPT(xdg_surface,           xdg_surface_destroy,           xdg_surface_listener)
    INTERN_CLIENT_LIKE_CONCEPT(xdg_toplevel,          xdg_toplevel_destroy,          xdg_toplevel_listener)
    INTERN_CLIENT_LIKE_CONCEPT(zwp_tablet_manager_v2, zwp_tablet_manager_v2_destroy, empty_type)
    INTERN_CLIENT_LIKE_CONCEPT(zwp_tablet_seat_v2,    zwp_tablet_seat_v2_destroy,    zwp_tablet_seat_v2_listener)
    INTERN_CLIENT_LIKE_CONCEPT(zwp_tablet_tool_v2,    zwp_tablet_tool_v2_destroy,    zwp_tablet_tool_v2_listener)
#undef INTERN_CLIENT_LIKE_CONCEPT

    template <class T>
    concept client_like_with_listener = client_like<T> && !std::is_base_of_v<empty_type, listener_type<T>>;

    template <client_like T>
    auto make_unique(T* raw = nullptr) noexcept {
        return std::unique_ptr<T, decltype (deleter<T>)>(raw, deleter<T>);
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
                        ([](auto...) noexcept {
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

    using color = aux::versor<std::uint8_t, 4>;

    struct unique_fd {
        int fd;

        unique_fd(unique_fd const&) = delete;
        unique_fd& operator=(unique_fd const&) = delete;

        explicit unique_fd(int fd = -1) noexcept
            : fd{fd}
            {
            }
        unique_fd(unique_fd&& other) noexcept
            : fd{std::exchange(other.fd, -1)}
            {
            }
        ~unique_fd() noexcept {
            if (this->fd != -1) {
                ::close(fd);
                this->fd = -1;
            }
        }
        auto& operator=(unique_fd&& other) noexcept {
            if (this != &other) {
                std::swap(this->fd, other.fd);
            }
            return *this;
        }
        operator int() const noexcept { return this->fd; }
    };

    template <class T = color, wl_shm_format format = WL_SHM_FORMAT_ARGB8888, size_t bypp = 4>
    [[nodiscard]] inline auto shm_allocate_buffer(wl_shm* shm, size_t cx, size_t cy) {
        std::string_view xdg_runtime_dir = std::getenv("XDG_RUNTIME_DIR");
        if (xdg_runtime_dir.empty() || !std::filesystem::exists(xdg_runtime_dir)) {
            throw std::runtime_error("No XDG_RUNTIME_DIR settings...");
        }
        std::string tmp_path(xdg_runtime_dir);
        tmp_path += "/weston-shared-XXXXXX";
        unique_fd fd{::mkostemp(tmp_path.data(), O_CLOEXEC)};
        if (fd >= 0) {
            ::unlink(tmp_path.c_str());
        }
        else {
            throw std::runtime_error("mkostemp failed...");
        }
        if (::ftruncate(fd, bypp*cx*cy) < 0) {
            throw std::runtime_error("ftruncate failed...");
        }
        void* data = mmap(nullptr, bypp*cx*cy, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (data == MAP_FAILED) {
            throw std::runtime_error("mmap failed...");
        }
        return std::tuple{
            std::move(fd),
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

template <uint8_t A = 0xff>
constexpr aux::versor<uint8_t, 4> hue(uint32_t h) noexcept {
    constexpr uint8_t s = 0xff;
    uint8_t v = static_cast<uint8_t>(h);
    if (h <= s) return { 0, v, s, A };
    h -= s;
    v -= s;
    if (h <= s) return { 0, s,~v, A };
    h -= s;
    v -= s;
    if (h <= s) return { v, s, 0, A };
    h -= s;
    v -= s;
    if (h <= s) return { s,~v, 0, A };
    h -= s;
    v -= s;
    if (h <= s) return { s, 0, v, A };
    h -= s;
    v -= s;
    if (h <= s) return {~v, 0, s, A };

    return {0, 0, 0, A};
}

#include <set>

#include <cairo/cairo.h>
#include <linux/input-event-codes.h>

int main(int, char** argv) {
    using namespace aux;
    auto display = wrapper{wl_display_connect(nullptr)};
    auto registry = wrapper{wl_display_get_registry(display)};

    wrapper<wl_compositor> compositor;
    wrapper<wl_shm> shm;

    wrapper<wl_seat> seat;
    wrapper<wl_pointer> pointer;
    wrapper<wl_keyboard> keyboard;
    wrapper<wl_touch> touch;
    bool quit = false;

    wrapper<xdg_wm_base> shell;

    size_t scale = 2; // possible maximum scale, would be adjust to the smallest output...
    size_t cx = 1920;
    size_t cy = 1080;
    std::vector<wrapper<wl_output>> outputs;

    uint32_t M = 144;
    uint32_t N = 233;
    double D = 2.0;
    wl_surface* pointer_surface = nullptr;
    vec2d pointer_current = {};
    std::vector<vec2d> vertices;

    wrapper<zwp_tablet_manager_v2>        tablet_mgr;
    wrapper<zwp_tablet_seat_v2>           tablet_seat;
    std::set<wrapper<zwp_tablet_tool_v2>> tablet_tools;

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
                        if (s == WL_KEYBOARD_KEY_STATE_RELEASED) {
                            switch (k) {
                            case KEY_ESC:
                                vertices.clear();
                                break;
                            }
                        }
                    });
                }
                if (capabilities & WL_SEAT_CAPABILITY_POINTER) {
                    pointer = wrapper{wl_seat_get_pointer(seat)};
                    pointer->motion = lamed([&](auto, auto, auto, auto x, auto y) noexcept {
                        pointer_current = {
                            scale * wl_fixed_to_double(x),
                            scale * wl_fixed_to_double(y),
                        };
                    });
                    pointer->enter = lamed([&](auto, auto, auto, auto surface, auto x, auto y) noexcept {
                        pointer_surface = surface;
                        pointer_current = {
                            scale * wl_fixed_to_double(x),
                            scale * wl_fixed_to_double(y),
                        };
                    });
                    pointer->leave = lamed([&](auto...) noexcept {
                        pointer_surface = nullptr;
                    });
                    pointer->axis = lamed([&](auto, auto, auto, auto axis, auto value) noexcept {
                        if (axis == WL_POINTER_AXIS_HORIZONTAL_SCROLL) {
                            if (value < 0) {
                                auto tmp = M;
                                M = N;
                                N += tmp;
                            }
                            else {
                                auto tmp = N;
                                N = M;
                                M = std::max<uint32_t>(tmp - M, 1);
                            }
                            std::cout << N << std::endl;
                        }
                        if (axis == WL_POINTER_AXIS_VERTICAL_SCROLL) {
                            if (value < 0) {
                                D *= 1.1;
                            }
                            else {
                                D /= 1.1;
                            }
                            std::cout << D << std::endl;
                        }

                    });
                }
                if (capabilities & WL_SEAT_CAPABILITY_TOUCH) {
                    touch = wrapper{wl_seat_get_touch(seat)};
                    touch->motion = lamed([&](auto, auto, auto, auto, auto x, auto y) {
                        vertices.emplace_back(scale*wl_fixed_to_double(x), scale*wl_fixed_to_double(y));
                    });
                }
            });
        }
        else if (interface == interface_ptr<xdg_wm_base>->name) {
            shell = wrapper{registry_bind<xdg_wm_base>(registry, name, version)};
            shell->ping = [](auto, auto shell, auto serial) noexcept {
                xdg_wm_base_pong(shell, serial);
            };
        }
        else if (interface == interface_ptr<wl_output>->name) {
            outputs.emplace_back(wrapper{registry_bind<wl_output>(registry, name, version)});
            outputs.back()->mode = lamed([&](auto, auto, auto, int32_t width, int32_t height, auto) noexcept {
                cx = scale * 3 * std::min<size_t>(cx, width)  / 4;
                cy = scale * 3 * std::min<size_t>(cy, height) / 4;
            });
            outputs.back()->scale = lamed([&](auto, auto, auto s) {
                scale = std::min<int>(s, scale);
            });
        }
        else if (interface == interface_ptr<zwp_tablet_manager_v2>->name) {
            tablet_mgr = wrapper{registry_bind<zwp_tablet_manager_v2>(registry, name, version)};
        }

        if (tablet_mgr && seat && !tablet_seat) {
            tablet_seat = zwp_tablet_manager_v2_get_tablet_seat(tablet_mgr, seat);
            tablet_seat->tool_added = lamed([&](auto, auto, auto t) {
                auto& tool = *tablet_tools.insert(t).first;
                std::cout << "Tool added: " << std::endl;
                tool->removed = lamed([&](auto, auto t) {
                    std::cout << "Tool removed: " << std::endl;
                    tablet_tools.erase(t);
                });
                tool->type = [](auto, auto, auto type) {
                    std::cout << "Tool type: " << type << std::endl;
                };
                tool->proximity_in = [](auto...) {
                    std::cout << "Tool proximity in" << std::endl;
                };
                tool->proximity_out = [](auto...) {
                    std::cout << "Tool proximity out" << std::endl;
                };
                tool->capability = lamed([&](auto, auto, auto capability) {
                    std::cout << "Tool capability: " << capability;
                    switch (capability) {
                    case ZWP_TABLET_TOOL_V2_CAPABILITY_TILT:
                        std::cout << ", Tilt supported.";
                        tool->tilt = lamed([&](auto, auto, auto x, auto y) {
                            //vertices.back().position -= vec2d{wl  _fixed_to_double(x), wl_fixed_to_double(y)};
                            //vertices.back().pressure += std::sqrt(x*x + y*y);
                        });
                        break;
                    case ZWP_TABLET_TOOL_V2_CAPABILITY_PRESSURE:
                        std::cout << ", Pressure supported.";
                        tool->pressure = lamed([&](auto, auto, auto pressure) {
                            //vertices.back().pressure += pressure;
                        });
                        break;
                    case ZWP_TABLET_TOOL_V2_CAPABILITY_DISTANCE:
                        std::cout << ", Distance supported.";
                        tool->distance = [](auto, auto, auto distance) {
                            std::cout << "=== " << distance << std::endl;
                        };
                        break;
                    case ZWP_TABLET_TOOL_V2_CAPABILITY_ROTATION:
                        std::cout << ", Rotation supported.";
                        tool->rotation = [](auto, auto, auto rot) {
                            std::cout << "@@@ " << rot << std::endl;
                        };
                        break;
                    }
                    std::cout << std::endl;
                });
                tool->button = [](auto, auto, auto serial, auto button, auto state) {
                    std::cout << "^^^ " << std::tuple{serial, button, state} << std::endl;
                };
                tool->motion = lamed([&](auto, auto, auto x, auto y) {
                    //vertices.back().position += vec2d{wl_fixed_to_double(x), wl_fixed_to_double(y)};
                    vertices.emplace_back(scale*wl_fixed_to_double(x), scale*wl_fixed_to_double(y));
                });
                tool->frame = lamed([&](auto, auto, auto) {
                    // if (vertices.back().pressure > 0) {
                    //     vertices.push_back({});
                    // }
                });
            });
        }
    });
    wl_display_roundtrip(display);
    wl_display_roundtrip(display);

    auto surface = wrapper{wl_compositor_create_surface(compositor)};
    wl_surface_set_buffer_scale(surface, scale);
    auto xsurface = wrapper{xdg_wm_base_get_xdg_surface(shell, surface)};
    xsurface->configure = [](auto, auto xsurface, auto serial) noexcept {
        xdg_surface_ack_configure(xsurface, serial);
    };

    auto que = sycl::queue();
    std::cout << que.get_device().get_info<sycl::info::device::name>() << std::endl;
    std::cout << que.get_device().get_info<sycl::info::device::vendor>() << std::endl;
    auto channels = [&] {
        auto deleter = [&](auto ptr) { sycl::free(ptr, que); };
        using unique_type = std::unique_ptr<double, decltype (deleter)>;
        return std::array<unique_type, 4> {
            unique_type{ sycl::malloc_device<double>(cx*cy, que), deleter },
            unique_type{ sycl::malloc_device<double>(cx*cy, que), deleter },
            unique_type{ sycl::malloc_device<double>(cx*cy, que), deleter },
            unique_type{ sycl::malloc_device<double>(cx*cy, que), deleter },
        };
    }();

    auto [fd, buffer, pixels] = shm_allocate_buffer(shm, cx, cy);
    auto toplevel = wrapper{xdg_surface_get_toplevel(xsurface)};
    toplevel->configure = lamed([&](auto, auto, auto w, auto h, auto) {
        cx = scale*w;
        cy = scale*h;
        if (cx * cy) {
            std::tie(fd, buffer, pixels) = shm_allocate_buffer(shm, cx, cy);
            for (auto& channel : channels) {
                channel.reset(sycl::malloc_device<double>(cx*cy, que));
            }
        }
    });
    toplevel->close = lamed([&](auto...) {
        quit = true;
    });
    xdg_toplevel_set_app_id(toplevel, std::filesystem::path(argv[0]).filename().c_str());
    if (pointer) {
        pointer->button = lamed([&](auto, auto, auto serial, auto, auto button, auto state) noexcept {
            if (state == WL_POINTER_BUTTON_STATE_PRESSED) {
                switch (button) {
                case BTN_LEFT:
                    vertices.emplace_back(pointer_current);
                    break;
                case BTN_RIGHT:
                    xdg_toplevel_show_window_menu(toplevel, seat, serial,
                                                  pointer_current[0] / scale,
                                                  pointer_current[1] / scale); //!!!
                    break;
                }
            }
        });
    }

    wl_surface_commit(surface);

    while (wl_display_dispatch(display) != -1) {
        if (quit) {
            break;
        }
        if (cx * cy) {
            static constexpr double TAU = 2.0 * std::numbers::pi;
            static constexpr double PSI = 1.0 / std::numbers::phi;

            auto buffer_ch = std::tuple{
                sycl::buffer<double, 2>{channels[0].get(), {cy, cx}},
                sycl::buffer<double, 2>{channels[1].get(), {cy, cx}},
                sycl::buffer<double, 2>{channels[2].get(), {cy, cx}},
                sycl::buffer<double, 2>{channels[3].get(), {cy, cx}},
            };
            auto buffer_px = sycl::buffer<color, 2>{pixels, {cy, cx}};
            que.submit([&](auto& h) noexcept {
                auto ach = std::get<0>(buffer_ch).template get_access<sycl::access::mode::write>(h);
                auto rch = std::get<1>(buffer_ch).template get_access<sycl::access::mode::write>(h);
                auto gch = std::get<2>(buffer_ch).template get_access<sycl::access::mode::write>(h);
                auto bch = std::get<3>(buffer_ch).template get_access<sycl::access::mode::write>(h);
                auto pix = buffer_px.template get_access<sycl::access::mode::write>(h);
                h.parallel_for({cy, cx}, [=](auto idx) noexcept {
                    pix[idx] = { };
                    ach[idx] = { };
                    rch[idx] = { };
                    gch[idx] = { };
                    bch[idx] = { };
                });
            });
            if (vertices.empty() == false) {
                auto buffer_vtx = sycl::buffer<vec2d, 1>{vertices.data(), vertices.size()};
                que.submit([&](auto& h) noexcept {
                    auto ach = std::get<0>(buffer_ch).template get_access<sycl::access::mode::write>(h);
                    auto rch = std::get<1>(buffer_ch).template get_access<sycl::access::mode::write>(h);
                    auto gch = std::get<2>(buffer_ch).template get_access<sycl::access::mode::write>(h);
                    auto bch = std::get<3>(buffer_ch).template get_access<sycl::access::mode::write>(h);
                    auto vtx = buffer_vtx.get_access<sycl::access::mode::read>(h);
                    h.parallel_for({vertices.size()}, [=](auto idx) noexcept {
                        for (uint32_t i = 0; i < N; ++i) {
                            auto theta = std::polar(sqrt(1+i)/D, (1+i) * TAU * PSI);
                            auto pt = vtx[idx] + vec2d{theta.real(), theta.imag()};
                            size_t x = pt[0];
                            size_t y = pt[1];
                            if (0 < x && x < cx && 0 < y && y < cy) {
                                double d = 1.0 - static_cast<double>(i+1)/N;
                                ach[{y, x}] += 255; // d/4;
                                rch[{y, x}] += hue(d*1530)[2]/1;
                                gch[{y, x}] += hue(d*1530)[1]/1;
                                bch[{y, x}] += hue(d*1530)[0]/1;
                            }
                        }
                    });
                });
            }
            que.submit([&](auto& h) noexcept {
                auto ach = std::get<0>(buffer_ch).template get_access<sycl::access::mode::read>(h);
                auto rch = std::get<1>(buffer_ch).template get_access<sycl::access::mode::read>(h);
                auto gch = std::get<2>(buffer_ch).template get_access<sycl::access::mode::read>(h);
                auto bch = std::get<3>(buffer_ch).template get_access<sycl::access::mode::read>(h);
                auto pix = buffer_px.template get_access<sycl::access::mode::read_write>(h);
                h.parallel_for({cy, cx}, [=](auto idx) noexcept {
                    auto& p = pix[idx];
                    p[3] = 255;
                    p[2] = 255 - 255.0 / (rch[idx] + 1);
                    p[1] = 255 - 255.0 / (gch[idx] + 1);
                    p[0] = 255 - 255.0 / (bch[idx] + 1);
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

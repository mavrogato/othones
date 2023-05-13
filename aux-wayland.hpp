#ifndef INCLUDE_AUX_WAYLAND_HPP_E2A3E941_3904_4E36_9AA4_A6AA468FC308
#define INCLUDE_AUX_WAYLAND_HPP_E2A3E941_3904_4E36_9AA4_A6AA468FC308

#include <stdexcept>
#include <utility>
#include <string_view>

#include <wayland-client.h>
#include "xdg-shell-v6-client.h"
#include "zwp-tablet-v2-client.h"

namespace aux::inline wayland
{
    template <class> constexpr wl_interface const *const interface_ptr = nullptr;
    template <class> constexpr std::string_view interface_name = "";
    template <class T> concept client_like = (interface_ptr<T> != nullptr);
    template <client_like> constexpr void (*client_deleter)(void*) = nullptr;
    enum class null_listener_type { nil };
    template <client_like> struct listener_meta_type { using type = null_listener_type; };
    template <client_like T> using listener_type = typename listener_meta_type<T>::type;
    template <class T> concept client_like_without_listener = (std::is_same_v<listener_type<T>, null_listener_type>);
    template <class T> concept client_like_with_listener = !client_like_without_listener<T>;

#   define INTERN_CLIENT_LIKE_CONCEPT(CLIENT_LIKE, DELETER, LISTENER)         \
    template <> constexpr wl_interface const *const interface_ptr<CLIENT_LIKE> = &CLIENT_LIKE##_interface; \
    template <> constexpr std::string_view interface_name<CLIENT_LIKE> = #CLIENT_LIKE; \
    template <> constexpr void (*client_deleter<CLIENT_LIKE>)(CLIENT_LIKE*) = DELETER;     \
    template <> struct listener_meta_type<CLIENT_LIKE> { using type = LISTENER; };
    INTERN_CLIENT_LIKE_CONCEPT(wl_display,            wl_display_disconnect,         wl_display_listener)
    INTERN_CLIENT_LIKE_CONCEPT(wl_registry,           wl_registry_destroy,           wl_registry_listener)
    INTERN_CLIENT_LIKE_CONCEPT(wl_compositor,         wl_compositor_destroy,         null_listener_type)
    INTERN_CLIENT_LIKE_CONCEPT(wl_output,             wl_output_destroy,             wl_output_listener)
    INTERN_CLIENT_LIKE_CONCEPT(wl_shm,                wl_shm_destroy,                wl_shm_listener)
    INTERN_CLIENT_LIKE_CONCEPT(wl_seat,               wl_seat_destroy,               wl_seat_listener)
    INTERN_CLIENT_LIKE_CONCEPT(wl_surface,            wl_surface_destroy,            wl_surface_listener)
    INTERN_CLIENT_LIKE_CONCEPT(wl_shm_pool,           wl_shm_pool_destroy,           null_listener_type)
    INTERN_CLIENT_LIKE_CONCEPT(wl_buffer,             wl_buffer_destroy,             wl_buffer_listener)
    INTERN_CLIENT_LIKE_CONCEPT(wl_keyboard,           wl_keyboard_destroy,           wl_keyboard_listener)
    INTERN_CLIENT_LIKE_CONCEPT(wl_pointer,            wl_pointer_destroy,            wl_pointer_listener)
    INTERN_CLIENT_LIKE_CONCEPT(wl_touch,              wl_touch_destroy,              wl_touch_listener)
    INTERN_CLIENT_LIKE_CONCEPT(zxdg_shell_v6,         zxdg_shell_v6_destroy,         zxdg_shell_v6_listener)
    INTERN_CLIENT_LIKE_CONCEPT(zxdg_surface_v6,       zxdg_surface_v6_destroy,       zxdg_surface_v6_listener)
    INTERN_CLIENT_LIKE_CONCEPT(zxdg_toplevel_v6,      zxdg_toplevel_v6_destroy,      zxdg_toplevel_v6_listener)
    INTERN_CLIENT_LIKE_CONCEPT(zwp_tablet_manager_v2, zwp_tablet_manager_v2_destroy, null_listener_type)
#   undef  INTERN_CLIENT_LIKE_CONCEPT

    template <class Callback, size_t I>
    void func(void* data, auto... args) {
        std::get<I>(*reinterpret_cast<Callback*>(data))(args...);
    }

    template <client_like T, class... Callback>
    class wrapper {
    private:
        template <size_t... I>
        auto init(std::index_sequence<I...>) noexcept {
            if constexpr (sizeof... (I)) {
                return listener_type<T>{func<std::tuple<Callback...>, I>...};
            }
            else {
                return listener_type<T>{};
            }
        }

    public:
        wrapper() noexcept
            : ptr{nullptr}
            , callback{}
            , listener{}
            {
            }
        // wrapper(T* raw)
        //     : ptr{raw}
        //     , callback{}
        //     , listener{}
        //     {
        //     }
        explicit wrapper(T* raw, Callback... callback)
            : ptr{raw}
            , callback{std::tuple{callback...}}
            , listener{init(std::index_sequence_for<Callback...>())}
            {
                if (raw == nullptr) throw std::runtime_error("null wrapped...");

                if constexpr (sizeof... (callback)) {
                    if (0 != wl_proxy_add_listener(reinterpret_cast<wl_proxy*>(this->ptr),
                                                   reinterpret_cast<void(**)(void)>(&this->listener),
                                                   &this->callback))
                    {
                        throw std::runtime_error("failed to add listener...");
                    }
                    std::cout << "OK" << std::endl;
                }
            }
        // explicit wrapper(T* raw, listener_type<T> listener, void* data = nullptr)
        //     : ptr{raw}
        //     , listener{listener}
        //     {
        //         if (raw == nullptr) throw std::runtime_error("wrapped null attached...");
        //         // if (0 != wl_proxy_add_listener(reinterpret_cast<wl_proxy*>(this->ptr),
        //         //                                reinterpret_cast<void(**)(void)>(&this->listener),
        //         //                                data))
        //         // {
        //         //     throw std::runtime_error("failed to add listener...");
        //         // }
        //         add(listener, data);
        //     }
        wrapper(wrapper&& other) noexcept
            : ptr{std::exchange(other.ptr, nullptr)}
            , callback{other.callback}
            , listener{other.listener}
            {
            }

        ~wrapper() noexcept {
            if (this->ptr) {
                //std::cout << "deleting... " << this->ptr << '(' << interface_name<T> << ':' << this->get_id() << ')' << std::endl;
                client_deleter<T>(this->ptr);
                this->ptr = nullptr;
            }
        }

        auto& operator=(wrapper&& other) {
            if (this != &other) {
                this->ptr = std::exchange(other.ptr, nullptr);
                this->callback = other.callback;
                this->listener = other.listener;
            }
            return *this;
        }

        auto get() const {
            if (this->ptr) {
                return this->ptr;
            }
            throw std::runtime_error("wrapped null referred...");
        }
        operator T*() const { return this->get(); }
        explicit operator bool() const noexcept { return ptr != nullptr; }

        auto get_id() const {
            return wl_proxy_get_id(reinterpret_cast<wl_proxy*>(this->get()));
        }

        // void add(listener_type<T> listener, void* data = nullptr) {
        //     static auto sl = listener;
        //     this->listener = listener;
        //     if (0 != wl_proxy_add_listener(reinterpret_cast<wl_proxy*>(this->ptr),
        //                                    reinterpret_cast<void(**)(void)>(&this->listener),
        //                                    data))
        //     {
        //         throw std::runtime_error("failed to add listener...");
        //     }
        // }

    // public:
    //     template <class Ch>
    //     friend auto& operator<<(std::basic_ostream<Ch>& output, wrapper& v) {
    //         output << '(';
    //         output << interface_name<T> << ' ';
    //         output << v.ptr;
    //         if constexpr (sizeof (listener_type<T>) >= sizeof (void*)) {
    //             constexpr size_t N = sizeof (listener_type<T>) / sizeof (void*);
    //             for (size_t i = 0; i < N; ++i) {
    //                 output << ' ' << reinterpret_cast<void (**)(void)>(&v.listener) + i;
    //             }
    //         }
    //         output << ')';
    //         return output;
    //     }

    private:
        T* ptr;
        std::tuple<Callback...> callback;
        listener_type<T> listener;
    };

    template <client_like T>
    auto wl_registry_bind(wl_registry* registry, uint32_t name, uint32_t version) noexcept {
        return static_cast<T*>(::wl_registry_bind(registry, name, interface_ptr<T>, version));
    }

} // aux::wayland

#endif/*INCLUDE_AUX_WAYLAND_HPP_E2A3E941_3904_4E36_9AA4_A6AA468FC308*/

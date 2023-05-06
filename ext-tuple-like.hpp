#ifndef INCLUDE_EXT_TUPLE_LIKE_HPP_C042BCC2_E9E2_42D4_8F6E_01E343683BAB
#define INCLUDE_EXT_TUPLE_LIKE_HPP_C042BCC2_E9E2_42D4_8F6E_01E343683BAB

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
        }(std::make_index_sequence<std::tuple_size<T>::value>());
        return output.put(')');
    }

} // ::std::ext

#endif/*INCLUDE_EXT_TUPLE_LIKE_HPP_C042BCC2_E9E2_42D4_8F6E_01E343683BAB*/

/*
template meta-programming classes for convenience in kernel specialization
*/

#pragma once
#include <type_traits>

#include <boost/mpl/list.hpp>
#include <boost/mpl/front.hpp>
#include <boost/mpl/begin.hpp>
#include <boost/mpl/list_c.hpp>
#include <boost/mpl/pop_front.hpp>
#include <boost/mpl/empty.hpp>
#include <boost/mpl/joint_view.hpp>
#include <boost/mpl/equal.hpp>
#include <boost/mpl/insert.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/size.hpp>

#include <string>
#include <unordered_map>
#include <map>
#include <iostream>

#include "type_repr.h"

namespace fn_builder {

namespace mpl = boost::mpl;

namespace aux {

    template <typename T, typename = void>
    struct has_value : std::false_type{};

    template <typename T>
    struct has_value<T, decltype((void)T::value, void())> : std::true_type {};

    template <typename T, typename = void>
    struct has_id_fn : std::false_type{};

    template <typename T>
    struct has_id_fn<T, decltype((void)T::get_id, void())> : std::true_type {};


    template <typename T>
    std::string type_repr(std::false_type) {
        return type_repr::to_std_type_str<T>();
    }

    template <typename T>
    std::string type_repr(std::true_type) {
        return std::to_string(T::value);
    }

    template <typename Seq>
    std::string get_default_str() {
        using front = typename mpl::front<Seq>::type;
        using back  = typename mpl::pop_front<Seq>::type;
        return type_repr<front>(has_value<front>{}) + get_default_str<back>();   
    }

    template <>
    std::string get_default_str<mpl::l_end>() {
        return std::string("");
    }

    template <typename Seq>
    size_t get_default_id(int ver) {
        std::hash<std::string> hasher;
        return hasher(std::to_string(ver) + get_default_str<Seq>());
    }

    template <typename Seq, template <class ...> class C, class... ArgsSoFar>
    struct ApplyArgs {
        using front_type = typename mpl::front<Seq>::type;
        using back_list  = typename mpl::pop_front<Seq>::type;
        using type = typename ApplyArgs<back_list, C, ArgsSoFar ..., front_type>::type;
    };

    template <template <class ...> class C, class... ArgsSoFar>
    struct ApplyArgs<mpl::l_end, C, ArgsSoFar ...> {
        using type = C<ArgsSoFar ...>;
    };

    template <typename Fn, typename Seq>
    size_t id_switcher(std::true_type, int ver) {
        return Fn::get_id();
    }

    template <typename Fn, typename Seq>
    size_t id_switcher(std::false_type, int ver) {
        return get_default_id<Seq>(ver);
    }

    template <typename Seq, typename Map, int N, template <class ...> class Fn>
    struct BuildFn_ {
        static void build_func_(Map &map, int ver) {
            using front = typename mpl::front<Seq>::type;
            using fn_t  = typename ApplyArgs<front, Fn>::type;
            size_t id = id_switcher<fn_t, front>(has_id_fn<fn_t>{}, ver);            
            map[id] = &fn_t::fn;

            using back = typename mpl::pop_front<Seq>::type;
            BuildFn_<back, Map, N+1, Fn>::build_func_(map, ver);
        }

        static void build_func_(Map &map) {
            build_func_(map, 0);
        }
    };

    template <typename Map, int N, template <class ...> class Fn>
    struct BuildFn_<mpl::l_end, Map, N, Fn> {
        static void build_func_(Map &map, int ver) {}
        static void build_func_(Map &map) {}
    };
}


template <typename Seq, template <class ...> class Fn>
struct FnBuilder {
    const static int size = mpl::size<Seq>::value;
    using front = typename mpl::front<Seq>::type;
    using fn_t  = typename aux::ApplyArgs<front, Fn>::type;
    using fn_ptr = decltype(&fn_t::fn);

    using map_t = std::map<size_t, fn_ptr>;

    static map_t build_fn(int ver) {
        map_t map;
        aux::BuildFn_<Seq, map_t, 0, Fn>::build_func_(map, ver);
        return map;
    }

    static void build_fn(map_t &map, int ver) {
        aux::BuildFn_<Seq, map_t, 0, Fn>::build_func_(map, ver);
    }

    static map_t build_fn() {
        return build_fn(0);
    }

    static void build_fn(map_t &map) {
        build_fn(map, 0);
    }
};

}
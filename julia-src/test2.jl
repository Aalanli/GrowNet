map_type(::Type{Float16}) = Float32
map_type(::Type{T}) where{T} = T

struct T{S}
    a::map_type(S)
end

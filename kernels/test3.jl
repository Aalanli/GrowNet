using Core: Typeof

function s(a::Int)
    a + 3
end

println(Typeof(s))
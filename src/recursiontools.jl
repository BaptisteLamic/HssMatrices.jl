module RecursionTools

import Base.Threads: fetch
import Base.Threads: wait

export spawn
export wait
export fetch

abstract type AbstractRecursion end

struct ThreadedRecursion{T} <: AbstractRecursion
    task::T
end

struct SimpleRecursion{T,G} <: AbstractRecursion
    fun::T
    args::G
end

function spawn(f, args, threaded)
    if threaded
        return ThreadedRecursion(Threads.@spawn f(args...))
    else
        return SimpleRecursion(f, args)
    end
end

function fetch(recursion::ThreadedRecursion)
    return fetch(recursion.task)
end

function fetch(recursion::SimpleRecursion)
    return recursion.fun(recursion.args...)
end

function wait(recursion::AbstractRecursion)
    _ = fetch(recursion)
    return
end

end
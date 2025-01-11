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

struct RecursionContext
    maxSplittingDepth::Int
    depth::Int
end

function RecursionContext(multithreaded::Bool)
    if multithreaded
        maxSplittingDepth = log2(Threads.nthreads())+1
    else 
        maxSplittingDepth = 0
    end
    return RecursionContext(maxSplittingDepth,0)
end

function hasToSpawn(context::RecursionContext)
    return context.depth<context.maxSplittingDepth
end

function updateAfterSpawn(context::RecursionContext)
    return RecursionContext(context.maxSplittingDepth, context.depth+1)
end

function spawn(f, args, threaded::Bool)
    if threaded
        return ThreadedRecursion(Threads.@spawn f(args...))
    else
        return SimpleRecursion(f, args)
    end
end

function spawn(f, args, context::RecursionContext)
    if hasToSpawn(context)
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
### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ ae54cd80-f216-11ef-2c8b-bf7191131a43
using Base.Threads

# ╔═╡ 5bfec871-c3a4-4ad5-81bb-b572b4d16a39
using ThreadPinning

# ╔═╡ 81cb1aeb-49a9-4006-b15e-b59db1189a74
using LinearAlgebra, BenchmarkTools

# ╔═╡ 06d034a4-1bc7-4392-a7df-c07d9293fe45
using ChunkSplitters

# ╔═╡ 9e8ab16f-1a28-4f99-8a71-a6ab16ee01e3
using OhMyThreads: @tasks

# ╔═╡ adb5039f-86d8-4b0c-a61a-fd9588003b61
using OhMyThreads: tmapreduce

# ╔═╡ 2adcdd0e-8544-41ec-8cf3-c82d5019bf65
using PlutoUI

# ╔═╡ d02a4193-f2f9-4480-9e4a-4ea6ad06a49e
md"""
# Multithreading in Julia

_Part of this notebook is inspired by the material of th [Julia for HPC Course @ UCL ARC ](https://github.com/carstenbauer/JuliaUCL24) by Carsten Bauer._

## Setup
"""

# ╔═╡ 260d34d9-18f8-489a-b33b-f9c62312972d
nthreads()

# ╔═╡ 794e18cd-00ed-4bcb-a544-741fc287a181
md"""
## Thread pinning
"""

# ╔═╡ 4a31b295-ec6f-4285-aaa9-03fa83df60b0
with_terminal() do
	threadinfo()
end

# ╔═╡ 25e4adfe-b5cf-4491-b599-f66426f3a20f
with_terminal() do
	pinthreads(:cores)
	threadinfo()
end

# ╔═╡ e59ce18a-4f7d-48b1-a6c7-851926e917b5
md"""
## Spawning parallel tasks
"""

# ╔═╡ f372c9fe-6598-4978-b49a-e184ae734d5a
@time t = @spawn begin # `@spawn` returns right away
    sleep(2)
    3 + 3
end

# ╔═╡ 4a953505-9b84-4ffd-948d-79a7c8c6985b
@time fetch(t) # `fetch` waits for the task to finish

# ╔═╡ 8084e799-d9e6-4217-8ab1-aa786ceb9761
md"""
## Example: multi-threaded `map`
"""

# ╔═╡ acc1c4eb-de2d-4dc7-a63f-25f2ee2fed6e
BLAS.set_num_threads(1) # Fix number of BLAS threads

# ╔═╡ 7f4de8b1-85ec-41a8-83e2-77bf73b5a35f
function tmap(fn, itr)
    # for each i ∈ itr, spawn a task to compute fn(i)
    tasks = map(i -> @spawn(fn(i)), itr)
    # fetch and return all the results
    return fetch.(tasks)
end

# ╔═╡ 43187c6d-7ebb-474f-b07a-db1f34e5e0c7
M = [rand(100,100) for i in 1:(8 * nthreads())];

# ╔═╡ 3efbd364-cba1-420d-b350-245ade794bab
tmap(svdvals, M);

# ╔═╡ 2973daf0-fdf5-493c-b26f-460b5c59bfe9
serial_map_svdals_b = @benchmark map(svdvals, $M) samples=10 evals=3

# ╔═╡ 30d01115-5724-4d31-99ab-0228c4383442
threaded_map_svdals_b = @benchmark tmap(svdvals, $M) samples=10 evals=3

# ╔═╡ c1145c9d-edbf-4d80-8633-958f99910ad2
(minimum(serial_map_svdals_b.times) / minimum(threaded_map_svdals_b.times)) / nthreads() * 100 # parallel efficiency

# ╔═╡ 6e87dbc7-7040-4c80-80fe-d15be7dbe6cd
md"""
_**Exercise**_: do you see any difference if you increase the number of BLAS threads?

## Example: multi-threaded `for` loop (reduction)
"""

# ╔═╡ 910a5bec-b7e1-4899-8fab-52d30b35780c
function sum_threads(fn, data; nchunks=nthreads())
    psums = zeros(eltype(data), nchunks)
    @threads :static for (c, elements) in enumerate(chunks(data; n=nchunks))
        psums[c] = sum(fn, elements)
    end
    return sum(psums)
end

# ╔═╡ b548b502-7585-4918-95e9-246e9a29e611
v = randn(20_000_000);

# ╔═╡ 19ab6fa8-2519-4c21-81e5-dd7b75c82222
elapsed_serial_sum = @belapsed sum(sin, $v)

# ╔═╡ 459b2db4-df14-4673-8906-18754e71cfaf
elapsed_threaded_sum = @belapsed sum_threads(sin, $v)

# ╔═╡ 8e9cef78-96c5-47d7-81b0-8503dc9e36ce
(elapsed_serial_sum / elapsed_threaded_sum) / nthreads() * 100 # parallel efficiency

# ╔═╡ 22a457d4-9564-49e8-9ae0-9c7228e9f9b1
md"""
_**Exercise**_: do you see differences if you change the scheduler type? Remember you can choose between `:dynamic` (currently the default if omitted), `:greedy` (only if using Julia v1.11+), and `:static`.
"""

# ╔═╡ 01194bb1-7f55-4926-903a-0c20e287de1f
function sum_map_spawn(fn, data; nchunks=nthreads())
    ts = map(chunks(data, n=nchunks)) do elements
        @spawn sum(fn, elements)
    end
    return sum(fetch.(ts))
end

# ╔═╡ 8836cbc4-f239-4521-8702-3ed84a7c409a
elapsed_spawn_sum = @belapsed sum_map_spawn(sin, $v)

# ╔═╡ 8c2e0ba6-db0f-4a3b-b629-a2c4d25d517c
(elapsed_serial_sum / elapsed_spawn_sum) / nthreads() * 100 # parallel efficiency

# ╔═╡ 8cf8c11b-2098-4323-b397-47bd86bbbe7c
md"""
### Bonus: using `OhMyThreads.jl`
"""

# ╔═╡ 25e597e6-f9af-4473-8f04-7c6d7dbd8c4f
function sum_tasks(fn, data; nchunks=nthreads())
    psums = zeros(eltype(data), nchunks)
    @tasks for (c, elements) in enumerate(chunks(data; n=nchunks))
        psums[c] = sum(fn, elements)
    end
    return sum(psums)
end

# ╔═╡ 55310da7-78b8-406d-9356-736aad25551a
elapsed_tasks_sum = @belapsed sum_tasks(sin, $v)

# ╔═╡ 6996cb25-04ca-4887-ad91-f48d762acb99
(elapsed_serial_sum / elapsed_tasks_sum) / nthreads() * 100 # parallel efficiency

# ╔═╡ b856eb64-8d82-4d60-806b-7bc92de1307d
elapsed_tmapreduce_sum = @belapsed tmapreduce(sin, +, $v)

# ╔═╡ 9677c335-38cd-42de-b29a-76197b2eef24
(elapsed_serial_sum / elapsed_tmapreduce_sum) / nthreads() * 100 # parallel efficiency

# ╔═╡ f81966bb-bd14-4bd9-88cd-d850fe724df5
md"""
## Multi-threading: is it always worth it?
"""

# ╔═╡ b0753df5-f7f3-4724-ba95-4a19210758a4
function overhead!(v)
    for idx in eachindex(v)
        v[idx] = idx
    end
	return v
end

# ╔═╡ 08dcc36f-a1ed-4d24-92e6-a5bca2f2fcf7
function overhead_threads!(v)
	@threads for idx in eachindex(v)
		v[idx] = idx
	end
	return v
end

# ╔═╡ c11cbb81-99d1-486b-9919-fc7fe976618b
md"N = $(@bind N Slider(10 .^ (0:9); default=10, show_value=true))"

# ╔═╡ 662dfbf2-1d02-468c-aabf-2a24bdc86e54
@assert overhead!(Vector{Int}(undef, N)) == overhead_threads!(Vector{Int}(undef, N))

# ╔═╡ cbc0c9e5-0591-41fe-8033-c3cebefcf349
@btime overhead!(v) setup=(v = Vector{Int}(undef, N));

# ╔═╡ 467afccf-d12a-4714-897a-243e59ce9c4d
@btime overhead_threads!(v) setup=(v = Vector{Int}(undef, N));

# ╔═╡ e49d2504-ba42-448b-bd5e-77e44294d9be
md"""
_**Exercise**_: do you see any improvement in the parallel efficiency if you change the size of the problem (here: `N`)?

## Unbalanced workload: computing hexadecimal $\pi$

_This section is inspired by the blogpost [Computing the hexadecimal value of pi](https://giordano.github.io/blog/2017-11-21-hexadecimal-pi/) by Mosè Giordano._

The [Bailey–Borwein–Plouffe formula](https://en.wikipedia.org/wiki/Bailey%E2%80%93Borwein%E2%80%93Plouffe_formula) is one of the [several algorithms to compute $\pi$](https://en.wikipedia.org/wiki/Approximations_of_%CF%80):

```math
\pi = \sum_{k = 0}^{\infty}\left[ \frac{1}{16^k} \left( \frac{4}{8k + 1} -
\frac{2}{8k + 4} - \frac{1}{8k + 5} - \frac{1}{8k + 6} \right) \right]
```

What makes this formula stand out among other approximations of $\pi$ is that it allows one to directly extract the $n$-th fractional digit of the hexadecimal value of $\pi$ without computing the preceding ones.

The Wikipedia article about the Bailey–Borwein–Plouffe formula explains that the $n + 1$-th fractional digit $d_n$ is given by

```math
d_{n} = 16 \left[ 4 \Sigma(n, 1) - 2 \Sigma(n, 4) - \Sigma(n, 5) - \Sigma(n, 6) \right]
```

where

```math
\Sigma(n, j) = \sum_{k = 0}^{n} \frac{16^{n-k} \bmod (8k+j)}{8k+j} + \sum_{k=n+1}^{\infty} \frac{16^{n-k}}{8k+j}
```

Only the fractional part of expression in square brackets on the right side of $d_n$ is relevant, thus, in order to avoid rounding errors, when we compute each term of the finite sum above we can take only the fractional part. This allows us to always use ordinary double precision floating-point arithmetic, without resorting to arbitrary-precision numbers. In addition note that the terms of the infinite sum get quickly very small, so we can stop the summation when they become negligible.

### Serial implementation
"""

# ╔═╡ f13f0d0d-5d50-4338-875b-f32f9057fcba
# Return the fractional part of x, modulo 1, always positive
fpart(x) = mod(x, one(x))

# ╔═╡ c13e9aaa-65c9-45ef-bee6-3041a9f81161
function Σ(n, j)
    # Compute the finite sum
    s = 0.0
    denom = j
    for k in 0:n
        s = fpart(s + powermod(16, n - k, denom) / denom)
        denom += 8
    end
    # Compute the infinite sum
    num = 1 / 16
    while (frac = num / denom) > eps(s)
        s     += frac
        num   /= 16
        denom += 8
    end
    return fpart(s)
end

# ╔═╡ e9b8663f-dc5e-4843-b0cd-9c25f44ff022
pi_digit(n) =
    floor(Int, 16 * fpart(4Σ(n-1, 1) - 2Σ(n-1, 4) - Σ(n-1, 5) - Σ(n-1, 6)))

# ╔═╡ 8c6472d9-2827-4065-bde7-93bc87049492
pi_string(n) = "0x3." * join(string.(pi_digit.(1:n), base = 16)) * "p0"

# ╔═╡ d3e3b29b-2101-41e8-89f0-a20b35c362d6
md"""
Let's make sure this works:
"""

# ╔═╡ 47a0bdf9-1735-453e-b2d3-8f6a31de3956
pi_string(13)

# ╔═╡ 93b23606-a593-474c-9554-e356af1ef38f
# Parse the string as a double-precision floating point number
parse(Float64, pi_string(13))

# ╔═╡ 955068de-9d57-4187-bcad-49d31e0ebc8d
Float64(π) == parse(Float64, pi_string(13))

# ╔═╡ 48070266-168b-456c-a352-214832bbf08f
N_pi = 1_000

# ╔═╡ 7aee39f9-3ab2-4045-b9e9-cbfddfd9c053
setprecision(BigFloat, 4 * N_pi) do
    BigFloat(π) == parse(BigFloat, pi_string(N_pi))
end

# ╔═╡ 241418a3-7235-48ae-a4c5-1bfd3fff96c7
pi_serial_b = @benchmark pi_string(N_pi)

# ╔═╡ 8a1a20f6-e0c2-43c2-8b51-ff315e797352
pi_serial_t = minimum(pi_serial_b.times)

# ╔═╡ a183420d-b8e5-4d29-895c-b80461d15381
md"""
### Multi-threaded implementation

Since the Bailey–Borwtimesn–Plouffe formula extracts the $n$-th digit of $\pi$ without computing the other ones, we can write a multi-threaded version of `pi_string`, taking advantage of native support for [multi-threading](https://docs.julialang.org/en/v1/manual/multi-threading/) in Julia. However note that the computational cost of `pi_digit` is $O(n\log(n))$, so the larger the value of $n$, the longer the function will take, which makes this workload very unbalanced. _**Question**_: what do you expect to be the worst performing scheduler?

#### For-loop: static scheduler
"""

# ╔═╡ feede8b2-0755-40cc-a6e5-a0855fb87cba
function pi_string_threads_static(N)
    digits = Vector{Int}(undef, N)
    @threads :static for n in eachindex(digits)
        digits[n] = pi_digit(n)
    end
    return "0x3." * join(string.(digits, base = 16)) * "p0"
end

# ╔═╡ 38a05e8e-482f-4a69-8c4c-b62fb157cf73
@assert pi_string_threads_static(N_pi) == pi_string(N_pi)

# ╔═╡ 7d87cfba-280d-4b04-8abd-9ea5e97c9c64
pi_threads_static_b = @benchmark pi_string_threads_static(N_pi)

# ╔═╡ 92e165ab-0b75-4334-b904-88f991a13745
pi_serial_t / minimum(pi_threads_static_b.times) / nthreads() * 100 # parallel efficiency

# ╔═╡ 91e0e138-3d29-48be-810d-465488c49406
md"""
#### For-loop: dynamic scheduler
"""

# ╔═╡ be136b87-4421-4607-b40c-672c46011f58
function pi_string_threads_dynamic(N)
    digits = Vector{Int}(undef, N)
    @threads :dynamic for n in eachindex(digits)
        digits[n] = pi_digit(n)
    end
    return "0x3." * join(string.(digits, base = 16)) * "p0"
end

# ╔═╡ d55a8275-b030-4847-a75d-242fee7b00ec
@assert pi_string_threads_dynamic(N_pi) == pi_string(N_pi)

# ╔═╡ e9d36306-2cdd-4015-b790-0941c3216e53
pi_threads_dynamic_b = @benchmark pi_string_threads_dynamic(N_pi)

# ╔═╡ b998b5aa-da7c-4fcf-98e7-67c013d297b4
pi_serial_t / minimum(pi_threads_dynamic_b.times) / nthreads() * 100 # parallel efficiency

# ╔═╡ 3c5dc7f5-a08d-4e54-9aea-89b8cdfe0f06
md"""
#### For-loop: greedy scheduler (only Julia v1.11+)
"""

# ╔═╡ 787cc8b2-a541-4614-84f1-ce641fd6f96d
function pi_string_threads_greedy(N)
    digits = Vector{Int}(undef, N)
    @threads :greedy for n in eachindex(digits)
        digits[n] = pi_digit(n)
    end
    return "0x3." * join(string.(digits, base = 16)) * "p0"
end

# ╔═╡ 7bd079f3-fe13-4191-a01d-5f6288fba5bf
@assert pi_string_threads_greedy(N_pi) == pi_string(N_pi)

# ╔═╡ a7edbd19-0342-4235-9c11-df49e54059b7
pi_threads_greedy_b = @benchmark pi_string_threads_greedy(N_pi)

# ╔═╡ 1cacbcb4-82a3-4769-876f-e61a04c04626
pi_serial_t / minimum(pi_threads_greedy_b.times) / nthreads() * 100 # parallel efficiency

# ╔═╡ deea8134-0952-4948-8a92-bdada62a60f3
md"""
#### Tasks
"""

# ╔═╡ b8111477-848e-4862-abc0-00bf54545109
function pi_string_tasks(N)
    tasks = [Threads.@spawn pi_digit(n) for n in 1:N]
    digits = [fetch(t) for t in tasks]
    return "0x3." * join(string.(digits, base = 16)) * "p0"
end

# ╔═╡ 3b924baf-cc2a-41b3-8af6-bc7107cbe21d
@assert pi_string_tasks(N_pi) == pi_string(N_pi)

# ╔═╡ d4854d9e-824f-4225-9ce8-8ddefb543478
pi_tasks_b = @benchmark pi_string_tasks(N_pi)

# ╔═╡ 2e4ff435-60db-49c2-8822-1f6d4c35d04c
pi_serial_t / minimum(pi_tasks_b.times) / nthreads() * 100 # parallel efficiency

# ╔═╡ 5a183210-6fb3-416e-aa1f-e8de5c3b7672
md"""
#### Bonus: using `OhMyThreads.jl`
"""

# ╔═╡ d92687f4-40fd-488f-8348-770b254e1d5f
function pi_string_omt(N; ntasks::Int=8 * nthreads(), scheduler::Symbol=:dynamic)
    digits = Vector{Int}(undef, N)
    @tasks for n in eachindex(digits)
        @set ntasks=ntasks
        @set scheduler=scheduler
        digits[n] = pi_digit(n)
    end
    return "0x3." * join(string.(digits, base = 16)) * "p0"
end

# ╔═╡ 664c1906-1e3a-4d35-a131-b8fc3a4cb329
@assert pi_string_omt(N_pi) == pi_string(N_pi)

# ╔═╡ edc876d6-3ad4-4016-9e9b-f1bffc704284
pi_omt_b = @benchmark pi_string_omt(N_pi; ntasks=32 * nthreads())

# ╔═╡ 287e5268-25a2-41ad-aa34-da65ac4f4553
pi_serial_t / minimum(pi_omt_b.times) / nthreads() * 100 # parallel efficiency

# ╔═╡ 0c103dd4-8933-4d11-9d37-2feb64e5c5cc
md"""
## Notebook Setup
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
ChunkSplitters = "ae650224-84b6-46f8-82ea-d812ca08434e"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
OhMyThreads = "67456a42-1dca-4109-a031-0a68de7e3ad5"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
ThreadPinning = "811555cd-349b-4f26-b7bc-1f208b848042"

[compat]
BenchmarkTools = "~1.5.0"
ChunkSplitters = "~3.1.1"
OhMyThreads = "~0.7.0"
PlutoUI = "~0.7.61"
ThreadPinning = "~1.0.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.3"
manifest_format = "2.0"
project_hash = "3831fc4791e0ba95756e3baedd51bfcd8471520f"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "3b86719127f50670efe356bc11073d84b4ed7a5d"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.42"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.BangBang]]
deps = ["Accessors", "ConstructionBase", "InitialValues", "LinearAlgebra", "Requires"]
git-tree-sha1 = "e2144b631226d9eeab2d746ca8880b7ccff504ae"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.3"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTablesExt = "Tables"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "f1dff6729bc61f4d49e140da1af55dcd1ac97b2f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.5.0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.ChunkSplitters]]
git-tree-sha1 = "efd065d66c7d683e355a14f32ef1e149dbd37b24"
uuid = "ae650224-84b6-46f8-82ea-d812ca08434e"
version = "3.1.1"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConstructionBase]]
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Hwloc]]
deps = ["CEnum", "Hwloc_jll", "Printf"]
git-tree-sha1 = "6a3d80f31ff87bc94ab22a7b8ec2f263f9a6a583"
uuid = "0e44f5e4-bd66-52a0-8798-143a42290a1d"
version = "3.3.0"

    [deps.Hwloc.extensions]
    HwlocTrees = "AbstractTrees"

    [deps.Hwloc.weakdeps]
    AbstractTrees = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f93a9ce66cd89c9ba7a4695a47fd93b4c6bc59fa"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.12.0+0"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "a007feb38b422fbdab534406aeca1b86823cb4d6"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "1833212fd6f580c20d4291da9c1b4e8a655b128e"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.0.0"

[[deps.MacroTools]]
git-tree-sha1 = "72aebe0b5051e5143a079a4685a46da330a40472"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.15"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OhMyThreads]]
deps = ["BangBang", "ChunkSplitters", "StableTasks", "TaskLocalValues"]
git-tree-sha1 = "5f81bdb937fd857bac9548fa8ab9390a06864bb5"
uuid = "67456a42-1dca-4109-a031-0a68de7e3ad5"
version = "0.7.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"

    [deps.Pkg.extensions]
    REPLExt = "REPL"

    [deps.Pkg.weakdeps]
    REPL = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "7e71a55b87222942f0f9337be62e26b1f103d3e4"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.61"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.StableTasks]]
git-tree-sha1 = "db1a5a7807c3b21fbbd853c835ce4fcb178993c7"
uuid = "91464d47-22a1-43fe-8b7f-2d57ee82463f"
version = "0.1.6"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SysInfo]]
deps = ["Dates", "DelimitedFiles", "Hwloc", "PrecompileTools", "Random", "Serialization"]
git-tree-sha1 = "7aaebfbf5b3a39268f4a0caaa43e878e1138d25c"
uuid = "90a7ee08-a23f-48b9-9006-0e0e2a9e4608"
version = "0.3.0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TaskLocalValues]]
git-tree-sha1 = "d155450e6dff2a8bc2fcb81dcb194bd98b0aeb46"
uuid = "ed4db957-447d-4319-bfb6-7fa9ae7ecf34"
version = "0.1.2"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.ThreadPinning]]
deps = ["DelimitedFiles", "Libdl", "LinearAlgebra", "PrecompileTools", "Preferences", "Random", "StableTasks", "SysInfo", "ThreadPinningCore"]
git-tree-sha1 = "d47dbc7862f69ce1973fff227237275ff4a10781"
uuid = "811555cd-349b-4f26-b7bc-1f208b848042"
version = "1.0.2"

    [deps.ThreadPinning.extensions]
    DistributedExt = "Distributed"
    MPIExt = "MPI"

    [deps.ThreadPinning.weakdeps]
    Distributed = "8ba89e20-285c-5b6f-9357-94700520ee1b"
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"

[[deps.ThreadPinningCore]]
deps = ["LinearAlgebra", "PrecompileTools", "StableTasks"]
git-tree-sha1 = "bb3c6f3b5600fbff028c43348365681b34d06499"
uuid = "6f48bc29-05ce-4cc8-baad-4adcba581a18"
version = "0.4.5"

[[deps.Tricks]]
git-tree-sha1 = "6cae795a5a9313bbb4f60683f7263318fc7d1505"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.10"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╟─d02a4193-f2f9-4480-9e4a-4ea6ad06a49e
# ╠═ae54cd80-f216-11ef-2c8b-bf7191131a43
# ╠═260d34d9-18f8-489a-b33b-f9c62312972d
# ╟─794e18cd-00ed-4bcb-a544-741fc287a181
# ╠═5bfec871-c3a4-4ad5-81bb-b572b4d16a39
# ╠═4a31b295-ec6f-4285-aaa9-03fa83df60b0
# ╠═25e4adfe-b5cf-4491-b599-f66426f3a20f
# ╟─e59ce18a-4f7d-48b1-a6c7-851926e917b5
# ╠═f372c9fe-6598-4978-b49a-e184ae734d5a
# ╠═4a953505-9b84-4ffd-948d-79a7c8c6985b
# ╟─8084e799-d9e6-4217-8ab1-aa786ceb9761
# ╠═81cb1aeb-49a9-4006-b15e-b59db1189a74
# ╠═acc1c4eb-de2d-4dc7-a63f-25f2ee2fed6e
# ╠═7f4de8b1-85ec-41a8-83e2-77bf73b5a35f
# ╠═43187c6d-7ebb-474f-b07a-db1f34e5e0c7
# ╠═3efbd364-cba1-420d-b350-245ade794bab
# ╠═2973daf0-fdf5-493c-b26f-460b5c59bfe9
# ╠═30d01115-5724-4d31-99ab-0228c4383442
# ╠═c1145c9d-edbf-4d80-8633-958f99910ad2
# ╟─6e87dbc7-7040-4c80-80fe-d15be7dbe6cd
# ╠═06d034a4-1bc7-4392-a7df-c07d9293fe45
# ╠═910a5bec-b7e1-4899-8fab-52d30b35780c
# ╠═b548b502-7585-4918-95e9-246e9a29e611
# ╠═19ab6fa8-2519-4c21-81e5-dd7b75c82222
# ╠═459b2db4-df14-4673-8906-18754e71cfaf
# ╠═8e9cef78-96c5-47d7-81b0-8503dc9e36ce
# ╟─22a457d4-9564-49e8-9ae0-9c7228e9f9b1
# ╠═01194bb1-7f55-4926-903a-0c20e287de1f
# ╠═8836cbc4-f239-4521-8702-3ed84a7c409a
# ╠═8c2e0ba6-db0f-4a3b-b629-a2c4d25d517c
# ╟─8cf8c11b-2098-4323-b397-47bd86bbbe7c
# ╠═9e8ab16f-1a28-4f99-8a71-a6ab16ee01e3
# ╠═25e597e6-f9af-4473-8f04-7c6d7dbd8c4f
# ╠═55310da7-78b8-406d-9356-736aad25551a
# ╠═6996cb25-04ca-4887-ad91-f48d762acb99
# ╠═adb5039f-86d8-4b0c-a61a-fd9588003b61
# ╠═b856eb64-8d82-4d60-806b-7bc92de1307d
# ╠═9677c335-38cd-42de-b29a-76197b2eef24
# ╟─f81966bb-bd14-4bd9-88cd-d850fe724df5
# ╠═b0753df5-f7f3-4724-ba95-4a19210758a4
# ╠═08dcc36f-a1ed-4d24-92e6-a5bca2f2fcf7
# ╠═662dfbf2-1d02-468c-aabf-2a24bdc86e54
# ╟─c11cbb81-99d1-486b-9919-fc7fe976618b
# ╠═cbc0c9e5-0591-41fe-8033-c3cebefcf349
# ╠═467afccf-d12a-4714-897a-243e59ce9c4d
# ╟─e49d2504-ba42-448b-bd5e-77e44294d9be
# ╠═f13f0d0d-5d50-4338-875b-f32f9057fcba
# ╠═c13e9aaa-65c9-45ef-bee6-3041a9f81161
# ╠═e9b8663f-dc5e-4843-b0cd-9c25f44ff022
# ╠═8c6472d9-2827-4065-bde7-93bc87049492
# ╟─d3e3b29b-2101-41e8-89f0-a20b35c362d6
# ╠═47a0bdf9-1735-453e-b2d3-8f6a31de3956
# ╠═93b23606-a593-474c-9554-e356af1ef38f
# ╠═955068de-9d57-4187-bcad-49d31e0ebc8d
# ╠═48070266-168b-456c-a352-214832bbf08f
# ╠═7aee39f9-3ab2-4045-b9e9-cbfddfd9c053
# ╠═241418a3-7235-48ae-a4c5-1bfd3fff96c7
# ╠═8a1a20f6-e0c2-43c2-8b51-ff315e797352
# ╟─a183420d-b8e5-4d29-895c-b80461d15381
# ╠═feede8b2-0755-40cc-a6e5-a0855fb87cba
# ╠═38a05e8e-482f-4a69-8c4c-b62fb157cf73
# ╠═7d87cfba-280d-4b04-8abd-9ea5e97c9c64
# ╠═92e165ab-0b75-4334-b904-88f991a13745
# ╟─91e0e138-3d29-48be-810d-465488c49406
# ╠═be136b87-4421-4607-b40c-672c46011f58
# ╠═d55a8275-b030-4847-a75d-242fee7b00ec
# ╠═e9d36306-2cdd-4015-b790-0941c3216e53
# ╠═b998b5aa-da7c-4fcf-98e7-67c013d297b4
# ╟─3c5dc7f5-a08d-4e54-9aea-89b8cdfe0f06
# ╠═787cc8b2-a541-4614-84f1-ce641fd6f96d
# ╠═7bd079f3-fe13-4191-a01d-5f6288fba5bf
# ╠═a7edbd19-0342-4235-9c11-df49e54059b7
# ╠═1cacbcb4-82a3-4769-876f-e61a04c04626
# ╟─deea8134-0952-4948-8a92-bdada62a60f3
# ╠═b8111477-848e-4862-abc0-00bf54545109
# ╠═3b924baf-cc2a-41b3-8af6-bc7107cbe21d
# ╠═d4854d9e-824f-4225-9ce8-8ddefb543478
# ╠═2e4ff435-60db-49c2-8822-1f6d4c35d04c
# ╟─5a183210-6fb3-416e-aa1f-e8de5c3b7672
# ╠═d92687f4-40fd-488f-8348-770b254e1d5f
# ╠═664c1906-1e3a-4d35-a131-b8fc3a4cb329
# ╠═edc876d6-3ad4-4016-9e9b-f1bffc704284
# ╠═287e5268-25a2-41ad-aa34-da65ac4f4553
# ╟─0c103dd4-8933-4d11-9d37-2feb64e5c5cc
# ╠═2adcdd0e-8544-41ec-8cf3-c82d5019bf65
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002

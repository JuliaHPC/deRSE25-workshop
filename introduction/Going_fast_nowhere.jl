### A Pluto.jl notebook ###
# v0.20.3

#> [frontmatter]
#> title = "Going fast nowhere"
#> date = "2023-10-12"
#> license = "MIT"
#> description = "A talk about benchmarking and surprising subtilities"
#> tags = ["talks", "benchmarking", "BenchmarkTools", "low-level", "julia"]
#> 
#>     [[frontmatter.author]]
#>     name = "Valentin Churavy"
#>     url = "https://vchuravy.dev"
#>     image = "https://avatars.githubusercontent.com/u/145258?v=4"

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

# ╔═╡ 2ae8a548-76e9-11ed-2045-d960f32fa458
begin
	using BenchmarkTools
	using LLVM
	using LLVM.Interop
	using Unitful
	using PlutoUI
	using BenchmarkPlots
end

# ╔═╡ f133db40-461f-4a09-af14-c5d99ef6ce04
html"<button onclick='present()'>Toggle presentation mode</button>"

# ╔═╡ 1fe94954-965b-43be-ad90-3711f07bc25d
md"""
## A note on benchmarking

**Premature optimization is the root of all evil & If you don't measure you won't improve**

### Tools

- BenchmarkTools.jl https://github.com/JuliaCI/BenchmarkTools.jl
- Profiler https://docs.julialang.org/en/latest/manual/profile/
- ProfileView.jl https://github.com/timholy/ProfileView.jl
- PProf.jl https://github.com/JuliaPerf/PProf.jl

### Other

- VTunes/Perf https://docs.julialang.org/en/latest/manual/profile/#External-Profiling-1
- LIKWID.jl
- MCAnalyzer.jl
"""

# ╔═╡ aae1c36f-681d-4a91-b46f-294411643b92
md"""
## BenchmarkTools.jl

Solid package that tries to eliminate common pitfalls in performance measurment.

- `@benchmark` macro that will repeatedly evaluate your code to gain enough samples
- Caveat: You probably want to escape $ your input data
"""

# ╔═╡ 7b3f894d-bca9-448a-8cbc-15799e239a9a

@bind N Slider(1:9)

# ╔═╡ fb5c3d82-7003-4989-b1d2-ddd8c05d75eb
data = rand(10^N);

# ╔═╡ 1843304e-ad7e-4820-a88f-bd8fbedebae6
function sum(X::Vector{Float64})
    acc = 0::Int64
    for x in X
        (acc += x)::Float64
    end
    acc::Union{Int64, Float64}
end
    

# ╔═╡ 6828ebc2-e8e1-4bfb-9961-52c9ab37e8a4
@benchmark sum($data)

# ╔═╡ 0886d018-b4aa-4ec6-a10c-15248ab0f108
md"""
## Figuring out what is happening
The stages of the compiler

- `@code_lowered`
- `@code_typed` & `@code_warntype`
- `@code_llvm`
- `@code_native`

Where is a function defined `@which` & `@edit`
"""

# ╔═╡ 294f27e9-f7a5-4c08-be98-6b49cd87377a
@code_warntype sum(data)

# ╔═╡ 97bf6798-6200-4ddc-9c97-fd695180bfe4
md"""
## A simple example: counting
"""

# ╔═╡ c2ce0d4b-3dfe-4f50-a9f3-753178d4f7d5
function f(N)
    acc = 0
    for i in 1:N
        acc += 1
    end
    return acc
end

# ╔═╡ e4d6cdac-9e4f-45be-b132-38a2d20b492b
K = 100_000_000*100

# ╔═╡ 11dcf6cb-c7a4-4bc7-ac51-6e0f17a99ee0
result = @benchmark f($N)

# ╔═╡ 3c5dcf61-7f18-41a7-8a24-56f515cbdeb2
begin
	t = time(minimum(result)) * u"ns" # in ns
	pFreq = round(typeof(1u"PHz"), K/t)
end;

# ╔═╡ 6bbbed58-75ed-4dd1-995e-d1a212e25425
md"""

So we are doing **$(K/1_000_000)** million additions in **$t**...

That would mean our processor operates at **$pFreq**

We wish...

Let's do a basic check, 10x bigger input.
"""

# ╔═╡ e9cb13b8-be41-45ac-a377-beb35a237cb7
@benchmark f($(Ref(10*K))[])

# ╔═╡ 69875711-258f-4fae-b0b4-4ea3ea482b4a
md"""
Let's explore what code we are **actually** running.

Using Julia's reflection macros we can see all of the stages of code-generation.
"""

# ╔═╡ 345be244-db9d-4b97-ac7d-18e108e5dc97
@code_lowered f(K)

# ╔═╡ 04b5bbf5-2a5e-4952-8db1-ec03a2714e83
@code_typed optimize=false f(K)

# ╔═╡ a877d8dc-5c9d-41c2-88c7-775adfb63c2f
@code_typed optimize=true f(K)

# ╔═╡ f7d79610-06f8-4864-9a6e-44f88b8caae6
with_terminal() do
	@code_llvm optimize=false f(K)
end

# ╔═╡ 4c8fea62-0607-4af0-a34e-7198a0cdb9cd
with_terminal() do
	@code_llvm optimize=true f(K)
end

# ╔═╡ 4b8d16ed-bc5b-4bee-b0e0-183de945175a
with_terminal() do
	@code_native f(K)
end

# ╔═╡ fb4e75e7-cce2-4b7d-aaf0-527bf01c3866
md"""
### Conclusion
LLVM realised that our loop:
```julia
for i in 1:N
  acc += 1
end
```

Just ended up being $acc = 1 * N$
"""

# ╔═╡ 14ad9189-88d0-447a-9b5a-df5626795f50
md"""
## Exercise

```julia
function g(N)
    acc = 0
    for i in 1:N
        acc += 1.0
    end
    acc
end  
```

```julia
function h(N)
    acc = 0.0
    for i in 1:N
        acc += 1.0
    end
    acc
end
```

Take some time to study the different stages of the compilation pipeline.
"""

# ╔═╡ a975d900-a348-43fe-b635-71a3785c7731
function h(N)
    acc = 0.0
    for i in 1:N
        acc += 1.0
    end
    acc
end

# ╔═╡ d85de968-78f4-40d0-a90f-085f2b725a08
md"""
## Can we actually measure the speed of our original code?
"""

# ╔═╡ 55d7238f-89ac-4ae2-8ed3-79093cb279cc
"""
    clobber()

Force the compiler to flush pending writes to global memory.
Acts as an effective read/write barrier.
"""
@inline clobber() = @asmcall("", "~{memory}", true) 

# ╔═╡ 135431d7-cd46-44bc-87ce-9eb810b9a7a5
"""
    escape(val)

The `escape` function can be used to prevent a value or
expression from being optimized away by the compiler. This function is
intended to add little to no overhead.
See: https://youtu.be/nXaxk27zwlk?t=2441
"""
@inline escape(val::T) where T = @asmcall("", "X,~{memory}", true, Nothing, Tuple{T}, val)

# ╔═╡ 1d55abb6-eb4d-4694-92b7-c6613522d07b
function k(::Type{T}, N) where T
    acc = zero(T)
    for i in 1:N
        acc += one(T)
        clobber()
    end
    return acc
end

# ╔═╡ 1aa09f44-bf05-4d75-8aac-39642927cbb1
with_terminal() do
	@code_llvm debuginfo=:none k(Int64, 10)
end

# ╔═╡ e2346440-a593-4954-9f73-8999ad33a2d2
with_terminal() do
	@code_native debuginfo=:none k(Int64, 10)
end

# ╔═╡ 0850e134-fdf8-416e-959e-5b34eb6b646f
function m(::Type{T}, N) where T
    acc = zero(T)
    for i in 1:N
        acc += one(T)
        escape(acc)
    end
    return acc
end

# ╔═╡ 39e20d3d-2fda-4646-89ad-4ef0a05afc0a
with_terminal() do
	@code_llvm debuginfo=:none m(Int64, 10)
end

# ╔═╡ d603d95f-6dee-40a4-b7f3-8c71e23f0bb3
result2 = @benchmark m(Int64, $K)

# ╔═╡ 5f650ae1-ff55-44a3-8bc7-262f972eb20e
@benchmark m(Int64, $(K*10))

# ╔═╡ 23355993-e6d3-4fb4-9748-c306e1137aa8
begin
	t2 = time(minimum(result2)) * u"ns" # in ns
	pFreq2 = round(typeof(1u"MHz"), K/t2)
end;

# ╔═╡ 75d3553e-294d-4944-b027-fd0ed3cc73e0
md"""
Frequency estimation: $pFreq2 ~ $(round(typeof(1u"GHz"), pFreq2))

Note: Benchmarking is hard, careful evalutaion of what you are trying to benchmark.

- If we were just interesting in how fast f(N) was we would have been fine with our first measurement
- But we were interested in the speed of addition as a proxy of perfromance
- Integer math on a computer is associative, Floating-Point math is not.
"""

# ╔═╡ 1831a121-b647-4dcf-a015-09d0e5d48058
md"""
## Revisiting `h`

```julia
function h(N)
    acc = 0.0
    for i in 1:N
        acc += 1.0
    end
    acc
end
```
"""

# ╔═╡ 8b4e76bc-4364-498d-872b-9001d4ea006d
@benchmark h($K)

# ╔═╡ cbb18e79-30c2-4a3a-9d6e-0fa7d2a4d691
function l(N)
    acc = 0.0
    @simd for i in 1:N
        acc += 1.0
    end
    acc
end

# ╔═╡ 3c85f0ac-f913-43cc-853e-1eb2773ba767
@benchmark l($(K))

# ╔═╡ e0950c9d-f293-4daa-818b-684a56f39179
md"""
## Performance annotiations in Julia


- https://docs.julialang.org/en/v1/manual/performance-tips/
- Julia does bounds checking by default ones(10)[11] is an error
- `@inbounds` Turns of bounds-checking locally
- `@fastmath` Turns of strict IEEE749 locally -- be very careful this might **not do** what you want
- `@simd` and `@simd ivdep` stronger gurantuees to encourage LLVM to use SIMD operations
"""

# ╔═╡ 976069a2-5a69-4074-a829-9313e8fa5f41
with_terminal() do
	@code_llvm debuginfo=:none l(10)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkPlots = "ab8c0f59-4072-4e0d-8f91-a91e1495eb26"
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
LLVM = "929cbde3-209d-540e-8aea-75f648917ca0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[compat]
BenchmarkPlots = "~0.1.0"
BenchmarkTools = "~1.5.0"
LLVM = "~9.1.3"
PlutoUI = "~0.7.49"
Unitful = "~1.12.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.1"
manifest_format = "2.0"
project_hash = "3c4c5f3cd2cd5e31f99c461c3da570589fe90524"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BenchmarkPlots]]
deps = ["BenchmarkTools", "RecipesBase"]
git-tree-sha1 = "1cf60c704455accbfab5d55da8f5d3475a37f20e"
uuid = "ab8c0f59-4072-4e0d-8f91-a91e1495eb26"
version = "0.1.0"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "f1dff6729bc61f4d49e140da1af55dcd1ac97b2f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.5.0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

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

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "be3dc50a92e5a386872a493a10050136d4703f9b"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.6.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Unicode"]
git-tree-sha1 = "d422dfd9707bec6617335dc2ea3c5172a87d5908"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "9.1.3"

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

    [deps.LLVM.weakdeps]
    BFloat16s = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "05a8bd5a42309a9ec82f700876903abce1017dd3"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.34+0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

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
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

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
git-tree-sha1 = "eba4810d5e6a01f612b948c9fa94f905b49087b0"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.60"

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

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Tricks]]
git-tree-sha1 = "7822b97e99a1672bfb1b49b668a6d46d58d8cbcb"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.9"

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

[[deps.Unitful]]
deps = ["ConstructionBase", "Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "bb37ed24f338bc59b83e3fc9f32dd388e5396c53"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.12.4"

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
# ╟─f133db40-461f-4a09-af14-c5d99ef6ce04
# ╠═2ae8a548-76e9-11ed-2045-d960f32fa458
# ╟─1fe94954-965b-43be-ad90-3711f07bc25d
# ╟─aae1c36f-681d-4a91-b46f-294411643b92
# ╠═7b3f894d-bca9-448a-8cbc-15799e239a9a
# ╠═fb5c3d82-7003-4989-b1d2-ddd8c05d75eb
# ╠═1843304e-ad7e-4820-a88f-bd8fbedebae6
# ╠═6828ebc2-e8e1-4bfb-9961-52c9ab37e8a4
# ╟─0886d018-b4aa-4ec6-a10c-15248ab0f108
# ╠═294f27e9-f7a5-4c08-be98-6b49cd87377a
# ╟─97bf6798-6200-4ddc-9c97-fd695180bfe4
# ╠═c2ce0d4b-3dfe-4f50-a9f3-753178d4f7d5
# ╠═e4d6cdac-9e4f-45be-b132-38a2d20b492b
# ╠═11dcf6cb-c7a4-4bc7-ac51-6e0f17a99ee0
# ╟─3c5dcf61-7f18-41a7-8a24-56f515cbdeb2
# ╟─6bbbed58-75ed-4dd1-995e-d1a212e25425
# ╠═e9cb13b8-be41-45ac-a377-beb35a237cb7
# ╟─69875711-258f-4fae-b0b4-4ea3ea482b4a
# ╠═345be244-db9d-4b97-ac7d-18e108e5dc97
# ╠═04b5bbf5-2a5e-4952-8db1-ec03a2714e83
# ╠═a877d8dc-5c9d-41c2-88c7-775adfb63c2f
# ╠═f7d79610-06f8-4864-9a6e-44f88b8caae6
# ╠═4c8fea62-0607-4af0-a34e-7198a0cdb9cd
# ╠═4b8d16ed-bc5b-4bee-b0e0-183de945175a
# ╟─fb4e75e7-cce2-4b7d-aaf0-527bf01c3866
# ╟─14ad9189-88d0-447a-9b5a-df5626795f50
# ╟─a975d900-a348-43fe-b635-71a3785c7731
# ╟─d85de968-78f4-40d0-a90f-085f2b725a08
# ╟─55d7238f-89ac-4ae2-8ed3-79093cb279cc
# ╟─135431d7-cd46-44bc-87ce-9eb810b9a7a5
# ╠═1d55abb6-eb4d-4694-92b7-c6613522d07b
# ╠═1aa09f44-bf05-4d75-8aac-39642927cbb1
# ╠═e2346440-a593-4954-9f73-8999ad33a2d2
# ╠═0850e134-fdf8-416e-959e-5b34eb6b646f
# ╠═39e20d3d-2fda-4646-89ad-4ef0a05afc0a
# ╠═d603d95f-6dee-40a4-b7f3-8c71e23f0bb3
# ╠═5f650ae1-ff55-44a3-8bc7-262f972eb20e
# ╠═23355993-e6d3-4fb4-9748-c306e1137aa8
# ╟─75d3553e-294d-4944-b027-fd0ed3cc73e0
# ╟─1831a121-b647-4dcf-a015-09d0e5d48058
# ╠═8b4e76bc-4364-498d-872b-9001d4ea006d
# ╠═cbb18e79-30c2-4a3a-9d6e-0fa7d2a4d691
# ╠═3c85f0ac-f913-43cc-853e-1eb2773ba767
# ╟─e0950c9d-f293-4daa-818b-684a56f39179
# ╠═976069a2-5a69-4074-a829-9313e8fa5f41
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002

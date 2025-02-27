### A Pluto.jl notebook ###
# v0.20.4

#> [frontmatter]
#> title = "Exercise: Simple trees"
#> license = "MIT"
#> 
#>     [[frontmatter.author]]
#>     name = "Valentin Churavy"
#>     url = "https://vchuravy.dev"

using Markdown
using InteractiveUtils

# ╔═╡ 00422cac-f45e-11ef-1db0-bf6465850a6b
md"""
# Simple trees

A simple binary tree in C++ would look like this:

```C++
struct node {
    double weight;
    
    node *left;
    node *right;
    node *parent;
};
```
"""

# ╔═╡ d2166802-a01a-4239-88f8-2fbcbaebb462
md"""
## Task 1

Implement a binary tree in Julia
"""

# ╔═╡ 5656af52-3e6d-4411-8a33-1ee54ac00bae
md"""
## Task 2

Visualize your trees; as an example see https://github.com/Keno/AbstractTrees.jl `print_tree`
"""

# ╔═╡ 1af0a1ca-790f-47b8-8fed-22fa5d808979
md"""
## Task 3
Implement both **Depth-First Search** and **Breadth-First Search**
"""

# ╔═╡ dc12427a-b3b4-41a0-8eb9-082c5b98a856
md"""
## Task 4

Benchmark and profile your implementation, inspect the result of type-inference.
"""

# ╔═╡ Cell order:
# ╟─00422cac-f45e-11ef-1db0-bf6465850a6b
# ╠═d2166802-a01a-4239-88f8-2fbcbaebb462
# ╟─5656af52-3e6d-4411-8a33-1ee54ac00bae
# ╟─1af0a1ca-790f-47b8-8fed-22fa5d808979
# ╟─dc12427a-b3b4-41a0-8eb9-082c5b98a856

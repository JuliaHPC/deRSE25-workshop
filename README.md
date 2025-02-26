# deRSE'25 Julia tutorial 

To find prerendered notebooks go to https://juliahpc.github.io/deRSE25-workshop/.

## Requirements
For the tutorials we will use [Pluto.jl](https://plutojl.org/) notebooks in [Julia](https://julialang.org/).

### Installing Julia
Download and install Julia for your platform from [here](https://julialang.org/downloads/).

### Installing Pluto.jl
Start Julia 
```shell
$ julia
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.10.4 (2024-06-04)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

```
and run the following commands to install Pluto.jl:
```julia
] add Pluto
```
Start Pluto.jl
```julia
import Pluto
Pluto.run()
```
Load the notebooks from this repository. Alternatively, you can view a prerendered static html version of the notebooks (see above).


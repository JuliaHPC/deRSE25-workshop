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

<details>
<summary>
    Other options for running the notebooks without installing Julia on your machine
</summary>

## Using a Docker container

We also provide a [Docker image](https://github.com/orgs/JuliaHPC/packages/container/package/derse25-workshop) (built for `linux/amd64` and `linux/arm64`) for running the notebook, which you can pull with

```sh
docker pull ghcr.io/juliahpc/derse25-workshop:main
```

Pluto can then be run on MacOS or Linux with

```sh
docker run -p 1234:1234 -ti ghcr.io/juliahpc/derse25-workshop:main julia -e 'using Pluto; Pluto.run(; host="0.0.0.0", port=1234)'
```

or if using PowerShell on Windows with

```PowerShell
docker run -p 1234:1234 -ti ghcr.io/juliahpc/derse25-workshop:main julia -e 'using Pluto; Pluto.run(; host=""""0.0.0.0"""", port=1234)'
```

This will launch Pluto within the container, and if successful you should see a message similar to

```
[ Info: Loading...
┌ Info:
└ Go to http://0.0.0.0:1234/?secret=hgY7as1X in your browser to start writing ~ have fun!
```

where `hgY7as1X` in the URL will be replaced with another random alphanumeric string.
The Pluto notebook environment is accessed as a web app, so you should open a browser window and navigate to the URL indicated in the message to open the Pluto interface.
If you get `Unable to connect` message or similar when trying to open the URL, you may need to replace the `0.0.0.0` component with `localhost`, so for the example above you would navigate to `http://localhost:1234/?secret=hgY7as1X`.

Once you have the Pluto interface open in your browser, you can load the notebooks saved under `/root`. To open a notebook, find the `Open a notebook` section in the Pluto interface, click on the `Enter path or URL...` field and select `root/` and then choose to the desidered notebook from the drop-down file navigator and finally click the `Open` button to open it.

#### GitHub Codespaces

> [!NOTE]
> GitHub Codespaces is a convenient environment for running notebooks on the web for free, but the resources on the free plan are limited, and parallel scaling efficiency may be be poor in some cases.

You can also take advantage of the ability of [GitHub Codespaces](https://github.com/features/codespaces) to run custom web apps.
Go go the [Codespaces page of this repository](https://github.com/JuliaHPC/deRSE25-workshop/codespaces), click on the green button on the top right "Create codespace on main" and wait a few seconds for the codespace to start.
In the bottom panel, go to the "Terminal" tab (other tabs should be "Problems", "Output", "Debug console", "Ports") and when you see the message (this can take a few seconds to appear after the codespace started, hold on)

```
[ Info: Loading...
┌ Info:
└ Go to http://localhost:1234/ in your browser to start writing ~ have fun!
```

go to the "Ports" tab, right click on the "Pluto server (1234)" port and click on "Open in browser" (alternatively, click on the globe-shaped button under the "Forwarded Addresses" column).
This will open the Pluto landing page in a new tab in your browser and from there you can open the desired notebooks.

If you want to make your app accessible to others (please remember to make sure there's no sensitive or private data in it!), navigate to the "Ports" tab, right click on the "Pluto server (1234)" port and then "Port visibility" -> "Public".

The `.devcontainer` used here has been adapted from the [Julia workshop for the UCL Festival of Digital Research & Scholarship 2024](https://github.com/UCL-ARC/julia-workshop), in turn based on the [Zero-setup R workshops with GitHub Codespaces](https://github.com/revodavid/devcontainers-rstudio) repository presented at [rstudio::conf 2022](https://rstudioconf2022.sched.com/event/11iag/zero-setup-r-workshops-with-github-codespaces).

</details>

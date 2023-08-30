using CognitiveModel
using Documenter

DocMeta.setdocmeta!(CognitiveModel, :DocTestSetup, :(using CognitiveModel); recursive=true)

makedocs(;
    modules=[CognitiveModel],
    authors="Pengfei Song",
    repo="https://github.com/Song921012/CognitiveModel.jl/blob/{commit}{path}#{line}",
    sitename="CognitiveModel.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Song921012.github.io/CognitiveModel.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Song921012/CognitiveModel.jl",
    devbranch="main",
)

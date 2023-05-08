# Loading Packages and setup random seeds
using Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, Random, Plots
using DataFrames
using CSV
rng = Random.default_rng()

# load training dta

data = DataFrame(CSV.File("./output/datasmoothing.csv"))

trainingdata = Array(data[1:200, 4:7])'

# set up neural differential equation models
u0 = Array(data[1, 4:7])
datasize = 200
tspan = (0.0f0, 200.0f0)
tsteps = range(tspan[1], tspan[2], length=datasize)
dudt2 = Lux.Chain(Lux.Dense(4, 32, relu), Lux.Dense(32, 32, tanh), Lux.Dense(32, 4))
p, st = Lux.setup(rng, dudt2)
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat=tsteps)
prob_node = ODEProblem((u, p, t) -> dudt2(u, p, st)[1], u0, tspan, Lux.ComponentArray(p))


# simulate the neural differential equation models
function predict_neuralode(p)
    Array(prob_neuralode(u0, p, st)[1])
end

predict_neuralode(p)
size(predict_neuralode(p)) == size(trainingdata)
# loss function and callbacks

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, trainingdata .- pred)
    return loss, pred
end

loss_neuralode(p)

callback = function (p, l, pred; doplot=false)
    println(l)
    # plot current prediction against data
    if doplot
        plt = scatter(tsteps, trainingdata[1, :], label="data")
        scatter!(plt, tsteps, pred[1, :], label="prediction")
        display(plot(plt))
    end
    return false
end

pinit = Lux.ComponentArray(p)
callback(pinit, loss_neuralode(pinit)...; doplot=true)

# use Optimization.jl to solve the problem

adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

result_neuralode = Optimization.solve(optprob,
    ADAM(0.05),
    callback=callback,
    maxiters=300)

optprob2 = remake(optprob, u0=result_neuralode.u)

result_neuralode2 = Optimization.solve(optprob2,
    Optim.BFGS(initial_stepnorm=0.01),
    callback=callback,
    allow_f_increases=false)

optprob2 = remake(optprob, u0=result_neuralode2.u)

result_neuralode2 = Optimization.solve(optprob2,
    ADAM(0.001),
    maxiters=300,
    callback=callback,
    allow_f_increases=false)
optprob2 = remake(optprob, u0=result_neuralode2.u)

result_neuralode2 = Optimization.solve(optprob2,
    Optim.LBFGS(),
    callback=callback,
    allow_f_increases=false)

pfinal = result_neuralode2.u

callback(pfinal, loss_neuralode(pfinal)...; doplot=true)

# Multiple shooting
using OptimizationPolyalgorithms
using DiffEqFlux: group_ranges
group_size = 3
continuity_term = 200

function loss_function(data, pred)
    return sum(abs2, data - pred)
end

function loss_multiple_shooting(p)
    return multiple_shoot(p, trainingdata, tsteps, prob_node, loss_function, Tsit5(),
        group_size; continuity_term)
end
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_multiple_shooting(x), adtype)
optprob = Optimization.OptimizationProblem(optf, Lux.ComponentArray(pinit))
res_ms = Optimization.solve(optprob, PolyOpt(),
    callback=callback,
    allow_f_increases=false)

optprob2 = remake(optprob, u0=res_ms.u)
res_ms = Optimization.solve(optprob2, PolyOpt(),
    callback=callback,
    allow_f_increases=false)

pfinal = res_ms.u

callback(pfinal, loss_neuralode(pfinal)...; doplot=true)
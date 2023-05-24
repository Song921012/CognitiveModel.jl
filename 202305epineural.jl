##
# Loading Packages and setup random seeds
using Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, Random, Plots
using DataFrames
using CSV
using ComponentArrays
using OptimizationOptimisers
rng = Random.default_rng()
Random.seed!(rng, 10)

# load training dta

data = DataFrame(CSV.File("./output/datasmoothing.csv"))
choosentime = range(87, 150)
choosencolumn = [3, 5] # vaccined individuals, cases
trainingdata = Array(data[choosentime, choosencolumn])'
datascale = Array(data[choosentime, 2:end])'

# set up neural differential equation models
N = 38250000.0f0
σ1 = 0.19
γ1 = 0.1
hi0 = Float32(trainingdata[1, 1])
hv0 = Float32(trainingdata[2, 1])
v0 = Float32(trainingdata[2, 1])
e0 = Float32(datascale[1, 1] / σ1)
i0 = Float32(datascale[1, 1] / γ1)
u0 = [N, v0, e0, i0, hi0, hv0]
datasize = 64
tspan = (0.0f0, 64.0f0)
tsteps = range(tspan[1], tspan[2], length=datasize)
ann = Lux.Chain(Lux.Dense(1, 32, relu), Lux.Dense(32, 32, tanh), Lux.Dense(32, 2))
# relu, tanh, swish
#ann = Lux.Chain(Lux.Dense(1, 64, relu), Lux.Dense(64, 32, relu), Lux.Dense(32, 2))
p, st = Lux.setup(rng, ann)
ann([0.1], p, st)[1]
function SVEIR_nn(du, u, p, t)
    S, V, E, I, HI, HV = u
    N = 38250000.0f0
    ϵ = 0.8f0
    σ1 = 0.19f0
    γ1 = 0.1f0
    du[1] = -min(5.0f0, abs(ann([t], p, st)[1][1])) * I * S / N - abs(ann([t], p, st)[1][2]) * S
    du[2] = abs(ann([t], p, st)[1][2]) * S - (1.0f0 - ϵ) * min(1, abs(ann([t], p, st)[1][1])) * I * V / N
    du[3] = min(5.0f0, abs(ann([t], p, st)[1][1])) * I * S / N + (1.0f0 - ϵ) * min(5.0f0, abs(ann([t], p, st)[1][1])) * I * V / N - σ1 * E
    du[4] = σ1 * E - γ1 * I
    du[5] = min(5.0f0, abs(ann([t], p, st)[1][1])) * I * S / N + (1.0f0 - ϵ) * min(5.0f0, abs(ann([t], p, st)[1][1])) * I * V / N
    du[6] = abs(ann([t], p, st)[1][2]) * S
end
prob_neuralode = ODEProblem(SVEIR_nn, u0, tspan, ComponentArray(p))


# simulate the neural differential equation models
function predict_neuralode(θ)
    #Array(prob_neuralode(u0, p, st)[1])
    prob = remake(prob_neuralode, p=θ)
    Array(solve(prob, Tsit5(), saveat=tsteps))
end

predict_neuralode(p)[5:6, :]
size(predict_neuralode(p)[5:6, :]) == size(trainingdata)
# loss function and callbacks

function loss_neuralode(p)
    pred = predict_neuralode(p)[5:6, :]
    loss = sum(abs2, log.(trainingdata) .- log.(pred))
    return loss, pred
end

loss_neuralode(p)


callback = function (p, l, pred; doplot=false)
    println(l)
    # plot current prediction against data
    if doplot
        plt = scatter(tsteps, trainingdata[1, :], label="Accumulated cases")
        scatter!(tsteps, trainingdata[2, :], label="Accumulated vaccinated individuals")
        plot!(plt, tsteps, pred[1, :], label="Predicted accumulated cases")
        plot!(plt, tsteps, pred[2, :], label="Predicted accumulated vaccinated individuals")
        display(plot(plt))
    end
    return false
end

pinit = ComponentArray(p)
callback(pinit, loss_neuralode(pinit)...; doplot=true)

# use Optimization.jl to solve the problem
##
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

result_neuralode = Optimization.solve(optprob,
    OptimizationOptimisers.ADAM(0.05),
    callback=callback,
    maxiters=100)

optprob2 = remake(optprob, u0=result_neuralode.u)

result_neuralode2 = Optimization.solve(optprob2,
    Optim.BFGS(initial_stepnorm=0.01),
    callback=callback,
    allow_f_increases=false)



optprob2 = remake(optprob, u0=result_neuralode2.u)

result_neuralode2 = Optimization.solve(optprob2,
    Optimisers.ADAM(0.001),
    maxiters=300,
    callback=callback,
    allow_f_increases=false)
optprob2 = remake(optprob, u0=result_neuralode2.u)

result_neuralode2 = Optimization.solve(optprob2,
    Optim.LBFGS(),
    callback=callback,
    allow_f_increases=false)
optprob2 = remake(optprob, u0=result_neuralode2.u)

result_neuralode2 = Optimization.solve(optprob2,
    Optimisers.ADAM(0.001),
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

##
# beta(t), nu(t) function

β(t) = min(1, abs(ann([t], pfinal, st)[1][1]))
ν(t) = abs(ann([t], pfinal, st)[1][2])

plot(tsteps, β.(tsteps))
plot(tsteps, ν.(tsteps))
##
# Save neural network architechtures and 
using BSON: @save
@save "./output/annepi.bson" ann
psave=collect(pfinal)
@save "./output/annepipfinal.bson" psave
pred = predict_neuralode(pfinal)[5:6, :]
plt = scatter(tsteps, trainingdata[1, :], label="Accumulated cases")
plot!(plt, tsteps, pred[1, :], label="Predicted accumulated cases")
display(plot(plt))
savefig("./output/annepicase.png")
plt = scatter(tsteps, trainingdata[2, :], label="Accumulated vaccinated individuals")
#scatter!(tsteps, trainingdata[3, :], label="Minter data")
#scatter!(tsteps, trainingdata[4, :], label="Cinter data")

plot!(plt, tsteps, pred[2, :], label="Predicted accumulated vaccinated individuals")
#plot!(plt, tsteps, pred[3, :], label="Minter prediction")
#plot!(plt, tsteps, pred[4, :], label="Cinter prediction")
display(plot(plt))
savefig("./output/annepivac.png")

neuralepi = DataFrame()
neuralepi[!, "date"] = data[choosentime, 1]
neuralepi[!, "Case"] = trainingdata[1, :]
neuralepi[!, "Vaccine"] = trainingdata[2, :]
neuralepi[!, "PredCase"] = pred[1, :]
neuralepi[!, "PredVaccine"] = pred[2, :]
neuralepi[!, "beta"] = β.(tsteps)
neuralepi[!, "nu"] = ν.(tsteps)
CSV.write("./output/neuralepi.csv", neuralepi)

##
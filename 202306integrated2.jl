##
# Loading Packages and setup random seeds
using Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, Random, Plots
using Flux
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

# initial value of M1,C1,M2,C2
intercolumn = [10, 11] # M, C score vaccine
interdata = Array(data[choosentime, intercolumn])'
m10, c10 = Float32.(Array(interdata[:, 1]))

vaccolumn = [12, 13] # M, C score vaccine
vacdata = Array(data[choosentime, vaccolumn])'
m20, c20 = Float32.(Array(vacdata[:, 1]))


#
datasize = 64
tspan = (0.0f0, Float32(datasize))
tsteps = range(tspan[1], tspan[2], length=datasize)

# reload M1,C1,M2,C2
## Obtain M1(t), C1(t) function
using BSON: @load
@load "./output/anninter.bson" dudt2
anninter = dudt2
@load "./output/anninterpara.bson" psave
pintersave = psave
pinter, stinter = Lux.setup(rng, anninter)
pinterinit = ComponentArray(pinter)
pinterfinal = ComponentArray(pintersave, getaxes(pinterinit))
u0inter = [m10, c10]
interode = ODEProblem((u, p, t) -> anninter(u, p, stinter)[1], u0inter, tspan, pinterfinal)
solinter = solve(interode, Tsit5())
display(plot(solinter))
M1(t) = solinter(t)[1]
C1(t) = solinter(t)[2]
## Obtain M2(t), C2(t) function
@load "./output/annvac.bson" dudt2
annvac = dudt2
@load "./output/annvacpara.bson" psave
pvacsave = psave
pvac, stvac = Lux.setup(rng, annvac)
pvacinit = ComponentArray(pvac)
pvacfinal = ComponentArray(pvacsave, getaxes(pvacinit))
u0vac = [m20, c20]
vacode = ODEProblem((u, p, t) -> annvac(u, p, stvac)[1], u0vac, tspan, pvacfinal)
solvac = solve(vacode, Tsit5())
display(plot(solvac))
M2(t) = solvac(t)[1]
C2(t) = solvac(t)[2]

##
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

ann1 = Flux.Chain(Flux.Dense(5, 64, relu), Flux.Dense(64, 1))
ann2 = Flux.Chain(Flux.Dense(5, 64, relu), Flux.Dense(64, 1))
p1, re1 = Flux.destructure(ann1)
p2, re2 = Flux.destructure(ann2)
p1 = Float32.(p1)
p2 = Float32.(p2)
re1(p1)([1.0f0, m10, c10, m20, c20])
re2(p2)([1.0f0, m10, c10, m20, c20])

function SVEIR_nn(du, u, p, t)
    S, V, E, I, HI, HV = u
    N = 38250000.0f0
    ϵ = 0.8f0
    σ1 = 0.19f0
    γ1 = 0.1f0
    β = min(5.0f0, abs(re1(p[1:length(p1)])([t, M1(t), C1(t)])[1]))
    ν = abs(re2(p[(length(p1)+1):end])([t, M2(t), C2(t)])[1])
    du[1] = -β * I * S / N - ν * S
    du[2] = ν * S - (1.0f0 - ϵ) * β * I * V / N
    du[3] = β * I * S / N + (1.0f0 - ϵ) * β * I * V / N - σ1 * E
    du[4] = σ1 * E - γ1 * I
    du[5] = β * I * S / N + (1.0f0 - ϵ) * β * I * V / N
    du[6] = ν * S
end
prob_neuralode = ODEProblem(SVEIR_nn, u0, tspan, [p1; p2])


##
# simulate the neural differential equation models
function predict_neuralode(θ)
    #Array(prob_neuralode(u0, p, st)[1])
    prob = remake(prob_neuralode, p=θ)
    Array(solve(prob, Vern7(), saveat=tsteps))
end
p = [p1; p2]
predict_neuralode(p)[5:6, :]
size(predict_neuralode(p)[5:6, :]) == size(trainingdata)

sol = predict_neuralode(p)
pltinter = scatter(tsteps, interdata[1, :], label="Minter data")
scatter!(tsteps, interdata[2, :], label="Cinter data")
plot!(pltinter, tsteps, M1.(tsteps), label="Minter prediction")
plot!(pltinter, tsteps, C1.(tsteps), label="Cinter prediction")
display(plot(pltinter))
pltvac = scatter(tsteps, vacdata[1, :], label="Mvac data")
scatter!(tsteps, vacdata[2, :], label="Cvac data")
plot!(pltvac, tsteps, M2.(tsteps), label="Mvac prediction")
plot!(pltvac, tsteps, C2.(tsteps), label="Cvac prediction")
display(plot(pltvac))

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

pinit = [p1; p2]
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
β(t) = min(5.0f0, abs(re1(pfinal[1:length(p1)])([t])[1]))
ν(t) = abs(re2(pfinal[(length(p1)+1):end])([t])[1])



plot(tsteps, β.(tsteps))
plot(tsteps, ν.(tsteps))
##
# Save neural network architechtures and 
using BSON: @save
@save "./output/ann1epi.bson" ann1
@save "./output/ann2epi.bson" ann2
psave = pfinal
@save "./output/ann12epipfinal.bson" psave
pred = predict_neuralode(pfinal)[5:6, :]
plt = scatter(tsteps, trainingdata[1, :], label="Accumulated cases")
plot!(plt, tsteps, pred[1, :], label="Predicted accumulated cases")
display(plot(plt))
savefig("./output/ann12epicase.png")
plt = scatter(tsteps, trainingdata[2, :], label="Accumulated vaccinated individuals")
#scatter!(tsteps, trainingdata[3, :], label="Minter data")
#scatter!(tsteps, trainingdata[4, :], label="Cinter data")

plot!(plt, tsteps, pred[2, :], label="Predicted accumulated vaccinated individuals")
#plot!(plt, tsteps, pred[3, :], label="Minter prediction")
#plot!(plt, tsteps, pred[4, :], label="Cinter prediction")
display(plot(plt))
savefig("./output/ann12epivac.png")

neuralepi = DataFrame()
neuralepi[!, "date"] = data[choosentime, 1]
neuralepi[!, "Case"] = trainingdata[1, :]
neuralepi[!, "Vaccine"] = trainingdata[2, :]
neuralepi[!, "PredCase"] = pred[1, :]
neuralepi[!, "PredVaccine"] = pred[2, :]
neuralepi[!, "beta"] = β.(tsteps)
neuralepi[!, "nu"] = ν.(tsteps)
CSV.write("./output/neuralepi12.csv", neuralepi)

##
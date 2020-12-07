using Revise
using DrWatson
@quickactivate "Twisted Particle Filter"

using CSV
using DataFrames
using Gadfly

df = CSV.read(datadir("tpf_likelihood.csv"), DataFrame)

M = unique(df[:Particles])
##
set_default_plot_size(21cm, 8cm)
plot(
    df,
    x=:Particles,
    y=:likelihood,
    color=:Method,
    Scale.x_discrete(levels=M, labels=string),
    Geom.boxplot,
    Theme(boxplot_spacing=0.2*Gadfly.cx),
    Guide.colorkey(title="Method", pos=[0.0*Gadfly.w,-0.3*Gadfly.h])
    )

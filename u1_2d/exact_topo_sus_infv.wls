#!/usr/bin/env wolframscript
(*
inp[n_,x_]:=-1/(2Pi)^2Im[NIntegrate[phi Exp[I n phi]Exp[x Cos[phi]],{phi,-Pi,Pi}]]
*)
inpp[n_,x_]:=-1/(2Pi)^3Re[NIntegrate[phi^2Exp[I n phi]Exp[x Cos[phi]],{phi,-Pi,Pi}]]
(*
wn1list[nmax_,beta_,p_]:=Table[BesselI[n,beta]^p,{n,-nmax,nmax}]
wn1[nmax_,beta_,p_]:=With[{w=wn1list[nmax,beta,p]},w/Total[w]]
chiT1n[nmax_,beta_,p_]:=Table[-inpp[n,beta]/BesselI[n,beta]-(p-1)(inp[n,beta]/BesselI[n,beta])^2,{n,-nmax,nmax}]
chiT1List[nmax_,beta_,p_]:=wn1[nmax,beta,p]chiT1n[nmax,beta,p]
chiT1[nmax_,beta_,p_]:=Total[chiT1List[nmax,beta,p]]
Qsquared1[nmax_,beta_,p_]:=p chiT1[nmax,beta,p]
*)
chiT1Inf[beta_]:=-inpp[0,beta]/BesselI[0,beta]
betas=Sort@Flatten@Table[b(v/64)^2,{b,1,13},{v,{64,66,68,70,72,74,76,82,88}}]
chis=chiT1Inf/@betas
str:=ToString@NumberForm[#,16]&
MapThread[Print[#1,"  ",#2]&]@{str/@N[betas]//StringPadRight, str/@chis}

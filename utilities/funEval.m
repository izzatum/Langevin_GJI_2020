function [f,g,H] = funEval(x,fh,dfh,ddfh)

f = fh(x);
g = dfh(x);
H = ddfh(x);
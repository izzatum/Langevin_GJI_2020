function val = cutPrecision(x)

    x(abs(x) < 1e-16) = 0;
    val = x;

end
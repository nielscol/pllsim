from libpll.tools import binary_lf_coefs, fixed_point

int_bits = 5
frac_bits = 8
# lf_params = dict(b0=3.1975968494E+01, b1=-2.8851627388E+01)
lf_params = dict(b0=24.746023869534508, b1=-22.328113694243058)

binary_lf_coefs(lf_params, int_bits, frac_bits)

print("\nQuantized Equivalent")
for k,v in lf_params.items():
    x = fixed_point(v, int_bits, frac_bits)
    print("%s\t->\t%.10E\tError = %.10E"%(k, x, x-v))

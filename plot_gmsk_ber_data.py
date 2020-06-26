import numpy as np
import matplotlib.pyplot as plt
from libpll.plot import razavify
import json

with open("./gmsk_ber_data.json", "r") as f:
    data = json.load(f)

plt.semilogy(data["cnr"], data["ber"])
plt.ylim(8e-5, 1)
plt.xlim(0,8)
plt.grid()
plt.ylabel("BER")
plt.xlabel("CNR [dB]")
plt.title("Simulated BER vs CNR")
razavify()
ticks = np.linspace(-4, 0, 5)
plt.yticks(10**ticks, ["10$^{%.0f}$"%x for x in ticks])
ticks = np.linspace(0, 8, 9)
plt.xticks(ticks, ["%.f"%x for x in ticks])
plt.show()


plt.title("PLL Start-up transient response")
plt.ylabel("Loop filter output")
plt.xlabel("Time [$\mu$s]")
plt.legend()
plt.axhline(9990, color="r")
plt.axhline(10010, color="r")
plot_td(main_pn_data["lf"], title="Loop Filter Output", tmax=50e-6)
plt.ylim((8800, 10300))
plt.plot((23e-6,23e-6),(8800, 9990), color="b", linestyle="--")
plt.text(25e-6, 10030, "Lock tolerance band", color="r", fontsize=12)
plt.text(23.5e-6, 9500, "Lock time", color="b", fontsize=12, rotation=90)
razavify(loc="lower left", bbox_to_anchor=[0,0])
plt.xticks([0, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5], ["0", "10", "20", "30", "40", "50"])

plt.tight_layout()
# plt.show()
plt.savefig("loop_filter_trans.pdf")


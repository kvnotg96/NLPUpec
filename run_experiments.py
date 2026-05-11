"""
Headless experimental driver extracted from neural_opt.py
Runs the 5 datasets, performs L-BFGS-B hyperparameter optimization
and generates all figures for Kevin's article.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import warnings, json, os
warnings.filterwarnings("ignore")

FIGDIR = "/home/claude/kevin/figures"
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size":   10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 130,
})

# =========================================================================
#  Datasets (synthetic 2-D, balanced binary)
# =========================================================================
def make_dataset(name="Spiral", n=400, seed=7):
    rng = np.random.RandomState(seed); h = n // 2
    if name == "Spiral":
        t = np.linspace(0, 4*np.pi, h); r = t/(4*np.pi)
        X0 = np.c_[r*np.cos(t),  r*np.sin(t)] + rng.randn(h,2)*.06
        X1 = np.c_[-r*np.cos(t),-r*np.sin(t)] + rng.randn(h,2)*.06
    elif name == "Moons":
        t0 = np.linspace(0, np.pi, h); t1 = np.linspace(np.pi, 2*np.pi, h)
        X0 = np.c_[np.cos(t0), np.sin(t0)] + rng.randn(h,2)*.09
        X1 = np.c_[1+np.cos(t1), .5+np.sin(t1)] + rng.randn(h,2)*.09
    elif name == "Circles":
        t = np.linspace(0, 2*np.pi, h)
        X0 = np.c_[.4*np.cos(t), .4*np.sin(t)] + rng.randn(h,2)*.05
        X1 = np.c_[np.cos(t),    np.sin(t)]    + rng.randn(h,2)*.05
    elif name == "XOR":
        q = h // 2
        X0 = np.vstack([rng.randn(q,2)*.4+[-.7,-.7], rng.randn(q,2)*.4+[ .7, .7]])
        X1 = np.vstack([rng.randn(q,2)*.4+[ .7,-.7], rng.randn(q,2)*.4+[-.7, .7]])
    else:  # Gaussians
        X0 = rng.randn(h,2)*.4 + [-.8, 0]
        X1 = rng.randn(h,2)*.4 + [ .8, 0]
    X = np.vstack([X0, X1]); X = (X - X.mean(0))/(X.std(0)+1e-8)
    y = np.vstack([np.zeros((h,1)), np.ones((h,1))])
    idx = rng.permutation(len(X))
    return X[idx], y[idx]

def act_fn(name, z, deriv=False):
    if name == "ReLU":
        if deriv: return (z > 0).astype(float)
        return np.maximum(0, z)
    if name == "Tanh":
        t = np.tanh(z); return 1 - t**2 if deriv else t
    if name == "ELU":
        a = 1.0
        if deriv: return np.where(z >= 0, 1.0, a*np.exp(np.clip(z,-30,0)))
        return np.where(z >= 0, z, a*(np.exp(np.clip(z,-30,0))-1))
    return np.maximum(0, z)

# =========================================================================
#  MLP with Adam (faithful to neural_opt.py)
# =========================================================================
class MLP:
    def __init__(self, n0=2, h1=32, h2=16, act="ReLU", lr=1e-3, lam=1e-4, seed=42):
        np.random.seed(seed)
        self.lr=lr; self.lam=lam; self.act=act
        def W(fi, fo): return np.random.randn(fi, fo)*np.sqrt(2.0/fi)
        self.W1=W(n0,h1); self.b1=np.zeros((1,h1))
        self.W2=W(h1,h2); self.b2=np.zeros((1,h2))
        self.W3=W(h2, 1); self.b3=np.zeros((1, 1))
        self._t=0
        self._m={k:np.zeros_like(v) for k,v in self._p()}
        self._v={k:np.zeros_like(v) for k,v in self._p()}
        self.loss_h=[]; self.acc_h=[]; self.norm_h=[]

    def _p(self):
        return [("W1",self.W1),("b1",self.b1),
                ("W2",self.W2),("b2",self.b2),
                ("W3",self.W3),("b3",self.b3)]

    @staticmethod
    def sig(z): return 1.0/(1.0+np.exp(-np.clip(z,-500,500)))

    def forward(self, X):
        self.X=X
        self.Z1=X@self.W1+self.b1;     self.A1=act_fn(self.act,self.Z1)
        self.Z2=self.A1@self.W2+self.b2; self.A2=act_fn(self.act,self.Z2)
        self.Z3=self.A2@self.W3+self.b3; self.A3=self.sig(self.Z3)
        return self.A3

    def loss_val(self, X, y):
        A3=self.forward(X); eps=1e-9
        bce=-np.mean(y*np.log(A3+eps)+(1-y)*np.log(1-A3+eps))
        reg=(self.lam/2)*(np.sum(self.W1**2)+np.sum(self.W2**2)+np.sum(self.W3**2))
        return bce + reg

    def backward(self, X, y):
        m=X.shape[0]; b1=0.9; b2=0.999; eps=1e-8; self._t+=1
        dZ3=(self.A3-y)/m
        dW3=self.A2.T@dZ3+self.lam*self.W3; db3=dZ3.sum(0,keepdims=True)
        dA2=dZ3@self.W3.T; dZ2=dA2*act_fn(self.act,self.Z2,True)
        dW2=self.A1.T@dZ2+self.lam*self.W2; db2=dZ2.sum(0,keepdims=True)
        dA1=dZ2@self.W2.T; dZ1=dA1*act_fn(self.act,self.Z1,True)
        dW1=X.T@dZ1+self.lam*self.W1;        db1_g=dZ1.sum(0,keepdims=True)
        g={"W1":dW1,"b1":db1_g,"W2":dW2,"b2":db2,"W3":dW3,"b3":db3}
        P={"W1":self.W1,"b1":self.b1,"W2":self.W2,"b2":self.b2,
           "W3":self.W3,"b3":self.b3}
        for k,gi in g.items():
            self._m[k]=b1*self._m[k]+(1-b1)*gi
            self._v[k]=b2*self._v[k]+(1-b2)*gi**2
            mh=self._m[k]/(1-b1**self._t)
            vh=self._v[k]/(1-b2**self._t)
            P[k] -= self.lr*mh/(np.sqrt(vh)+eps)
        self.W1=P["W1"]; self.b1=P["b1"]; self.W2=P["W2"]; self.b2=P["b2"]
        self.W3=P["W3"]; self.b3=P["b3"]

    def step(self, X, y): self.forward(X); self.backward(X,y)

    def record(self, X, y):
        l=self.loss_val(X,y)
        pred=(self.forward(X)>=.5).astype(int)
        acc=float(np.mean(pred==y))
        nrm=float(np.sqrt(sum(np.sum(w**2) for k,w in self._p() if k.startswith("W"))))
        self.loss_h.append(l); self.acc_h.append(acc); self.norm_h.append(nrm)
        return l, acc, nrm

    def train(self, X, y, epochs, patience=80, min_delta=1e-5, rec=5):
        best=float("inf"); wait=0; lr_dc=0
        for ep in range(epochs):
            self.step(X,y)
            if ep % rec == 0:
                l,acc,nrm = self.record(X,y)
                if l < best - min_delta: best=l; wait=0
                else: wait+=1
                if wait >= patience:
                    if lr_dc < 2: self.lr *= 0.3; lr_dc+=1; wait=0
                    else: break
        return ep

# =========================================================================
#  Hyperparameter L-BFGS-B optimizer
#   min  f(lam, lr) = BCE(W*(lam,lr)) + gamma * ||W*(lam,lr)||^2
#   s.t. lam in [1e-6, 0.1],  lr in [1e-4, 0.05]
# =========================================================================
class MemOpt:
    def __init__(self, X, y, h1=48, h2=24, act="ReLU", gamma=0.05, pretrain=200):
        self.X=X; self.y=y; self.h1=h1; self.h2=h2; self.act=act
        self.gamma=gamma; self.pretrain=pretrain
        self.iters=[]; self.result=None

    def _obj(self, p):
        lam=float(np.clip(p[0],1e-6,0.1))
        lr =float(np.clip(p[1],1e-4,0.05))
        net=MLP(self.X.shape[1], self.h1, self.h2, self.act, lr, lam, seed=0)
        for _ in range(self.pretrain): net.step(self.X, self.y)
        bce=net.loss_val(self.X,self.y)
        nrm2=sum(np.sum(w**2) for k,w in net._p() if k.startswith("W"))
        f=bce + self.gamma*nrm2
        self.iters.append({"lam":lam,"lr":lr,"f":float(f),
                           "bce":float(bce),"norm2":float(nrm2)})
        return f

    def run(self, x0=(1e-4,1e-3),
            bounds=((1e-6,0.1),(1e-4,0.05))):
        self.iters=[]
        self.result = minimize(self._obj, np.array(x0), method="L-BFGS-B",
                               bounds=bounds, options={"maxiter":40,"ftol":1e-8})
        return float(self.result.x[0]), float(self.result.x[1])

# =========================================================================
#  Master driver: run all 5 datasets, save metrics, generate figures
# =========================================================================
DATASETS = ["Spiral","Moons","Circles","XOR","Gaussians"]
H1, H2, ACT, GAMMA = 48, 24, "ReLU", 0.002
EPOCHS = 1200

results = {}

for name in DATASETS:
    print(f"\n=== Dataset: {name} ===")
    X, y = make_dataset(name, n=400, seed=7)

    # Baseline network (default lam, lr)
    net_b = MLP(2, H1, H2, ACT, lr=3e-3, lam=1e-4, seed=42)
    net_b.train(X, y, EPOCHS, patience=80)
    lb, ab, nb = net_b.record(X, y)
    print(f"  Baseline:  loss={lb:.4f}  acc={ab*100:.1f}%  ||W||={nb:.3f}")

    # Hyperparameter optimization via L-BFGS-B
    opt = MemOpt(X, y, h1=H1, h2=H2, act=ACT, gamma=GAMMA, pretrain=200)
    lam_star, lr_star = opt.run()
    print(f"  L-BFGS-B:  lam*={lam_star:.5f}  lr*={lr_star:.5f}  evals={len(opt.iters)}")

    # Optimized network with the found (lam*, lr*)
    net_o = MLP(2, H1, H2, ACT, lr=lr_star, lam=lam_star, seed=42)
    net_o.train(X, y, EPOCHS, patience=80)
    lo, ao, no = net_o.record(X, y)
    print(f"  Optimized: loss={lo:.4f}  acc={ao*100:.1f}%  ||W||={no:.3f}")
    red = (nb - no)/nb*100 if nb > 1e-9 else 0.0
    print(f"  Norm reduction: {red:.1f}%  Acc delta: {(ao-ab)*100:+.1f}pp")

    results[name] = dict(
        X=X, y=y, net_b=net_b, net_o=net_o,
        lam_star=lam_star, lr_star=lr_star,
        iters=opt.iters,
        loss_b=lb, loss_o=lo, acc_b=ab, acc_o=ao,
        norm_b=nb, norm_o=no, norm_red=red,
    )

# Save a compact metric table
metrics = {
    n: dict(
        loss_b=results[n]["loss_b"], loss_o=results[n]["loss_o"],
        acc_b=results[n]["acc_b"],   acc_o=results[n]["acc_o"],
        norm_b=results[n]["norm_b"], norm_o=results[n]["norm_o"],
        norm_red=results[n]["norm_red"],
        lam_star=results[n]["lam_star"], lr_star=results[n]["lr_star"],
        n_evals=len(results[n]["iters"]),
        f0=results[n]["iters"][0]["f"], fT=results[n]["iters"][-1]["f"],
    ) for n in DATASETS
}
with open("/home/claude/kevin/metrics.json","w") as fp:
    json.dump(metrics, fp, indent=2)
print("\nSaved metrics.json")

# =========================================================================
#  FIGURE 1 — Five datasets (panel)
# =========================================================================
fig, axes = plt.subplots(1, 5, figsize=(13, 2.8))
for ax, name in zip(axes, DATASETS):
    X, y = results[name]["X"], results[name]["y"]
    m0 = (y.ravel()==0); m1 = (y.ravel()==1)
    ax.scatter(X[m0,0],X[m0,1], s=11, c="#1f77b4", alpha=.75, edgecolors="none", label="0")
    ax.scatter(X[m1,0],X[m1,1], s=11, c="#d62728", alpha=.75, edgecolors="none", label="1")
    ax.set_title(name); ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/figure1.pdf", bbox_inches="tight")
plt.close()
print("figure1.pdf: 5 datasets panel")

# =========================================================================
#  FIGURE 2 — Architecture diagram (drawn with matplotlib primitives)
# =========================================================================
fig, ax = plt.subplots(figsize=(9, 4.0))
ax.set_xlim(0,10); ax.set_ylim(0,6); ax.axis("off")

def layer(x, n, label, color, top=5.4, bot=0.6):
    ys = np.linspace(bot, top, n)
    for y in ys:
        c = plt.Circle((x,y), 0.18, fc=color, ec="black", lw=0.8, zorder=3)
        ax.add_patch(c)
    ax.text(x, top+0.35, label, ha="center", fontsize=10, weight="bold")
    return [(x,y) for y in ys]

xs   = [1.2, 3.6, 6.4, 8.8]
cols = ["#cccccc","#a5c8e1","#ffb997","#90ee90"]
labs = [r"Input ($n_0{=}2$)", r"Hidden 1 ($h_1$)", r"Hidden 2 ($h_2$)", "Output (1)"]
ns   = [2, 7, 5, 1]
nodes=[]
for x,n,l,c in zip(xs,ns,labs,cols):
    nodes.append(layer(x, n, l, c))

# Edges
for L,R in zip(nodes[:-1], nodes[1:]):
    for (x1,y1) in L:
        for (x2,y2) in R:
            ax.plot([x1+0.18,x2-0.18],[y1,y2], color="gray", lw=0.35, alpha=0.55, zorder=1)

# Activation labels
ax.text(2.4, -0.05, r"$z^{(1)}=XW^{(1)}+b^{(1)}$", fontsize=9, ha="center")
ax.text(2.4, -0.55, r"$a^{(1)}=\phi(z^{(1)})$",   fontsize=9, ha="center")
ax.text(5.0, -0.05, r"$z^{(2)}=a^{(1)}W^{(2)}+b^{(2)}$", fontsize=9, ha="center")
ax.text(5.0, -0.55, r"$a^{(2)}=\phi(z^{(2)})$",   fontsize=9, ha="center")
ax.text(7.6, -0.05, r"$z^{(3)}=a^{(2)}W^{(3)}+b^{(3)}$", fontsize=9, ha="center")
ax.text(7.6, -0.55, r"$\hat y=\sigma(z^{(3)})$",  fontsize=9, ha="center")
ax.set_ylim(-1.0, 6.2)
plt.savefig(f"{FIGDIR}/figure2.pdf", bbox_inches="tight")
plt.close()
print("figure2.pdf: architecture")

# =========================================================================
#  FIGURE 3 — Hierarchical (bilevel) diagram
# =========================================================================
fig, ax = plt.subplots(figsize=(9.5, 4.6)); ax.set_xlim(0,10); ax.set_ylim(0,6); ax.axis("off")

# Upper block
ax.add_patch(plt.Rectangle((0.4,3.4),9.2,2.0, fc="#e7f0fb", ec="#1f4e79", lw=1.2))
ax.text(5.0, 5.05, "Upper level (mixed-integer nonlinear)", ha="center", fontsize=11, weight="bold", color="#1f4e79")
ax.text(5.0, 4.45, r"$\min_{(\lambda,\alpha)\in\mathcal{B}}\; f(\lambda,\alpha)=\mathrm{BCE}(W^{\star}(\lambda,\alpha))+\gamma\,\Vert W^{\star}(\lambda,\alpha)\Vert_F^{2}$", ha="center", fontsize=11)
ax.text(5.0, 3.85, "Algorithm: L-BFGS-B (quasi-Newton, box constraints, finite differences)", ha="center", fontsize=10, weight="bold")

# Arrow down
ax.annotate("", xy=(5,2.85), xytext=(5,3.3), arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#1f4e79"))
ax.text(5.6, 3.05, r"$(\lambda,\alpha)$", fontsize=10)
# Arrow up
ax.annotate("", xy=(4.5,3.3), xytext=(4.5,2.85), arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#7c3a00"))
ax.text(3.6, 3.05, r"$W^{\star}(\lambda,\alpha)$", fontsize=10, color="#7c3a00")

# Lower block
ax.add_patch(plt.Rectangle((0.4,0.4),9.2,2.4, fc="#fdecd2", ec="#7c3a00", lw=1.2))
ax.text(5.0, 2.5, "Lower level (smooth nonconvex)", ha="center", fontsize=11, weight="bold", color="#7c3a00")
ax.text(5.0, 1.92, r"$W^{\star}(\lambda,\alpha)\in\arg\min_{W}\;\mathrm{BCE}(W)+(\lambda/2)\Vert W\Vert_F^{2}$", ha="center", fontsize=11)
ax.text(5.0, 1.32, "Algorithm: Adam (bias-corrected first/second moments)", ha="center", fontsize=10, weight="bold")
ax.text(5.0, 0.78, r"Step: $w \leftarrow w-\alpha\,\hat m_t/(\sqrt{\hat v_t}+\varepsilon)$", ha="center", fontsize=10)
plt.savefig(f"{FIGDIR}/figure3.pdf", bbox_inches="tight")
plt.close()
print("figure3.pdf: bilevel diagram")

# =========================================================================
#  FIGURE 4 — Loss / accuracy / norm curves (Spiral)
# =========================================================================
name="Spiral"; r=results[name]
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))
ep_b = np.arange(len(r["net_b"].loss_h))*5
ep_o = np.arange(len(r["net_o"].loss_h))*5

axes[0].plot(ep_b, r["net_b"].loss_h, color="#1f77b4", lw=1.6, label="Baseline")
axes[0].plot(ep_o, r["net_o"].loss_h, color="#d62728", lw=1.6, label=r"Optimized $(\lambda^\star,\alpha^\star)$")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel(r"$\mathcal{L}$ (BCE+L2)")
axes[0].set_title("Training loss"); axes[0].grid(alpha=.3); axes[0].legend()

axes[1].plot(ep_b, [a*100 for a in r["net_b"].acc_h], color="#1f77b4", lw=1.6, label="Baseline")
axes[1].plot(ep_o, [a*100 for a in r["net_o"].acc_h], color="#d62728", lw=1.6, label="Optimized")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
axes[1].set_title("Classification accuracy"); axes[1].grid(alpha=.3); axes[1].legend()

axes[2].plot(ep_b, r["net_b"].norm_h, color="#1f77b4", lw=1.6, label="Baseline")
axes[2].plot(ep_o, r["net_o"].norm_h, color="#d62728", lw=1.6, label="Optimized")
axes[2].set_xlabel("Epoch"); axes[2].set_ylabel(r"$\Vert W\Vert_F$")
axes[2].set_title("Frobenius weight norm"); axes[2].grid(alpha=.3); axes[2].legend()
plt.tight_layout()
plt.savefig(f"{FIGDIR}/figure4.pdf", bbox_inches="tight")
plt.close()
print("figure4.pdf: training curves Spiral")

# =========================================================================
#  FIGURE 5 — 3D surface of f(lam, lr) plus L-BFGS-B trajectory
#  This is the key "3D plot" requested.
# =========================================================================
# Use Spiral dataset for the surface (most representative)
X, y = results["Spiral"]["X"], results["Spiral"]["y"]
print("Building 3-D surface (this requires re-training many nets)...")
LAM_GRID = np.logspace(-6, -1, 18)   # 18 x 18 = 324 trainings, ~ok
LR_GRID  = np.logspace(-4, np.log10(0.05), 18)
F = np.zeros((len(LAM_GRID), len(LR_GRID)))
for i,lam in enumerate(LAM_GRID):
    for j,lr in enumerate(LR_GRID):
        net = MLP(2, H1, H2, ACT, lr=float(lr), lam=float(lam), seed=0)
        for _ in range(200): net.step(X, y)
        bce  = net.loss_val(X, y)
        nrm2 = sum(np.sum(w**2) for k,w in net._p() if k.startswith("W"))
        F[i,j] = bce + GAMMA*nrm2

LL, AA = np.meshgrid(np.log10(LAM_GRID), np.log10(LR_GRID), indexing="ij")

fig = plt.figure(figsize=(11.5, 4.6))

# 3D surface
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
surf = ax1.plot_surface(LL, AA, F, cmap=cm.viridis, alpha=0.85,
                        linewidth=0, antialiased=True, rcount=40, ccount=40)
# Project trajectory
spi_iters = results["Spiral"]["iters"]
tx = [np.log10(it["lam"]) for it in spi_iters]
ty = [np.log10(it["lr"])  for it in spi_iters]
tz = [it["f"] for it in spi_iters]
ax1.plot(tx, ty, tz, "o-", color="red", lw=1.8, markersize=4, label="L-BFGS-B trajectory")
ax1.scatter([tx[-1]],[ty[-1]],[tz[-1]], color="gold", s=80, ec="black", zorder=5, label=r"$(\lambda^\star,\alpha^\star)$")
ax1.set_xlabel(r"$\log_{10}\lambda$"); ax1.set_ylabel(r"$\log_{10}\alpha$")
ax1.set_zlabel(r"$f(\lambda,\alpha)$")
ax1.set_title(r"Upper-level objective surface $f(\lambda,\alpha)$ — Spiral")
ax1.view_init(elev=28, azim=-58)
ax1.legend(loc="upper left", fontsize=8)

# Contour with trajectory
ax2 = fig.add_subplot(1, 2, 2)
cs = ax2.contourf(LL, AA, F, levels=20, cmap=cm.viridis)
ax2.contour(LL, AA, F, levels=20, colors="white", linewidths=0.3, alpha=0.5)
ax2.plot(tx, ty, "o-", color="red", lw=1.5, markersize=4, label="L-BFGS-B")
ax2.scatter([tx[0]],[ty[0]], color="white", s=70, ec="black", zorder=5, label="start")
ax2.scatter([tx[-1]],[ty[-1]], color="gold", s=90, ec="black", zorder=5, label=r"$(\lambda^\star,\alpha^\star)$")
ax2.set_xlabel(r"$\log_{10}\lambda$"); ax2.set_ylabel(r"$\log_{10}\alpha$")
ax2.set_title("Level sets of $f$ and descent path")
ax2.legend(loc="upper left", fontsize=8)
fig.colorbar(cs, ax=ax2, fraction=0.046, pad=0.04, label=r"$f$")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/figure5.pdf", bbox_inches="tight")
plt.close()
print("figure5.pdf: 3D surface + contour")

# =========================================================================
#  FIGURE 6 — Convergence of L-BFGS-B per dataset
# =========================================================================
fig, ax = plt.subplots(figsize=(8.2, 3.6))
colors = ["#1f77b4","#ff7f0e","#2ca02c","#9467bd","#d62728"]
for name,c in zip(DATASETS, colors):
    fs = [it["f"] for it in results[name]["iters"]]
    # best-so-far cumulative minimum (true L-BFGS-B progress)
    best = np.minimum.accumulate(fs)
    ax.plot(np.arange(1,len(best)+1), best, "-", lw=1.8,
            color=c, label=f"{name}  (k={len(fs)})")
ax.set_xlabel("Function evaluation $k$")
ax.set_ylabel(r"Best-so-far $\min_{j\leq k} f(\lambda_j,\alpha_j)$")
ax.set_title("Cumulative best of the upper-level optimizer (five datasets)")
ax.set_yscale("log"); ax.grid(alpha=.3); ax.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/figure6.pdf", bbox_inches="tight")
plt.close()
print("figure6.pdf: L-BFGS-B convergence (best-so-far)")

# =========================================================================
#  FIGURE 7 — Decision boundaries (baseline vs optimized) for Spiral
# =========================================================================
def make_grid(X, h=0.02, pad=0.4):
    x0_min,x0_max = X[:,0].min()-pad, X[:,0].max()+pad
    x1_min,x1_max = X[:,1].min()-pad, X[:,1].max()+pad
    xx,yy = np.meshgrid(np.arange(x0_min,x0_max,h), np.arange(x1_min,x1_max,h))
    return xx,yy

name="Spiral"; r=results[name]; X=r["X"]; y=r["y"]
xx,yy = make_grid(X)
gridX = np.c_[xx.ravel(), yy.ravel()]
zb = r["net_b"].forward(gridX).reshape(xx.shape)
zo = r["net_o"].forward(gridX).reshape(xx.shape)

fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
for ax, Z, title in [(axes[0], zb, "Baseline"), (axes[1], zo, "Optimized")]:
    ax.contourf(xx,yy,Z, levels=[-0.01,0.5,1.01], colors=["#bcd6f0","#f6c4c4"], alpha=0.7)
    ax.contour(xx,yy,Z, levels=[0.5], colors="black", linewidths=1.2)
    m0=(y.ravel()==0); m1=(y.ravel()==1)
    ax.scatter(X[m0,0],X[m0,1], s=14, c="#1f4e79", edgecolors="white", lw=0.4, label="class 0")
    ax.scatter(X[m1,0],X[m1,1], s=14, c="#a30000", edgecolors="white", lw=0.4, label="class 1")
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
axes[0].legend(loc="lower right", fontsize=8)
plt.suptitle("Decision boundary on Spiral (Baseline vs Optimized hyperparameters)", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/figure7.pdf", bbox_inches="tight")
plt.close()
print("figure7.pdf: decision boundaries")

# =========================================================================
#  FIGURE 8 — Memory norm reduction across datasets
# =========================================================================
fig, ax = plt.subplots(figsize=(8.2, 3.4))
xs = np.arange(len(DATASETS))
nb = [results[d]["norm_b"] for d in DATASETS]
no_ = [results[d]["norm_o"] for d in DATASETS]
w = 0.36
ax.bar(xs-w/2, nb,   w, color="#1f77b4", label=r"Baseline $\Vert W\Vert_F$")
ax.bar(xs+w/2, no_,  w, color="#d62728", label=r"Optimized $\Vert W\Vert_F$")
for i,d in enumerate(DATASETS):
    red = results[d]["norm_red"]
    yt = max(nb[i], no_[i]) * 1.05
    ax.text(i, yt, f"{red:+.0f}%", ha="center", fontsize=9,
            color=("#2ca02c" if red>=0 else "#a30000"))
ax.set_xticks(xs); ax.set_xticklabels(DATASETS)
ax.set_ylabel(r"$\Vert W\Vert_F$")
ax.set_title(r"Frobenius weight norm: baseline vs optimized network")
ax.grid(alpha=.3, axis="y"); ax.legend()
plt.tight_layout()
plt.savefig(f"{FIGDIR}/figure8.pdf", bbox_inches="tight")
plt.close()
print("figure8.pdf: norm reduction bar chart")

# Print final summary table
print("\n=== Summary ===")
print(f"{'Dataset':<10} {'lam*':>10} {'lr*':>10} {'acc_b':>7} {'acc_o':>7} {'red':>7}")
for n in DATASETS:
    m = metrics[n]
    print(f"{n:<10} {m['lam_star']:>10.5f} {m['lr_star']:>10.5f} "
          f"{m['acc_b']*100:>6.1f}% {m['acc_o']*100:>6.1f}% {m['norm_red']:>6.1f}%")
print("\nDone. Figures in /home/claude/kevin/figures/")

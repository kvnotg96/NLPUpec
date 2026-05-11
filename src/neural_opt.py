"""
═══════════════════════════════════════════════════════════════════════════════
  MÓDULO EXPERIMENTAL v4.0
  Red Neuronal Profunda + Optimización No Lineal de Memoria
  ─────────────────────────────────────────────────────────
  • MLP 2 capas ocultas (ReLU / Tanh / ELU configurable)
  • Optimizador Adam con momentum adaptativo
  • Early stopping + LR decay automáticos
  • 5 datasets: Espiral, Lunas, Círculos, XOR, Gaussianas
  • Modelo NL L-BFGS-B corregido (γ pequeño por defecto)
  • Interpretador inteligente actualizado
═══════════════════════════════════════════════════════════════════════════════
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import minimize
import threading
import warnings
import io
import os
import datetime
from tkinter import filedialog, messagebox
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  PALETA
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG  = "#0d0d1a"
PANEL_BG = "#14142b"
CARD_BG  = "#0c2d5a"
ACCENT   = "#ff4d6d"
ACCENT2  = "#00c9e0"
ACCENT3  = "#7bed9f"
ACCENT4  = "#ffa94d"
ACCENT5  = "#c77dff"
TEXT     = "#f0f0f8"
MUTED    = "#7070a0"
FONT     = ("Consolas", 9)
FONT_B   = ("Consolas", 9,  "bold")
FONT_LG  = ("Consolas", 11, "bold")


# ─────────────────────────────────────────────────────────────────────────────
#  SCROLLED FRAME
# ─────────────────────────────────────────────────────────────────────────────
class ScrolledFrame(tk.Frame):
    def __init__(self, parent, bg=PANEL_BG, width=320, **kw):
        super().__init__(parent, bg=bg, width=width, **kw)
        self.pack_propagate(False)
        self._cv = tk.Canvas(self, bg=bg, highlightthickness=0, width=width-14)
        self._sb = tk.Scrollbar(self, orient="vertical", command=self._cv.yview)
        self._cv.configure(yscrollcommand=self._sb.set)
        self._sb.pack(side="right", fill="y")
        self._cv.pack(side="left", fill="both", expand=True)
        self.inner = tk.Frame(self._cv, bg=bg)
        self._wid  = self._cv.create_window((0,0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: self._cv.configure(
            scrollregion=self._cv.bbox("all")))
        self._cv.bind("<Configure>",  lambda e: self._cv.itemconfig(
            self._wid, width=e.width))
        self._cv.bind("<Enter>", lambda e: self._cv.bind_all("<MouseWheel>", self._mw))
        self._cv.bind("<Leave>", lambda e: self._cv.unbind_all("<MouseWheel>"))
        self._cv.bind("<Enter>", lambda e: (
            self._cv.bind_all("<Button-4>", self._mw),
            self._cv.bind_all("<Button-5>", self._mw)))
    def _mw(self, e):
        d = -1 if (e.num==4 or e.delta>0) else 1
        self._cv.yview_scroll(d, "units")


# ─────────────────────────────────────────────────────────────────────────────
#  DATASETS
# ─────────────────────────────────────────────────────────────────────────────
def make_dataset(name="Espiral", n=400, seed=7):
    rng = np.random.RandomState(seed)
    h   = n // 2

    if name == "Espiral":
        t  = np.linspace(0, 4*np.pi, h)
        r  = t / (4*np.pi)
        X0 = np.c_[r*np.cos(t), r*np.sin(t)] + rng.randn(h,2)*.06
        X1 = np.c_[-r*np.cos(t),-r*np.sin(t)] + rng.randn(h,2)*.06

    elif name == "Lunas":
        t0 = np.linspace(0, np.pi, h)
        t1 = np.linspace(np.pi, 2*np.pi, h)
        X0 = np.c_[np.cos(t0), np.sin(t0)] + rng.randn(h,2)*.09
        X1 = np.c_[1+np.cos(t1), .5+np.sin(t1)] + rng.randn(h,2)*.09

    elif name == "Círculos":
        t  = np.linspace(0, 2*np.pi, h)
        X0 = np.c_[.4*np.cos(t), .4*np.sin(t)] + rng.randn(h,2)*.05
        X1 = np.c_[np.cos(t), np.sin(t)] + rng.randn(h,2)*.05

    elif name == "XOR":
        q  = h // 2
        X0 = np.vstack([rng.randn(q,2)*.4+[-.7,-.7],
                        rng.randn(q,2)*.4+[ .7, .7]])
        X1 = np.vstack([rng.randn(q,2)*.4+[ .7,-.7],
                        rng.randn(q,2)*.4+[-.7, .7]])

    elif name == "Gaussianas":
        X0 = rng.randn(h, 2)*.4 + [-.8, 0]
        X1 = rng.randn(h, 2)*.4 + [ .8, 0]

    else:
        return make_dataset("Espiral", n, seed)

    X = np.vstack([X0, X1])
    # Normalizar
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    y = np.vstack([np.zeros((h,1)), np.ones((h,1))])
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


# ─────────────────────────────────────────────────────────────────────────────
#  ACTIVACIONES
# ─────────────────────────────────────────────────────────────────────────────
def act_fn(name, z, deriv=False):
    if name == "ReLU":
        if deriv: return (z > 0).astype(float)
        return np.maximum(0, z)
    elif name == "Tanh":
        t = np.tanh(z)
        if deriv: return 1 - t**2
        return t
    elif name == "ELU":
        alpha = 1.0
        if deriv:
            return np.where(z >= 0, 1.0, alpha * np.exp(np.clip(z,-30,0)))
        return np.where(z >= 0, z, alpha*(np.exp(np.clip(z,-30,0))-1))
    elif name == "Swish":
        s = 1/(1+np.exp(-np.clip(z,-30,30)))
        if deriv: return s + z*s*(1-s)
        return z * s
    return np.maximum(0, z)  # fallback ReLU


# ─────────────────────────────────────────────────────────────────────────────
#  MLP PROFUNDA con Adam
# ─────────────────────────────────────────────────────────────────────────────
class MLP:
    """
    Arquitectura:  entrada(n0) → oculta1(h1, act) → oculta2(h2, act) → salida(1, σ)
    Optimizador:   Adam  (β1=0.9, β2=0.999, ε=1e-8)
    Regulariz.:    L2 en W1, W2, W3
    """
    def __init__(self, n0=2, h1=32, h2=16, act="ReLU",
                 lr=1e-3, lam=1e-4, seed=42):
        np.random.seed(seed)
        self.lr  = lr
        self.lam = lam
        self.act = act
        self.n0, self.h1, self.h2 = n0, h1, h2

        # Xavier init
        def W(fan_in, fan_out):
            s = np.sqrt(2.0/fan_in)
            return np.random.randn(fan_in, fan_out)*s
        self.W1=W(n0,h1); self.b1=np.zeros((1,h1))
        self.W2=W(h1,h2); self.b2=np.zeros((1,h2))
        self.W3=W(h2, 1); self.b3=np.zeros((1, 1))

        # Adam moments
        self._t = 0
        self._m = {k: np.zeros_like(v) for k,v in self._params()}
        self._v = {k: np.zeros_like(v) for k,v in self._params()}

        # Historial
        self.loss_h=[]; self.acc_h=[]; self.norm_h=[]

        # Métricas finales (se sobrescriben al terminar train())
        self.best_loss   = float("inf")
        self.best_acc    = 0.0
        self.epoch_count = 0

    def _params(self):
        return [("W1",self.W1),("b1",self.b1),
                ("W2",self.W2),("b2",self.b2),
                ("W3",self.W3),("b3",self.b3)]

    @staticmethod
    def sigmoid(z):
        return 1.0/(1.0+np.exp(-np.clip(z,-500,500)))

    def forward(self, X):
        self.X  = X
        self.Z1 = X  @ self.W1 + self.b1
        self.A1 = act_fn(self.act, self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = act_fn(self.act, self.Z2)
        self.Z3 = self.A2 @ self.W3 + self.b3
        self.A3 = self.sigmoid(self.Z3)
        return self.A3

    def loss_val(self, X, y):
        A3  = self.forward(X); eps=1e-9
        bce = -np.mean(y*np.log(A3+eps)+(1-y)*np.log(1-A3+eps))
        reg = (self.lam/2)*(np.sum(self.W1**2)+
                             np.sum(self.W2**2)+
                             np.sum(self.W3**2))
        return bce + reg

    def backward(self, X, y):
        m    = X.shape[0]
        b1   = 0.9; b2 = 0.999; eps = 1e-8
        self._t += 1

        # Gradientes
        dZ3 = (self.A3 - y)/m
        dW3 = self.A2.T @ dZ3 + self.lam*self.W3
        db3 = dZ3.sum(0, keepdims=True)

        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * act_fn(self.act, self.Z2, deriv=True)
        dW2 = self.A1.T @ dZ2 + self.lam*self.W2
        db2 = dZ2.sum(0, keepdims=True)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * act_fn(self.act, self.Z1, deriv=True)
        dW1 = X.T  @ dZ1 + self.lam*self.W1
        db1_g = dZ1.sum(0, keepdims=True)

        grads = {"W1":dW1,"b1":db1_g,"W2":dW2,"b2":db2,"W3":dW3,"b3":db3}

        # Adam update
        params_dict = {"W1":self.W1,"b1":self.b1,
                       "W2":self.W2,"b2":self.b2,
                       "W3":self.W3,"b3":self.b3}
        for k, g in grads.items():
            self._m[k] = b1*self._m[k] + (1-b1)*g
            self._v[k] = b2*self._v[k] + (1-b2)*g**2
            m_hat = self._m[k]/(1-b1**self._t)
            v_hat = self._v[k]/(1-b2**self._t)
            params_dict[k] -= self.lr * m_hat/(np.sqrt(v_hat)+eps)

        self.W1=params_dict["W1"]; self.b1=params_dict["b1"]
        self.W2=params_dict["W2"]; self.b2=params_dict["b2"]
        self.W3=params_dict["W3"]; self.b3=params_dict["b3"]

    def step(self, X, y):
        self.forward(X); self.backward(X, y)

    def record(self, X, y):
        l   = self.loss_val(X, y)
        pred= (self.forward(X)>=.5).astype(int)
        acc = float(np.mean(pred==y))
        nrm = float(np.sqrt(sum(np.sum(w**2)
                                for k,w in self._params()
                                if k.startswith("W"))))
        self.loss_h.append(l); self.acc_h.append(acc); self.norm_h.append(nrm)
        return l, acc, nrm

    def train(self, X, y, epochs, patience=60, min_delta=1e-5,
              cb=None, record_every=5):
        """Entrenamiento completo con early stopping y LR decay."""
        best_loss = float("inf")
        best_acc  = 0.0
        wait      = 0
        lr_decay  = 0
        for ep in range(epochs):
            self.step(X, y)
            if ep % record_every == 0:
                l, acc, nrm = self.record(X, y)
                if cb: cb(ep, epochs, l, acc, nrm)
                # Early stopping
                if l < best_loss - min_delta:
                    best_loss = l; best_acc = acc; wait = 0
                else:
                    wait += 1
                if wait >= patience:
                    # LR decay (hasta 2 veces)
                    if lr_decay < 2:
                        self.lr *= 0.3; lr_decay += 1; wait = 0
                    else:
                        break
        # Guardar métricas finales como atributos del objeto
        self.best_loss   = best_loss if best_loss < float("inf") else (self.loss_h[-1] if self.loss_h else 0.0)
        self.best_acc    = best_acc  if best_acc  > 0.0          else (self.acc_h[-1]  if self.acc_h  else 0.0)
        self.epoch_count = ep + 1
        return ep


# ─────────────────────────────────────────────────────────────────────────────
#  OPTIMIZADOR NO LINEAL DE HIPERPARÁMETROS
# ─────────────────────────────────────────────────────────────────────────────
class MemOptimizer:
    """
    min  f(λ, α) = BCE(W*(λ,α)) + γ·‖W*(λ,α)‖²
    s.t. λ ∈ [λ_lo, λ_hi],  α ∈ [α_lo, α_hi]
    Método: L-BFGS-B
    γ pequeño para priorizar clasificación.
    """
    def __init__(self, X, y, h1=32, h2=16, act="ReLU",
                 gamma=0.05, pretrain=200):
        self.X=X; self.y=y
        self.h1=h1; self.h2=h2; self.act=act
        self.gamma=gamma; self.pretrain=pretrain
        self.iterations=[]; self.result=None

    def _obj(self, p):
        lam = float(np.clip(p[0], 1e-6, 0.1))
        lr  = float(np.clip(p[1], 1e-4, 0.05))
        net = MLP(n0=self.X.shape[1], h1=self.h1, h2=self.h2,
                  act=self.act, lr=lr, lam=lam, seed=0)
        for _ in range(self.pretrain):
            net.step(self.X, self.y)
        bce  = net.loss_val(self.X, self.y)
        norm = sum(np.sum(w**2) for k,w in net._params()
                   if k.startswith("W"))
        fval = bce + self.gamma*norm
        self.iterations.append({"lam":lam,"lr":lr,"f":fval,
                                 "bce":bce,"norm":norm})
        return fval

    def run(self, x0=None, lb=(1e-6,1e-4), la=(1e-4,0.05)):
        self.iterations=[]
        x0 = x0 or [1e-4, 1e-3]
        self.result = minimize(self._obj, x0, method="L-BFGS-B",
                               bounds=[lb, la],
                               options={"maxiter":40,"ftol":1e-8})
        lam = float(np.clip(self.result.x[0],*lb))
        lr  = float(np.clip(self.result.x[1],*la))
        return lam, lr


# ─────────────────────────────────────────────────────────────────────────────
#  INTERPRETADOR INTELIGENTE
# ─────────────────────────────────────────────────────────────────────────────
class Interpreter:
    @staticmethod
    def analizar(nb, no, opt_lam, opt_lr, gamma, iters):
        msgs = []
        lb=nb.loss_h[-1]; lo=no.loss_h[-1]
        ab=nb.acc_h[-1];  ao=no.acc_h[-1]
        nrmb=nb.norm_h[-1]; nrmo=no.norm_h[-1]
        red = (nrmb-nrmo)/nrmb*100 if nrmb>1e-9 else 0
        d_loss = (lb-lo)/lb*100   if lb>1e-9 else 0
        d_acc  = (ao-ab)*100

        # 1. Convergencia
        if   lb<0.15: msgs.append(("Convergencia","✅",f"Excelente convergencia (pérdida={lb:.4f}). La red base aprendió bien."))
        elif lb<0.35: msgs.append(("Convergencia","✅",f"Buena convergencia (pérdida={lb:.4f})."))
        elif lb<0.55: msgs.append(("Convergencia","⚠️",f"Convergencia parcial (pérdida={lb:.4f}). Sube épocas o neuronas."))
        else:         msgs.append(("Convergencia","❌",f"Mal convergencia (pérdida={lb:.4f}). Revisa LR o arquitectura."))

        # 2. Precisión base
        if   ab>=0.95: msgs.append(("Precisión Base","✅",f"{ab*100:.1f}% — Clasificación casi perfecta."))
        elif ab>=0.85: msgs.append(("Precisión Base","✅",f"{ab*100:.1f}% — Muy buena clasificación."))
        elif ab>=0.70: msgs.append(("Precisión Base","⚠️",f"{ab*100:.1f}% — Aceptable. Aumenta neuronas o épocas."))
        else:          msgs.append(("Precisión Base","❌",f"{ab*100:.1f}% — Insuficiente. Necesita más capacidad."))

        # 3. Precisión optimizada
        if   ao>=0.95: msgs.append(("Precisión Opt.","✅",f"{ao*100:.1f}% — ¡Clasificación de alto rendimiento!"))
        elif ao>=0.85: msgs.append(("Precisión Opt.","✅",f"{ao*100:.1f}% — Muy buena precisión optimizada."))
        elif ao>=0.70: msgs.append(("Precisión Opt.","⚠️",f"{ao*100:.1f}% — Razonable. Reduce λ o γ."))
        else:          msgs.append(("Precisión Opt.","❌",f"{ao*100:.1f}% — Baja. λ*={opt_lam:.5f} puede estar regularizando en exceso."))

        # 4. Delta precisión
        if d_acc>1:   msgs.append(("Δ Precisión","✅",f"La optimización gana {d_acc:.1f}% de precisión."))
        elif d_acc>-1:msgs.append(("Δ Precisión","✅",f"Sin pérdida de precisión (Δ={d_acc:+.1f}%). Trade-off equilibrado."))
        else:          msgs.append(("Δ Precisión","⚠️",f"Pérdida de {-d_acc:.1f}%. Reduce γ (actualmente {gamma:.2f})."))

        # 5. Memoria
        if   red>40: msgs.append(("Eficiencia Mem.","✅",f"‖W‖ reducida {red:.1f}% ({nrmb:.3f}→{nrmo:.3f}). Gran compresión."))
        elif red>10: msgs.append(("Eficiencia Mem.","✅",f"‖W‖ reducida {red:.1f}%. Buena compresión."))
        elif red>0:  msgs.append(("Eficiencia Mem.","ℹ️",f"Reducción pequeña ({red:.1f}%). γ={gamma:.2f} tiene poco peso."))
        else:        msgs.append(("Eficiencia Mem.","ℹ️",f"Sin reducción de norma. γ muy pequeño o red ya eficiente."))

        # 6. Hiperparámetros
        lr_c = "alta → rápido pero puede oscilar" if opt_lr>0.01 else \
               "moderada → buen equilibrio"        if opt_lr>1e-3 else \
               "baja → convergencia lenta pero estable"
        lam_c= "fuerte → puede sub-entrenar" if opt_lam>0.01 else \
               "moderada"                    if opt_lam>1e-4 else \
               "muy débil → mínima restricción de memoria"
        msgs.append(("Hiperparáms Opt.","ℹ️",
                     f"α*={opt_lr:.5f} ({lr_c})\n"
                     f"λ*={opt_lam:.5f} ({lam_c})"))

        # 7. Iteraciones optimizador
        if iters:
            fv = [it["f"] for it in iters]
            mej = (fv[0]-fv[-1])/abs(fv[0])*100 if abs(fv[0])>1e-9 else 0
            msgs.append(("L-BFGS-B","ℹ️",
                         f"{len(iters)} evaluaciones. "
                         f"f: {fv[0]:.4f} → {fv[-1]:.4f} ({mej:.1f}% mejora)."))

        # 8. Recomendaciones concretas
        recs = []
        if ao < 0.85:
            recs.append("• Aumenta h1 ≥ 64, h2 ≥ 32 neuronas")
        if lb > 0.3:
            recs.append("• Sube épocas a ≥ 1000")
        if opt_lam > 0.01:
            recs.append("• Reduce λ base o γ (está sobreregularizando)")
        if ao < ab - 0.05:
            recs.append("• γ demasiado alto: baja a γ < 0.1")
        if not recs:
            recs.append("• Configuración óptima. Prueba más épocas para refinar.")
        msgs.append(("Recomendaciones","💡", "\n".join(recs)))

        # 9. Veredicto
        score = sum([lb<0.2, lo<0.25, ab>0.9, ao>0.9, red>0, len(iters)>0])
        if score>=5:
            v=("Veredicto","🏆","Clasificación de alto rendimiento. Red y optimizador funcionando en sinergia.")
        elif score>=3:
            v=("Veredicto","✅","Buen rendimiento. Ajusta los sliders según las recomendaciones.")
        else:
            v=("Veredicto","⚠️","Rendimiento mejorable. Sigue las recomendaciones de arriba.")
        msgs.append(v)
        return msgs


# ─────────────────────────────────────────────────────────────────────────────
#  APP PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
class App(tk.Tk):
    DATASETS   = ["Espiral", "Lunas", "Círculos", "XOR", "Gaussianas"]
    ACTS       = ["ReLU", "Tanh", "ELU", "Swish"]

    def __init__(self):
        super().__init__()
        self.title("🧠  Red Neuronal Profunda + Optimización No Lineal  v4.0")
        self.configure(bg=DARK_BG)
        self.geometry("1460x900")
        self.minsize(1100, 720)

        # Variables
        self.v_dataset = tk.StringVar(value="Espiral")
        self.v_act     = tk.StringVar(value="ReLU")
        self.v_epochs  = tk.IntVar(value=1200)
        self.v_h1      = tk.IntVar(value=48)
        self.v_h2      = tk.IntVar(value=24)
        self.v_lr      = tk.DoubleVar(value=0.003)
        self.v_lam     = tk.DoubleVar(value=1e-4)
        self.v_gamma   = tk.DoubleVar(value=0.05)
        self.v_patience= tk.IntVar(value=80)

        # Estado
        self.X=None; self.y=None
        self.net_base=None; self.net_opt=None
        self.opt_done=False; self.opt_lam=None; self.opt_lr=None
        self.opt_iters=[]; self.running=False
        self._reload_data()
        self._build_ui()

    def _reload_data(self):
        self.X, self.y = make_dataset(self.v_dataset.get(), n=500)

    # ══════════════════════════════════════════════════════════════════════════
    #  LAYOUT
    # ══════════════════════════════════════════════════════════════════════════
    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=CARD_BG, pady=5)
        hdr.pack(fill="x")
        tk.Label(hdr,
            text="🧠  MÓDULO EXPERIMENTAL v4.0 — Red Neuronal Profunda + Optimización No Lineal de Memoria",
            font=("Consolas",11,"bold"), bg=CARD_BG, fg=ACCENT2).pack()
        tk.Label(hdr,
            text="MLP 3 capas · Adam · Early Stopping · L-BFGS-B · 5 Datasets · ReLU/Tanh/ELU/Swish",
            font=FONT, bg=CARD_BG, fg=MUTED).pack()

        body = tk.Frame(self, bg=DARK_BG)
        body.pack(fill="both", expand=True, padx=5, pady=4)

        self.sf = ScrolledFrame(body, bg=PANEL_BG, width=320)
        self.sf.pack(side="left", fill="y", padx=(0,5))
        self._build_sidebar(self.sf.inner)

        right = tk.Frame(body, bg=DARK_BG)
        right.pack(side="left", fill="both", expand=True)
        self._build_notebook(right)

    # ══════════════════════════════════════════════════════════════════════════
    #  SIDEBAR
    # ══════════════════════════════════════════════════════════════════════════
    def _build_sidebar(self, p):
        def sec(t):
            tk.Label(p, text=t, font=FONT_B, bg=PANEL_BG,
                     fg=ACCENT4, anchor="w").pack(fill="x", padx=10, pady=(9,2))
        def div():
            tk.Frame(p, bg="#252550", height=1).pack(fill="x", padx=8, pady=4)

        # ── Dataset ───────────────────────────────────────────────────────────
        sec("🗃  Dataset")
        ds_fr = tk.Frame(p, bg=PANEL_BG)
        ds_fr.pack(fill="x", padx=10, pady=2)
        for ds in self.DATASETS:
            tk.Radiobutton(ds_fr, text=ds, variable=self.v_dataset,
                           value=ds, font=FONT, bg=PANEL_BG, fg=TEXT,
                           selectcolor=CARD_BG, activebackground=PANEL_BG,
                           command=self._on_dataset_change).pack(
                               side="left", padx=4)

        # ── Activación ────────────────────────────────────────────────────────
        sec("⚡  Activación Oculta")
        act_fr = tk.Frame(p, bg=PANEL_BG)
        act_fr.pack(fill="x", padx=10, pady=2)
        for a in self.ACTS:
            tk.Radiobutton(act_fr, text=a, variable=self.v_act,
                           value=a, font=FONT, bg=PANEL_BG, fg=TEXT,
                           selectcolor=CARD_BG, activebackground=PANEL_BG
                           ).pack(side="left", padx=4)

        div()

        # ── Arquitectura ──────────────────────────────────────────────────────
        sec("🔷  Arquitectura")
        sliders_arch = [
            ("Neuronas h1", self.v_h1,  8, 128, 4,  "{:.0f}"),
            ("Neuronas h2", self.v_h2,  4,  64, 4,  "{:.0f}"),
        ]
        for lbl,var,lo,hi,res,fmt in sliders_arch:
            self._slider(p, lbl, var, lo, hi, res, fmt)

        div()

        # ── Entrenamiento ─────────────────────────────────────────────────────
        sec("📈  Entrenamiento")
        sliders_train = [
            ("Épocas",    self.v_epochs,  200, 5000, 100, "{:.0f}"),
            ("LR  base",  self.v_lr,     1e-4, 0.05, 1e-4,"{:.4f}"),
            ("λ   base",  self.v_lam,    1e-6, 0.05, 1e-5,"{:.5f}"),
            ("Patience",  self.v_patience, 20, 200,  10, "{:.0f}"),
        ]
        for lbl,var,lo,hi,res,fmt in sliders_train:
            self._slider(p, lbl, var, lo, hi, res, fmt)

        div()

        # ── Optimizador NL ────────────────────────────────────────────────────
        sec("⚙  Optimizador NL")
        self._slider(p, "γ (memoria)", self.v_gamma, 0.0, 0.5, 0.01, "{:.2f}")

        div()

        # ── Acciones ──────────────────────────────────────────────────────────
        sec("▶  Acciones")
        self.btn_opt   = self._btn(p,"⚙  Optimizar Hiperparámetros","#3d3d6b",self._run_opt)
        self.btn_train = self._btn(p,"▶  Entrenar Ambas Redes",     ACCENT,   self._run_train)
        self.btn_reset = self._btn(p,"↺  Reiniciar Todo",           CARD_BG,  self._reset)

        div()

        # ── Exportar PDFs ──────────────────────────────────────────────────────
        sec("📄  Exportar")
        self.btn_pdf_rep  = self._btn(p, "📊  Reporte Inteligente PDF",  "#1a472a", self._export_report_pdf)
        self.btn_pdf_math = self._btn(p, "📐  Modelo Matemático PDF",    "#1a1a5e", self._export_math_pdf)

        div()

        # ── Status ────────────────────────────────────────────────────────────
        sec("📡  Estado")
        self.lbl_status = tk.Label(p, text="Listo. Configura y entrena.",
                                   font=FONT, bg=PANEL_BG, fg=ACCENT3,
                                   wraplength=285, justify="left")
        self.lbl_status.pack(fill="x", padx=10, pady=2)

        div()

        # ── Métricas ──────────────────────────────────────────────────────────
        sec("📊  Métricas Finales")
        self.mvars={}
        rows=[("loss_b","Pérd. Base",ACCENT2),("loss_o","Pérd. Opt.",ACCENT3),
              ("acc_b", "Prec. Base",ACCENT2),("acc_o", "Prec. Opt.",ACCENT3),
              ("norm_b","‖W‖  Base", ACCENT), ("norm_o","‖W‖  Opt.", ACCENT3),
              ("red",   "Δ Mem.",    ACCENT4),("ep_b",  "Épocas B.", ACCENT2),
              ("ep_o",  "Épocas O.", ACCENT3),
              ("opt_lr","α* ópt.",   ACCENT4),("opt_lam","λ* ópt.",  ACCENT4)]
        for k,lbl,col in rows:
            r=tk.Frame(p,bg=PANEL_BG); r.pack(fill="x",padx=10,pady=1)
            tk.Label(r,text=lbl+":",font=FONT,bg=PANEL_BG,
                     fg=TEXT,width=12,anchor="w").pack(side="left")
            v=tk.StringVar(value="—"); self.mvars[k]=v
            tk.Label(r,textvariable=v,font=FONT_B,
                     bg=PANEL_BG,fg=col,width=10,anchor="e").pack(side="right")

        div()

        # ── Diagrama MLP compacto ─────────────────────────────────────────────
        sec("🔷  Arquitectura MLP")
        self.mlp_mini = tk.Canvas(p, width=290, height=160,
                                  bg=DARK_BG, highlightthickness=0)
        self.mlp_mini.pack(padx=8, pady=4)
        self._draw_mlp_mini()

        div()

        # Fórmulas rápidas
        sec("📐  Modelo (resumen)")
        for txt,col in [
            ("min f(λ,α) = ℒ(W*(λ,α)) + γ‖W*‖²", ACCENT2),
            ("ℒ = BCE + (λ/2)(‖W₁‖²+‖W₂‖²+‖W₃‖²)", TEXT),
            ("Adam:  mₜ=β₁mₜ₋₁+(1-β₁)g", MUTED),
            ("       vₜ=β₂vₜ₋₁+(1-β₂)g²", MUTED),
            ("       θ←θ - α·m̂ₜ/(√v̂ₜ+ε)", MUTED),
        ]:
            tk.Label(p, text=txt, font=("Consolas",8),
                     bg=PANEL_BG, fg=col, anchor="w").pack(
                         fill="x", padx=14, pady=0)

        tk.Frame(p, bg=PANEL_BG, height=16).pack()

    def _slider(self, parent, label, var, lo, hi, res, fmt):
        f = tk.Frame(parent, bg=PANEL_BG)
        f.pack(fill="x", padx=10, pady=2)
        tk.Label(f, text=label, font=FONT, bg=PANEL_BG,
                 fg=TEXT, width=11, anchor="w").pack(side="left")
        val = tk.Label(f, text=fmt.format(var.get()),
                       font=FONT_B, bg=PANEL_BG, fg=ACCENT2, width=8)
        val.pack(side="right")
        def upd(v):
            val.config(text=fmt.format(float(v)))
            if var in (self.v_h1, self.v_h2):
                self.after(20, self._draw_mlp_mini)
                if hasattr(self,'mlp_tab_cv'):
                    self.after(20, self._redraw_mlp_tab)
        tk.Scale(f, variable=var, from_=lo, to=hi, resolution=res,
                 orient="horizontal", bg=PANEL_BG, fg=TEXT,
                 troughcolor=CARD_BG, highlightthickness=0,
                 sliderlength=13, length=128, showvalue=False,
                 command=upd).pack(side="right", padx=4)

    def _btn(self, p, text, color, cmd):
        b = tk.Button(p, text=text, font=FONT_B, bg=color, fg=TEXT,
                      bd=0, cursor="hand2", pady=7, command=cmd,
                      activebackground=DARK_BG, activeforeground=TEXT)
        b.pack(fill="x", padx=10, pady=3)
        return b

    def _on_dataset_change(self):
        self._reload_data()
        self._reset_plots_only()
        self._init_boundary()

    # ══════════════════════════════════════════════════════════════════════════
    #  NOTEBOOK — 6 PESTAÑAS
    # ══════════════════════════════════════════════════════════════════════════
    def _build_notebook(self, parent):
        sty = ttk.Style()
        sty.theme_use("default")
        sty.configure("N.TNotebook", background=DARK_BG, borderwidth=0)
        sty.configure("N.TNotebook.Tab", background=CARD_BG, foreground=TEXT,
                      font=("Consolas",9,"bold"), padding=[10,5])
        sty.map("N.TNotebook.Tab",
                background=[("selected",ACCENT2)],
                foreground=[("selected",DARK_BG)])

        self.nb = ttk.Notebook(parent, style="N.TNotebook")
        self.nb.pack(fill="both", expand=True)

        tabs = [
            ("  📈  Entrenamiento  ", self._build_tab_train),
            ("  🗺  Frontera  ",      self._build_tab_boundary),
            ("  🧠  Interpretador  ", self._build_tab_interp),
            ("  🔍  Optimizador  ",   self._build_tab_opt),
            ("  🔷  Arquitectura  ",  self._build_tab_mlp),
            ("  📐  Modelo Matemático  ", self._build_tab_math),
        ]
        for label, builder in tabs:
            f = tk.Frame(self.nb, bg=DARK_BG)
            self.nb.add(f, text=label)
            builder(f)

    # ── TAB 1: Entrenamiento ──────────────────────────────────────────────────
    def _build_tab_train(self, p):
        self.fig1 = plt.Figure(figsize=(10.5,6.8), facecolor=DARK_BG)
        self.fig1.subplots_adjust(wspace=0.32, left=0.07,
                                  right=0.97, top=0.92, bottom=0.1)
        self.ax_loss = self.fig1.add_subplot(1,3,1, facecolor=PANEL_BG)
        self.ax_norm = self.fig1.add_subplot(1,3,2, facecolor=PANEL_BG)
        self.ax_acc  = self.fig1.add_subplot(1,3,3, facecolor=PANEL_BG)
        for ax in [self.ax_loss,self.ax_norm,self.ax_acc]:
            self._sax(ax)
        self._placeholder([self.ax_loss,self.ax_norm,self.ax_acc])
        c=FigureCanvasTkAgg(self.fig1,p); c.get_tk_widget().pack(fill="both",expand=True,padx=4,pady=4)
        self.cv1=c

    # ── TAB 2: Frontera ───────────────────────────────────────────────────────
    def _build_tab_boundary(self, p):
        self.fig2 = plt.Figure(figsize=(10.5,6.8), facecolor=DARK_BG)
        self.fig2.subplots_adjust(wspace=0.3, left=0.07,
                                  right=0.97, top=0.92, bottom=0.08)
        self.ax_bd_b = self.fig2.add_subplot(1,2,1, facecolor=PANEL_BG)
        self.ax_bd_o = self.fig2.add_subplot(1,2,2, facecolor=PANEL_BG)
        for ax in [self.ax_bd_b, self.ax_bd_o]: self._sax(ax)
        self._init_boundary()
        c=FigureCanvasTkAgg(self.fig2,p); c.get_tk_widget().pack(fill="both",expand=True,padx=4,pady=4)
        self.cv2=c

    def _init_boundary(self):
        for ax,t in [(self.ax_bd_b,"Red Base — sin entrenar"),
                     (self.ax_bd_o,"Red Optimizada — sin entrenar")]:
            ax.cla(); ax.set_facecolor(PANEL_BG)
            ax.scatter(self.X[:,0],self.X[:,1],c=self.y.ravel(),
                       cmap="coolwarm",s=8,alpha=0.5,edgecolors="none")
            ax.set_title(t,fontsize=8,fontweight="bold",color=ACCENT2,pad=4)
            self._sax(ax)
        if hasattr(self,'cv2'): self.cv2.draw()

    # ── TAB 3: Interpretador ──────────────────────────────────────────────────
    def _build_tab_interp(self, p):
        hdr=tk.Frame(p,bg=DARK_BG); hdr.pack(fill="x",padx=8,pady=(7,2))
        tk.Label(hdr,text="🧠  INTERPRETADOR INTELIGENTE DE RESULTADOS",
                 font=("Consolas",11,"bold"),bg=DARK_BG,fg=ACCENT2).pack(anchor="w")
        tk.Label(hdr,text="Análisis automático · Diagnóstico · Recomendaciones",
                 font=FONT,bg=DARK_BG,fg=MUTED).pack(anchor="w",pady=(0,4))
        frm=tk.Frame(p,bg=DARK_BG); frm.pack(fill="both",expand=True)
        sb=tk.Scrollbar(frm); sb.pack(side="right",fill="y")
        self.txt_interp=tk.Text(frm,bg=PANEL_BG,fg=TEXT,font=("Consolas",9),
                                wrap="word",bd=0,padx=14,pady=10,
                                yscrollcommand=sb.set,state="disabled",
                                spacing1=2,spacing2=4,spacing3=4,
                                selectbackground=CARD_BG)
        self.txt_interp.pack(side="left",fill="both",expand=True)
        sb.config(command=self.txt_interp.yview)
        for tag,fg,bold,bg in [
            ("h1", ACCENT2, True,  DARK_BG),
            ("h2", ACCENT4, True,  DARK_BG),
            ("h3", ACCENT3, True,  DARK_BG),
            ("ok", ACCENT3, False, DARK_BG),
            ("warn",ACCENT4,False, DARK_BG),
            ("err", ACCENT, False, DARK_BG),
            ("info",MUTED,  False, DARK_BG),
            ("body",TEXT,   False, DARK_BG),
            ("sep", "#202050",False,DARK_BG),
        ]:
            self.txt_interp.tag_configure(
                tag, foreground=fg, background=bg,
                font=("Consolas",9,"bold" if bold else "normal"))
        self._interp_placeholder()

    def _interp_placeholder(self):
        T=self.txt_interp; T.config(state="normal"); T.delete("1.0","end")
        T.insert("end","═"*68+"\n","sep")
        T.insert("end","  Entrena ambas redes para activar el interpretador.\n","info")
        T.insert("end","  Evaluará: convergencia, precisión, memoria, trade-off,\n","info")
        T.insert("end","  hiperparámetros, optimizador y veredicto global.\n","info")
        T.insert("end","═"*68+"\n","sep")
        T.config(state="disabled")

    def _update_interp(self, msgs):
        T=self.txt_interp; T.config(state="normal"); T.delete("1.0","end")
        T.insert("end","═"*68+"\n","sep")
        T.insert("end","   ANÁLISIS AUTOMÁTICO DE RESULTADOS\n","h1")
        T.insert("end","═"*68+"\n","sep")
        T.insert("end","\n","body")
        for cat,nivel,texto in msgs:
            tag="ok"  if "✅" in nivel or "🏆" in nivel else \
                "warn" if "⚠️" in nivel or "💡" in nivel else \
                "err"  if "❌" in nivel else "info"
            T.insert("end",f"  {nivel}  ","body")
            T.insert("end",f"[{cat}]\n","h2")
            for line in texto.split("\n"):
                T.insert("end",f"     {line}\n",tag)
            T.insert("end","  "+"─"*64+"\n","sep")
        T.insert("end","\n  Modifica los sliders y reentrena para comparar.\n","info")
        T.config(state="disabled"); T.see("1.0")

    # ── TAB 4: Optimizador ────────────────────────────────────────────────────
    def _build_tab_opt(self, p):
        # Cabecera informativa
        hdr = tk.Frame(p, bg=DARK_BG)
        hdr.pack(fill="x", padx=8, pady=(6,2))
        tk.Label(hdr,
            text="🔍  TRAYECTORIA DEL OPTIMIZADOR L-BFGS-B",
            font=("Consolas",11,"bold"), bg=DARK_BG, fg=ACCENT2).pack(side="left")
        tk.Label(hdr,
            text="   Ejecuta ⋅Optimizar Hiperparámetros⋅ para ver el análisis",
            font=("Consolas",8), bg=DARK_BG, fg=MUTED).pack(side="left")

        # Figura 2×3 = 6 subplots
        self.fig3 = plt.Figure(figsize=(10.8, 7.0), facecolor=DARK_BG)
        self.fig3.subplots_adjust(
            hspace=0.52, wspace=0.38,
            left=0.08, right=0.96,
            top=0.93,  bottom=0.09)

        self.ax_fv   = self.fig3.add_subplot(2, 3, 1, facecolor=PANEL_BG)  # f por iter
        self.ax_lm   = self.fig3.add_subplot(2, 3, 2, facecolor=PANEL_BG)  # tray. λ
        self.ax_lr2  = self.fig3.add_subplot(2, 3, 3, facecolor=PANEL_BG)  # tray. α
        self.ax_sc   = self.fig3.add_subplot(2, 3, 4, facecolor=PANEL_BG)  # mapa (λ,α)→f
        self.ax_grad = self.fig3.add_subplot(2, 3, 5, facecolor=PANEL_BG)  # mejora relativa
        self.ax_bce  = self.fig3.add_subplot(2, 3, 6, facecolor=PANEL_BG)  # BCE vs norma

        self._opt_axes = [self.ax_fv, self.ax_lm, self.ax_lr2,
                          self.ax_sc, self.ax_grad, self.ax_bce]
        for ax in self._opt_axes:
            self._sax(ax)
        self._placeholder(self._opt_axes)

        c = FigureCanvasTkAgg(self.fig3, p)
        c.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=(0,4))
        self.cv3 = c

    # ── TAB 5: Arquitectura MLP ───────────────────────────────────────────────
    def _build_tab_mlp(self, p):
        hdr=tk.Frame(p,bg=DARK_BG); hdr.pack(fill="x",padx=10,pady=(8,2))
        tk.Label(hdr,text="🔷  ARQUITECTURA DEL PERCEPTRÓN MULTICAPA PROFUNDO",
                 font=("Consolas",12,"bold"),bg=DARK_BG,fg=ACCENT2).pack(side="left")
        self.mlp_tab_cv=tk.Canvas(p,bg=DARK_BG,highlightthickness=0)
        self.mlp_tab_cv.pack(fill="both",expand=True,padx=10,pady=(2,4))
        self.mlp_tab_cv.bind("<Configure>",lambda e: self._redraw_mlp_tab())
        # Leyenda
        leg=tk.Frame(p,bg=PANEL_BG,pady=4); leg.pack(fill="x",padx=10,pady=(0,8))
        for col,txt in [
            (ACCENT2,"Entrada  x∈ℝ²"),
            (ACCENT3,f"Oculta 1  (h1)  — act(·)"),
            (ACCENT5,f"Oculta 2  (h2)  — act(·)"),
            (ACCENT, "Salida  ŷ=σ(z)"),
            (ACCENT4,"W matrices, b sesgos"),
        ]:
            r=tk.Frame(leg,bg=PANEL_BG); r.pack(side="left",padx=12)
            cc=tk.Canvas(r,width=13,height=13,bg=PANEL_BG,highlightthickness=0)
            cc.create_oval(1,1,12,12,fill=col,outline=""); cc.pack(side="left")
            tk.Label(r,text=" "+txt,font=("Consolas",8),bg=PANEL_BG,fg=TEXT).pack(side="left")

    def _redraw_mlp_tab(self):
        if not hasattr(self,'mlp_tab_cv'): return
        c=self.mlp_tab_cv
        c.delete("all")
        W=c.winfo_width() or 900; H=c.winfo_height() or 540
        if W<50 or H<50: return
        h1=min(self.v_h1.get(),10); h2=min(self.v_h2.get(),8)
        layers=[2,h1,h2,1]
        names=["Entrada\n(n₀=2)",f"Oculta 1\n(h1={self.v_h1.get()})",
               f"Oculta 2\n(h2={self.v_h2.get()})","Salida\n(m=1)"]
        acts=["x","act(·)","act(·)","σ(·)"]
        cols=[ACCENT2,ACCENT3,ACCENT5,ACCENT]
        radii=[20,16,14,22]
        margin=70
        xs=[margin+i*(W-2*margin)//3 for i in range(4)]
        pos=[]
        for n,x in zip(layers,xs):
            usable=H-110; gap=usable/(n+1)
            pos.append([(x,55+gap*(j+1)) for j in range(n)])
        np.random.seed(99)
        # Conexiones
        for li in range(3):
            for (x1,y1) in pos[li]:
                for (x2,y2) in pos[li+1]:
                    lw=max(0.4,min(2.5,abs(np.random.randn())*1.2))
                    c.create_line(x1,y1,x2,y2,fill="#1e2060",width=lw)
        # Labels matrices
        w_labels=[("W₁∈ℝ^{2×h1}","b₁∈ℝ^{h1}"),
                  ("W₂∈ℝ^{h1×h2}","b₂∈ℝ^{h2}"),
                  ("W₃∈ℝ^{h2×1}","b₃∈ℝ")]
        for li,(wl,bl) in enumerate(w_labels):
            mx=(xs[li]+xs[li+1])//2
            c.create_text(mx,12,text=wl,fill=ACCENT4,font=("Consolas",9,"bold"))
            c.create_text(mx,26,text=bl, fill=MUTED,  font=("Consolas",7))
        # Nodos
        labels_by_layer=[
            ["x₁","x₂"],
            [f"h{i+1}" for i in range(h1)],
            [f"h{i+1}" for i in range(h2)],
            ["ŷ"]
        ]
        for li,(nodes,col,r,lbs) in enumerate(zip(pos,cols,radii,labels_by_layer)):
            for idx,(x,y) in enumerate(nodes):
                c.create_oval(x-r+2,y-r+2,x+r+2,y+r+2,fill="#06060f",outline="")
                c.create_oval(x-r,y-r,x+r,y+r,fill=col,outline="#ffffff",width=1)
                lbl=lbs[idx] if idx<len(lbs) else ""
                c.create_text(x,y,text=lbl,fill=DARK_BG,font=("Consolas",8,"bold"))
            # Puntos suspensivos
            need = (li==1 and self.v_h1.get()>10) or \
                   (li==2 and self.v_h2.get()>8)
            if need and nodes:
                lx,ly=nodes[-1]
                for dy in [22,32,42]:
                    c.create_oval(lx-3,ly+dy-3,lx+3,ly+dy+3,fill=TEXT,outline="")
                extra=self.v_h1.get()-10 if li==1 else self.v_h2.get()-8
                c.create_text(lx,ly+56,text=f"({extra}+)",fill=MUTED,font=("Consolas",7))
        # Nombres y activaciones
        for x,name,act,col in zip(xs,names,acts,cols):
            c.create_text(x,40,text=name,fill=col,font=("Consolas",9,"bold"),justify="center")
            c.create_text(x,H-16,text=f"f={act}",fill=col,font=("Consolas",8,"italic"))
        # Forward pass
        c.create_text(W//2,H-2,
            text="⟶  x ↦ Z₁=xW₁+b₁ ↦ A₁=act(Z₁) ↦ Z₂=A₁W₂+b₂ ↦ A₂=act(Z₂) ↦ Z₃=A₂W₃+b₃ ↦ ŷ=σ(Z₃)",
            fill=ACCENT4,font=("Consolas",8,"bold"))

    # ── TAB 6: Modelo Matemático ──────────────────────────────────────────────
    def _build_tab_math(self, p):
        hdr=tk.Frame(p,bg=CARD_BG,pady=6); hdr.pack(fill="x")
        tk.Label(hdr,text="📐  FORMULACIÓN MATEMÁTICA DEL MODELO — v4.0",
                 font=("Consolas",12,"bold"),bg=CARD_BG,fg=ACCENT2).pack()
        tk.Label(hdr,text="MLP 3 capas · Adam · Early Stopping · L-BFGS-B · 9 secciones",
                 font=FONT,bg=CARD_BG,fg=MUTED).pack()
        body=tk.Frame(p,bg=DARK_BG); body.pack(fill="both",expand=True)
        sb=tk.Scrollbar(body); sb.pack(side="right",fill="y")
        T=tk.Text(body,bg=DARK_BG,fg=TEXT,font=("Consolas",10),
                  wrap="word",bd=0,padx=30,pady=14,yscrollcommand=sb.set,
                  state="disabled",spacing1=2,spacing2=4,spacing3=4,
                  selectbackground=CARD_BG)
        T.pack(side="left",fill="both",expand=True)
        sb.config(command=T.yview)
        for tag,fg,sz,bold,bg in [
            ("h1",ACCENT2,13,True, DARK_BG),("h2",ACCENT4,11,True, DARK_BG),
            ("h3",ACCENT3,10,True, DARK_BG),("eq","#dde8ff",11,False,DARK_BG),
            ("eb",ACCENT2,11,True, "#0c1e40"),("body",TEXT,9,False,DARK_BG),
            ("note",MUTED,8,False, DARK_BG), ("def",ACCENT3,9,False,DARK_BG),
            ("sep","#1a1a50",8,False,DARK_BG),
        ]:
            T.tag_configure(tag,foreground=fg,background=bg,
                            font=("Consolas",sz,"bold" if bold else "normal"),
                            lmargin1=(60 if tag in("eq","eb") else
                                      50 if tag in("note","def") else 30),
                            lmargin2=(60 if tag in("eq","eb") else
                                      50 if tag in("note","def") else 30),
                            spacing1=(14 if tag=="h1" else
                                      12 if tag=="h2" else
                                       8 if tag=="h3" else
                                       7 if tag in("eb","eq") else 2),
                            spacing3=(6 if tag in("h1","h2") else
                                      4 if tag=="h3" else
                                      7 if tag in("eb","eq") else 2))
        T.config(state="normal")
        def w(t,tg="body"): T.insert("end",t+"\n",tg)
        def sep(): T.insert("end","═"*80+"\n","sep")
        def div(): T.insert("end","─"*60+"\n","sep")

        sep()
        w("📐  MODELO MATEMÁTICO FORMAL — v4.0","h1")
        w("MLP 3 capas + Adam + Early Stopping + Optimización No Lineal L-BFGS-B","note")
        sep()

        w("§ 1.  ESPACIO DE DATOS","h2"); div()
        w("Conjunto de entrenamiento:","body")
        w("𝒟 = { (xⁿ, yⁿ) }ₙ₌₁ᴺ  ⊆  ℝ^{n₀} × {0,1}","eb")
        w("  xⁿ ∈ ℝ^{n₀}   —  vector de características  (n₀=2 en este módulo)","def")
        w("  yⁿ ∈ {0,1}    —  etiqueta binaria","def")
        w("  N = 500        —  tamaño del dataset (normalizado μ=0, σ=1)","def")
        w("","body")

        w("§ 2.  ARQUITECTURA MLP PROFUNDA","h2"); div()
        w("Red de 4 capas (entrada + 2 ocultas + salida):","body")
        w("  Capa 0 — Entrada  :  n₀  neuronas","def")
        w("  Capa 1 — Oculta 1 :  h₁  neuronas  (ReLU / Tanh / ELU / Swish)","def")
        w("  Capa 2 — Oculta 2 :  h₂  neuronas  (misma activación)","def")
        w("  Capa 3 — Salida   :  1   neurona    (Sigmoid)","def")
        w("","body")
        w("Parámetros  Θ = { W₁,b₁, W₂,b₂, W₃,b₃ }:","h3")
        w("  W₁ ∈ ℝ^{n₀×h₁},  b₁ ∈ ℝ^{h₁}","def")
        w("  W₂ ∈ ℝ^{h₁×h₂},  b₂ ∈ ℝ^{h₂}","def")
        w("  W₃ ∈ ℝ^{h₂×1},   b₃ ∈ ℝ","def")
        w("  |Θ| = n₀h₁ + h₁ + h₁h₂ + h₂ + h₂ + 1","note")
        w("","body")

        w("§ 3.  FORWARD PASS","h2"); div()
        w("  Z₁ = X·W₁ + b₁  ∈ ℝ^{N×h₁}","eq")
        w("  A₁ = φ(Z₁)       ∈ ℝ^{N×h₁}   [φ ∈ {ReLU,Tanh,ELU,Swish}]","eq")
        w("  Z₂ = A₁·W₂ + b₂ ∈ ℝ^{N×h₂}","eq")
        w("  A₂ = φ(Z₂)       ∈ ℝ^{N×h₂}","eq")
        w("  Z₃ = A₂·W₃ + b₃ ∈ ℝ^{N×1}","eq")
        w("  Ŷ  = σ(Z₃)  =  (1+e⁻ᶻ³)⁻¹  ∈ (0,1)^{N×1}","eb")
        w("","body")
        w("Activaciones disponibles φ:","h3")
        w("  ReLU(z)  = max(0,z)","def")
        w("  Tanh(z)  = (eᶻ−e⁻ᶻ)/(eᶻ+e⁻ᶻ)","def")
        w("  ELU(z)   = z  si z≥0,  α(eᶻ−1)  si z<0,  α=1","def")
        w("  Swish(z) = z·σ(z)   (auto-gated, sin punto muerto)","def")
        w("","body")

        w("§ 4.  FUNCIÓN DE PÉRDIDA","h2"); div()
        w("  ℒ(Θ;λ)  =  BCE(Θ)  +  Rλ(Θ)","eb")
        w("","body")
        w("  BCE = -(1/N) Σₙ [yⁿ log(Ŷⁿ+ε) + (1-yⁿ)log(1-Ŷⁿ+ε)]","eq")
        w("  Rλ  = (λ/2)(‖W₁‖²_F + ‖W₂‖²_F + ‖W₃‖²_F)","eq")
        w("  ε = 10⁻⁹  (estabilidad numérica)","note")
        w("","body")

        w("§ 5.  OPTIMIZADOR ADAM","h2"); div()
        w("Adam (Adaptive Moment Estimation), Kingma & Ba, 2015:","body")
        w("  Para cada parámetro θ ∈ Θ y su gradiente g = ∂ℒ/∂θ:","body")
        w("","body")
        w("  mₜ = β₁·mₜ₋₁ + (1−β₁)·g          [1er momento]","eq")
        w("  vₜ = β₂·vₜ₋₁ + (1−β₂)·g²          [2do momento]","eq")
        w("  m̂ₜ = mₜ/(1−β₁ᵗ)                    [corrección sesgo]","eq")
        w("  v̂ₜ = vₜ/(1−β₂ᵗ)                    [corrección sesgo]","eq")
        w("  θₜ₊₁ = θₜ − α · m̂ₜ / (√v̂ₜ + ε)   [actualización]","eb")
        w("","body")
        w("  β₁=0.9, β₂=0.999, ε=10⁻⁸   (valores estándar)","def")
        w("  α > 0  —  tasa de aprendizaje global","def")
        w("","body")

        w("§ 6.  EARLY STOPPING + LR DECAY","h2"); div()
        w("Mecanismo de parada anticipada para evitar sobreajuste:","body")
        w("","body")
        w("  Si ℒ(t) ≥ ℒ_best − δ  durante 'patience' épocas:","eq")
        w("    → α ← 0.3·α  (LR decay, hasta 2 veces)","eq")
        w("    → Si se agota: detener entrenamiento","eq")
        w("","body")
        w("  δ = 10⁻⁵  (umbral mínimo de mejora)","def")
        w("  patience ∈ [20,200]  (configurable por slider)","def")
        w("","body")

        w("§ 7.  MODELO DE OPTIMIZACIÓN NO LINEAL","h2"); div()
        w("  ┌──────────────────────────────────────────────────────────────┐","eq")
        w("  │  min   f(λ,α) = ℒ(W*(λ,α)) + γ·‖W*(λ,α)‖²               │","eb")
        w("  │  (λ,α)                                                      │","eq")
        w("  │  s.a.  λ ∈ [10⁻⁶, 0.1]                                    │","eq")
        w("  │        α ∈ [10⁻⁴, 0.05]                                   │","eq")
        w("  └──────────────────────────────────────────────────────────────┘","eq")
        w("","body")
        w("  W*(λ,α) — Θ tras T₀=200 pasos de Adam con (λ,α)","def")
        w("  γ ≥ 0   — peso de memoria (recomendado γ ≤ 0.1)","def")
        w("  ‖W*‖²  = ‖W₁*‖²_F+‖W₂*‖²_F+‖W₃*‖²_F  (proxy memoria)","def")
        w("","body")
        w("Resolución: L-BFGS-B (cuasi-Newton, restricciones de caja)","h3")
        w("  ∇f calculado por diferencias finitas automáticas.","note")
        w("  Convergencia superlineal. Sin gradientes analíticos.","note")
        w("","body")

        w("§ 8.  MÉTRICA DE EFICIENCIA DE MEMORIA","h2"); div()
        w("  ‖Θ‖ := √(‖W₁‖²_F + ‖W₂‖²_F + ‖W₃‖²_F)","eb")
        w("","body")
        w("  Δ_mem := (‖Θ_base‖ − ‖Θ_opt‖) / ‖Θ_base‖ × 100%","eb")
        w("  Δ_mem > 0  ⇒  compresión exitosa","note")
        w("","body")

        w("§ 9.  INICIALIZACIÓN DE XAVIER (Glorot)","h2"); div()
        w("  W_k ~ 𝒩(0, σ²_k),   σ_k = √(2/n_{k-1})","eb")
        w("  Mantiene varianza de activaciones constante por capa.","note")
        w("","body")

        sep()
        w("§ 10.  FLUJO COMPUTACIONAL COMPLETO","h2")
        sep()
        w("  ┌────────────────────────────────────────────────────────────┐","eq")
        w("  │ 1. Generar 𝒟, normalizar X                               │","eq")
        w("  │ 2. Inicializar Θ con Xavier                               │","eq")
        w("  │ 3. [Opt. NL] L-BFGS-B → (λ*,α*)                         │","eq")
        w("  │ 4. Entrenar Red Base:  Adam(λ⁰,α⁰) + Early Stopping      │","eq")
        w("  │ 5. Entrenar Red Opt.:  Adam(λ*,α*) + Early Stopping       │","eq")
        w("  │ 6. Comparar ‖Θ_base‖ vs ‖Θ_opt‖  →  Δ_mem               │","eq")
        w("  │ 7. Interpretador automático → Diagnóstico + Veredicto     │","eq")
        w("  └────────────────────────────────────────────────────────────┘","eq")
        w("","body")
        sep()
        T.config(state="disabled"); T.see("1.0")

    # ══════════════════════════════════════════════════════════════════════════
    #  MLP MINI (sidebar)
    # ══════════════════════════════════════════════════════════════════════════
    def _draw_mlp_mini(self):
        c=self.mlp_mini; c.delete("all")
        W,H=290,160
        h1s=min(self.v_h1.get(),7); h2s=min(self.v_h2.get(),5)
        layers=[2,h1s,h2s,1]
        cols=[ACCENT2,ACCENT3,ACCENT5,ACCENT]
        xs=[30,105,185,262]
        for n,x in zip(layers,xs):
            gap=(H-20)/(n+1)
            ys=[10+gap*(j+1) for j in range(n)]
            # Conexiones (solo al siguiente)
        pos=[]
        for n,x in zip(layers,xs):
            gap=(H-20)/(n+1)
            pos.append([(x,10+gap*(j+1)) for j in range(n)])
        for li in range(3):
            for (x1,y1) in pos[li]:
                for (x2,y2) in pos[li+1]:
                    c.create_line(x1,y1,x2,y2,fill="#18184a",width=1)
        for li,(nodes,col) in enumerate(zip(pos,cols)):
            r=9 if li in(1,2) else 11
            for (x,y) in nodes:
                c.create_oval(x-r,y-r,x+r,y+r,fill=col,outline="#aaaacc",width=1)
        # Indicadores
        names=["In","h1","h2","Out"]
        vals=[2,self.v_h1.get(),self.v_h2.get(),1]
        for x,nm,v,col in zip(xs,names,vals,cols):
            c.create_text(x,H-6,text=f"{nm}({v})",fill=col,font=("Consolas",7))

    # ══════════════════════════════════════════════════════════════════════════
    #  LÓGICA OPTIMIZACIÓN
    # ══════════════════════════════════════════════════════════════════════════
    def _run_opt(self):
        if self.running: return
        self.running=True; self._btns("disabled")
        self.lbl_status.config(
            text="⚙ Optimizando (λ,α) con L-BFGS-B…\n~15-30 s")
        threading.Thread(target=self._opt_thread,daemon=True).start()

    def _opt_thread(self):
        try:
            mo=MemOptimizer(self.X,self.y,
                            h1=self.v_h1.get(),h2=self.v_h2.get(),
                            act=self.v_act.get(),
                            gamma=self.v_gamma.get(), pretrain=200)
            ls,ls2=mo.run(x0=[self.v_lam.get(),self.v_lr.get()])
            self.opt_lam=ls; self.opt_lr=ls2
            self.opt_iters=mo.iterations; self.opt_done=True
            self.mvars["opt_lr"].set(f"{ls2:.5f}")
            self.mvars["opt_lam"].set(f"{ls:.6f}")
            self.lbl_status.config(
                text=f"✔ Opt. lista.\nα*={ls2:.5f}\nλ*={ls:.6f}\n"
                     f"({len(mo.iterations)} eval.)\n▶ Entrena ahora.")
            self.after(0, self._update_opt_tab)
        except Exception as e:
            self.lbl_status.config(text=f"Error opt.:\n{e}")
        finally:
            self.running=False; self._btns("normal")

    # ══════════════════════════════════════════════════════════════════════════
    #  LÓGICA ENTRENAMIENTO
    # ══════════════════════════════════════════════════════════════════════════
    def _run_train(self):
        if self.running: return
        self.running=True; self._btns("disabled")
        self.lbl_status.config(text="▶ Entrenando…")
        threading.Thread(target=self._train_thread,daemon=True).start()

    def _train_thread(self):
        try:
            ep   = self.v_epochs.get()
            h1   = self.v_h1.get(); h2=self.v_h2.get()
            act  = self.v_act.get()
            pat  = self.v_patience.get()
            lr_b = self.v_lr.get(); lam_b=self.v_lam.get()
            lr_o = self.opt_lr  if self.opt_done else lr_b
            lam_o= self.opt_lam if self.opt_done else lam_b

            def cb(e,total,l,acc,nrm,label):
                pct=int(e/total*38)
                bar="█"*pct+"░"*(38-pct)
                self.lbl_status.config(
                    text=f"{label}\n[{bar}]\nÉp {e}/{total}\n"
                         f"Pérd={l:.4f} Acc={acc*100:.1f}%")

            self.net_base=MLP(n0=2,h1=h1,h2=h2,act=act,lr=lr_b,lam=lam_b,seed=1)
            ep_b=self.net_base.train(self.X,self.y,ep,patience=pat,
                cb=lambda e,t,l,a,n: cb(e,t,l,a,n,"Entrenando Base…"))

            self.net_opt=MLP(n0=2,h1=h1,h2=h2,act=act,lr=lr_o,lam=lam_o,seed=1)
            ep_o=self.net_opt.train(self.X,self.y,ep,patience=pat,
                cb=lambda e,t,l,a,n: cb(e,t,l,a,n,"Entrenando Opt.…"))

            nb=self.net_base; no=self.net_opt
            lb=nb.loss_h[-1]; lo=no.loss_h[-1]
            ab=nb.acc_h[-1];  ao=no.acc_h[-1]
            nb_n=nb.norm_h[-1]; no_n=no.norm_h[-1]
            red=(nb_n-no_n)/nb_n*100 if nb_n>1e-9 else 0

            self.mvars["loss_b"].set(f"{lb:.4f}")
            self.mvars["loss_o"].set(f"{lo:.4f}")
            self.mvars["acc_b"].set(f"{ab*100:.1f}%")
            self.mvars["acc_o"].set(f"{ao*100:.1f}%")
            self.mvars["norm_b"].set(f"{nb_n:.3f}")
            self.mvars["norm_o"].set(f"{no_n:.3f}")
            self.mvars["red"].set(f"{red:.1f}%")
            self.mvars["ep_b"].set(str(ep_b))
            self.mvars["ep_o"].set(str(ep_o))

            self.lbl_status.config(
                text=f"✔ Listo.\n"
                     f"Base:  {ab*100:.1f}% ({ep_b} ép.)\n"
                     f"Opt.:  {ao*100:.1f}% ({ep_o} ép.)\n"
                     f"Δ‖W‖: {red:.1f}%")

            msgs=Interpreter.analizar(nb,no,
                self.opt_lam or lam_b, self.opt_lr or lr_b,
                self.v_gamma.get(), self.opt_iters)

            self.after(0, self._update_plots)
            self.after(0, lambda: self._update_interp(msgs))
            self.after(0, self._draw_mlp_mini)
            self.after(50, self._redraw_mlp_tab)
        except Exception as e:
            import traceback; traceback.print_exc()
            self.lbl_status.config(text=f"Error:\n{e}")
        finally:
            self.running=False; self._btns("normal")

    # ══════════════════════════════════════════════════════════════════════════
    #  ACTUALIZACIÓN DE GRÁFICAS
    # ══════════════════════════════════════════════════════════════════════════
    def _update_plots(self):
        self._update_train_tab()
        self._update_boundary_tab()

    def _update_train_tab(self):
        nb=self.net_base; no=self.net_opt
        rec=5; xs=np.arange(len(nb.loss_h))*rec
        for ax,(db,do),title,ylabel in [
            (self.ax_loss,(nb.loss_h,no.loss_h),"Pérdida (BCE+L2)","Pérdida"),
            (self.ax_norm,(nb.norm_h,no.norm_h),"Norma ‖W‖ — Memoria","‖W‖"),
            (self.ax_acc,([a*100 for a in nb.acc_h],[a*100 for a in no.acc_h]),
             "Precisión de Clasificación","Accuracy (%)"),
        ]:
            ax.cla(); ax.set_facecolor(PANEL_BG)
            ax.plot(xs,db,color=ACCENT2,lw=1.6,label="Base")
            ax.plot(xs,do,color=ACCENT3,lw=1.6,linestyle="--",label="Opt.")
            if ax is self.ax_norm:
                ax.fill_between(xs,db,do,alpha=0.15,color=ACCENT3,label="Δmem")
            # Línea de 90% de precisión
            if ax is self.ax_acc:
                ax.axhline(90,color=ACCENT4,lw=0.8,linestyle=":",alpha=0.7)
                ax.text(xs[-1]*0.02,91,"90%",color=ACCENT4,fontsize=7)
            ax.set_title(title,fontsize=8,fontweight="bold",color=ACCENT2,pad=4)
            ax.set_xlabel("Época",fontsize=7,color=TEXT)
            ax.set_ylabel(ylabel,fontsize=7,color=TEXT)
            ax.legend(fontsize=7,facecolor=DARK_BG,edgecolor=CARD_BG,labelcolor=TEXT)
            self._sax(ax)
        self.cv1.draw()

    def _update_boundary_tab(self):
        x0,x1=self.X[:,0].min()-.3,self.X[:,0].max()+.3
        y0,y1=self.X[:,1].min()-.3,self.X[:,1].max()+.3
        xx,yy=np.meshgrid(np.linspace(x0,x1,220),np.linspace(y0,y1,220))
        grid=np.c_[xx.ravel(),yy.ravel()]
        for ax,net,title in [
            (self.ax_bd_b,self.net_base,"Base"),
            (self.ax_bd_o,self.net_opt, "Optimizada"),
        ]:
            ax.cla(); ax.set_facecolor(PANEL_BG)
            Z=net.forward(grid).reshape(xx.shape)
            ax.contourf(xx,yy,Z,levels=60,cmap="RdBu_r",alpha=0.65)
            ax.contour(xx,yy,Z,levels=[.5],colors=[ACCENT3],linewidths=2)
            ax.scatter(self.X[:,0],self.X[:,1],c=self.y.ravel(),
                       cmap="coolwarm",s=9,alpha=0.8,edgecolors="none")
            l=net.loss_h[-1]; a=net.acc_h[-1]
            ds=self.v_dataset.get()
            ax.set_title(f"Frontera — Red {title} [{ds}]\n"
                         f"Pérdida={l:.4f}   Acc={a*100:.1f}%",
                         fontsize=8,fontweight="bold",color=ACCENT2,pad=3)
            ax.set_xlabel("x₁",fontsize=7,color=TEXT)
            ax.set_ylabel("x₂",fontsize=7,color=TEXT)
            self._sax(ax)
        self.cv2.draw()

    def _update_opt_tab(self):
        if not self.opt_iters: return
        its  = self.opt_iters
        ns   = list(range(1, len(its)+1))
        fv   = np.array([it["f"]    for it in its])
        lms  = np.array([it["lam"]  for it in its])
        lrs  = np.array([it["lr"]   for it in its])
        bces = np.array([it["bce"]  for it in its])
        nrms = np.array([it["norm"] for it in its])
        n    = len(ns)

        # Mejora relativa acumulada respecto al valor inicial
        f0   = fv[0] if fv[0] != 0 else 1e-9
        mejora_rel = (f0 - fv) / abs(f0) * 100   # % de reducción acumulada

        # ── Paleta de colores para la trayectoria (degradado por iteración) ──
        cmap_traj = plt.get_cmap("cool")
        traj_cols = [cmap_traj(i/(max(n-1,1))) for i in range(n)]

        # ── helpers internos ─────────────────────────────────────────────────
        def annotate_opt(ax, x_opt, y_opt, label="Óptimo"):
            """Marca el punto óptimo con estrella y anotación."""
            ax.scatter([x_opt], [y_opt], color=ACCENT3, s=180,
                       marker="*", zorder=8, linewidths=0)
            ax.annotate(label,
                xy=(x_opt, y_opt),
                xytext=(8, 8), textcoords="offset points",
                color=ACCENT3, fontsize=7, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT3,
                                lw=0.8))

        def style_opt_ax(ax, title, xlabel, ylabel):
            ax.set_title(title, fontsize=8, fontweight="bold",
                         color=ACCENT2, pad=5)
            ax.set_xlabel(xlabel, fontsize=7, color=TEXT)
            ax.set_ylabel(ylabel, fontsize=7, color=TEXT)
            self._sax(ax)

        # ════════════════════════════════════════════════════════════════════
        #  SUBPLOT 1 — Convergencia de f(λ,α)
        # ════════════════════════════════════════════════════════════════════
        ax = self.ax_fv
        ax.cla(); ax.set_facecolor(PANEL_BG)

        # Área rellena bajo la curva
        ax.fill_between(ns, fv, fv.min(),
                        alpha=0.18, color=ACCENT4)
        # Línea principal con gradiente de color por tramos
        for i in range(n-1):
            ax.plot(ns[i:i+2], fv[i:i+2],
                    color=traj_cols[i], lw=2.2, solid_capstyle="round")
        # Puntos individuales
        ax.scatter(ns, fv, c=ns, cmap="cool", s=28,
                   zorder=5, edgecolors="white", linewidths=0.5)
        # Línea horizontal en mínimo
        ax.axhline(fv.min(), color=ACCENT3, lw=0.9,
                   linestyle="--", alpha=0.7)
        ax.text(ns[0]+0.1, fv.min()*1.005,
                f"mín={fv.min():.4f}",
                color=ACCENT3, fontsize=7)
        # Marcador de óptimo
        i_opt = int(np.argmin(fv))
        ax.scatter([ns[i_opt]], [fv[i_opt]], color=ACCENT3,
                   s=120, marker="*", zorder=9, linewidths=0)
        style_opt_ax(ax,
            f"Convergencia de f(λ,α)  [inicio→{fv[0]:.3f}, fin→{fv[-1]:.3f}]",
            "Evaluación", "f(λ, α)")
        ax.set_xlim(0.5, n+0.5)

        # ════════════════════════════════════════════════════════════════════
        #  SUBPLOT 2 — Trayectoria de λ
        # ════════════════════════════════════════════════════════════════════
        ax = self.ax_lm
        ax.cla(); ax.set_facecolor(PANEL_BG)

        for i in range(n-1):
            ax.plot(ns[i:i+2], lms[i:i+2],
                    color=traj_cols[i], lw=2.2, solid_capstyle="round")
        ax.scatter(ns, lms, c=fv, cmap="plasma",
                   s=30, zorder=5, edgecolors="white", linewidths=0.4)
        # Línea vertical en el óptimo
        ax.axvline(ns[i_opt], color=ACCENT, lw=0.8, linestyle=":")
        if self.opt_lam is not None:
            ax.axhline(self.opt_lam, color=ACCENT3,
                       lw=0.9, linestyle="--", alpha=0.8)
            ax.text(ns[0]+0.1, self.opt_lam,
                    f" λ*={self.opt_lam:.5f}",
                    color=ACCENT3, fontsize=7, va="bottom")
        style_opt_ax(ax,
            f"Trayectoria de λ  [λ*={self.opt_lam:.5f}]"
            if self.opt_lam else "Trayectoria de λ",
            "Evaluación", "λ  (regularización L2)")
        ax.set_xlim(0.5, n+0.5)
        # Eje y con notación científica si los valores son muy pequeños
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v,_: f"{v:.2e}" if abs(v)<0.001 else f"{v:.4f}"))

        # ════════════════════════════════════════════════════════════════════
        #  SUBPLOT 3 — Trayectoria de α (LR)
        # ════════════════════════════════════════════════════════════════════
        ax = self.ax_lr2
        ax.cla(); ax.set_facecolor(PANEL_BG)

        for i in range(n-1):
            ax.plot(ns[i:i+2], lrs[i:i+2],
                    color=traj_cols[i], lw=2.2, solid_capstyle="round")
        ax.scatter(ns, lrs, c=fv, cmap="plasma",
                   s=30, zorder=5, edgecolors="white", linewidths=0.4)
        ax.axvline(ns[i_opt], color=ACCENT, lw=0.8, linestyle=":")
        if self.opt_lr is not None:
            ax.axhline(self.opt_lr, color=ACCENT3,
                       lw=0.9, linestyle="--", alpha=0.8)
            ax.text(ns[0]+0.1, self.opt_lr,
                    f" α*={self.opt_lr:.5f}",
                    color=ACCENT3, fontsize=7, va="bottom")
        style_opt_ax(ax,
            f"Trayectoria de α (LR)  [α*={self.opt_lr:.5f}]"
            if self.opt_lr else "Trayectoria de α (LR)",
            "Evaluación", "α  (learning rate)")
        ax.set_xlim(0.5, n+0.5)
        # Forzar eje y positivo (evita negativos por artefactos numéricos)
        y_lo = max(0, lrs.min() * 0.8)
        y_hi = lrs.max() * 1.25 + 1e-8
        ax.set_ylim(y_lo, y_hi)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v,_: f"{v:.2e}" if abs(v)<0.001 else f"{v:.4f}"))

        # ════════════════════════════════════════════════════════════════════
        #  SUBPLOT 4 — Mapa de calor (λ, α) → f
        # ════════════════════════════════════════════════════════════════════
        ax = self.ax_sc
        ax.cla(); ax.set_facecolor(PANEL_BG)

        if n >= 3:
            # Scatter principal: tamaño proporcional a 1/f (mejores = más grandes)
            sizes = 40 + 200 * (fv.max() - fv) / (fv.max() - fv.min() + 1e-9)
            sc = ax.scatter(lms, lrs, c=fv, cmap="plasma_r",
                            s=sizes, zorder=4,
                            edgecolors="white", linewidths=0.4,
                            alpha=0.85)
            # Flecha de trayectoria (primero → último)
            for i in range(min(n-1, 12)):
                ax.annotate("",
                    xy=(lms[i+1], lrs[i+1]),
                    xytext=(lms[i], lrs[i]),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="#aaaadd",
                        lw=0.7, alpha=0.5))
            # Punto inicial
            ax.scatter([lms[0]], [lrs[0]], color=ACCENT4,
                       s=100, marker="o", zorder=7,
                       edgecolors="white", linewidths=0.8,
                       label="Inicio")
            # Punto óptimo
            if self.opt_lam is not None and self.opt_lr is not None:
                annotate_opt(ax, self.opt_lam, self.opt_lr, f"λ*={self.opt_lam:.4f}\nα*={self.opt_lr:.4f}")
            # Colorbar compacta y legible
            cb = self.fig3.colorbar(sc, ax=ax, shrink=0.75,
                                    pad=0.03, aspect=20)
            cb.set_label("f(λ,α)", color=TEXT, fontsize=7)
            cb.ax.yaxis.set_tick_params(colors=TEXT, labelsize=6)
            ax.legend(fontsize=7, facecolor=DARK_BG,
                      edgecolor=CARD_BG, labelcolor=TEXT,
                      loc="lower right")
        # Ejes con formato limpio
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda v,_: f"{v:.2e}" if abs(v)<0.001 else f"{v:.4f}"))
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v,_: f"{v:.2e}" if abs(v)<0.001 else f"{v:.4f}"))
        style_opt_ax(ax,
            "Espacio (λ, α) → f   [tamaño ∝ calidad]",
            "λ  (regularización)", "α  (learning rate)")

        # ════════════════════════════════════════════════════════════════════
        #  SUBPLOT 5 — Mejora relativa acumulada %
        # ════════════════════════════════════════════════════════════════════
        ax = self.ax_grad
        ax.cla(); ax.set_facecolor(PANEL_BG)

        # Barras de mejora acumulada
        bar_cols = [ACCENT3 if m >= 0 else ACCENT for m in mejora_rel]
        bars = ax.bar(ns, mejora_rel, color=bar_cols,
                      width=0.6, alpha=0.85, edgecolor="none")
        # Línea suavizada (media móvil)
        if n >= 4:
            window = max(2, n//5)
            smooth = np.convolve(mejora_rel,
                                 np.ones(window)/window, mode="valid")
            xs_s   = np.linspace(ns[window-1], ns[-1], len(smooth))
            ax.plot(xs_s, smooth, color=ACCENT2, lw=1.8,
                    linestyle="-", alpha=0.9, label="Tendencia")
        # Línea 0
        ax.axhline(0, color=MUTED, lw=0.7, alpha=0.5)
        # Etiqueta del máximo
        i_max = int(np.argmax(mejora_rel))
        ax.text(ns[i_max], mejora_rel[i_max]+0.5,
                f"+{mejora_rel[i_max]:.1f}%",
                ha="center", va="bottom",
                color=ACCENT3, fontsize=7, fontweight="bold")
        ax.legend(fontsize=7, facecolor=DARK_BG,
                  edgecolor=CARD_BG, labelcolor=TEXT)
        style_opt_ax(ax,
            f"Mejora relativa acumulada  [total={mejora_rel[-1]:.1f}%]",
            "Evaluación", "Reducción de f  (%)")
        ax.set_xlim(0.5, n+0.5)

        # ════════════════════════════════════════════════════════════════════
        #  SUBPLOT 6 — BCE vs ‖W‖² (trade-off memoria/rendimiento)
        # ════════════════════════════════════════════════════════════════════
        ax = self.ax_bce
        ax.cla(); ax.set_facecolor(PANEL_BG)

        # Scatter: cada punto = una evaluación del optimizador
        sc2 = ax.scatter(nrms, bces, c=ns, cmap="viridis",
                         s=50, zorder=5,
                         edgecolors="white", linewidths=0.4,
                         alpha=0.9)
        # Línea de trayectoria
        ax.plot(nrms, bces, color="#3a3a7a",
                lw=0.8, alpha=0.5, zorder=3)
        # Punto inicial
        ax.scatter([nrms[0]], [bces[0]], color=ACCENT4,
                   s=100, marker="o", zorder=7,
                   edgecolors="white", linewidths=0.8,
                   label="Inicio")
        # Punto óptimo (mínimo f = mínimo BCE+γ‖W‖²)
        if self.opt_lam is not None and self.opt_lr is not None:
            ax.scatter([nrms[i_opt]], [bces[i_opt]],
                       color=ACCENT3, s=160,
                       marker="*", zorder=9,
                       label=f"Óptimo (eval {ns[i_opt]})")
        # Colorbar compacta
        cb2 = self.fig3.colorbar(sc2, ax=ax, shrink=0.75,
                                  pad=0.03, aspect=20)
        cb2.set_label("Evaluación", color=TEXT, fontsize=7)
        cb2.ax.yaxis.set_tick_params(colors=TEXT, labelsize=6)
        # Anotación del compromiso
        ax.text(0.04, 0.96,
            "← menos memoria\n↓ menos pérdida\n★ óptimo",
            transform=ax.transAxes, fontsize=7,
            color=MUTED, va="top", linespacing=1.6)
        ax.legend(fontsize=7, facecolor=DARK_BG,
                  edgecolor=CARD_BG, labelcolor=TEXT,
                  loc="upper right")
        style_opt_ax(ax,
            "Trade-off  BCE vs ‖W‖²  (memoria vs rendimiento)",
            "‖W‖²  (memoria implícita)", "BCE  (pérdida)")

        self.cv3.draw()

    # ══════════════════════════════════════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════════════════════════════════════
    def _sax(self, ax):
        ax.tick_params(colors=TEXT,labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor(CARD_BG)
        ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
        ax.grid(True,color=CARD_BG,lw=0.5,alpha=0.6)

    def _placeholder(self, axes):
        for ax in axes:
            ax.text(0.5,0.5,"Sin datos.\nEntrena para visualizar.",
                    transform=ax.transAxes,ha="center",va="center",
                    color=MUTED,fontsize=9)

    def _btns(self, st):
        for b in [self.btn_opt,self.btn_train,self.btn_reset]:
            b.config(state=st)

    def _reset_plots_only(self):
        for ax in [self.ax_loss, self.ax_norm, self.ax_acc] + self._opt_axes:
            ax.cla(); ax.set_facecolor(PANEL_BG)
        self._placeholder([self.ax_loss, self.ax_norm, self.ax_acc] + self._opt_axes)
        self.cv1.draw(); self.cv3.draw()

    def _reset(self):
        self.net_base=None; self.net_opt=None
        self.opt_done=False; self.opt_lam=None; self.opt_lr=None
        self.opt_iters=[]
        for v in self.mvars.values(): v.set("—")
        self.lbl_status.config(text="Reiniciado. Listo.")
        self._reset_plots_only()
        self._reload_data()
        self._init_boundary()
        self._interp_placeholder()
        self._draw_mlp_mini()
        self.after(50, self._redraw_mlp_tab)


# ══════════════════════════════════════════════════════════════════════════════
#  GENERADORES DE REPORTES PDF
# ══════════════════════════════════════════════════════════════════════════════

    # ─── captura de figuras ──────────────────────────────────────────────────
    def _fig_to_png_bytes(self, fig):
        """Renderiza una figura matplotlib a bytes PNG en memoria."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        buf.seek(0)
        return buf

    # ─── utilidades de estilo PDF ────────────────────────────────────────────
    def _pdf_styles(self):
        """Devuelve diccionario de ParagraphStyle para los reportes."""
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.colors import HexColor
        from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfbase import pdfmetrics

        # Registrar DejaVu Sans (soporta Unicode completo: griego, math, flechas, etc.)
        _DJVU_PATH  = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        _DJVU_BOLD  = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
        _DJVU_MONO  = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
        _DJVU_MONOB = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf'
        try:
            if 'DJV' not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont('DJV',  _DJVU_PATH))
                pdfmetrics.registerFont(TTFont('DJVB', _DJVU_BOLD))
                pdfmetrics.registerFont(TTFont('DJVM', _DJVU_MONO))
                pdfmetrics.registerFont(TTFont('DJVMB', _DJVU_MONOB))
        except Exception:
            pass  # Fallback a Helvetica si no hay DejaVu

        _F_BODY = 'DJV'  if 'DJV'  in pdfmetrics.getRegisteredFontNames() else 'Helvetica'
        _F_BOLD = 'DJVB' if 'DJVB' in pdfmetrics.getRegisteredFontNames() else 'Helvetica-Bold'
        _F_MONO = 'DJVM' if 'DJVM' in pdfmetrics.getRegisteredFontNames() else 'Courier'

        C_BG    = HexColor("#0d0d1a")
        C_TEXT  = HexColor("#e8e8f0")
        C_MUTED = HexColor("#9090b8")
        C_ACC1  = HexColor("#00c9e0")
        C_ACC2  = HexColor("#7bed9f")
        C_ACC3  = HexColor("#ff4d6d")
        C_ACC4  = HexColor("#ffa94d")
        C_MATH  = HexColor("#c77dff")
        C_CARD  = HexColor("#0c2d5a")
        C_GOLD  = HexColor("#e8c84a")

        base = ParagraphStyle('base', fontName=_F_BODY,
                              fontSize=9.5, leading=14,
                              textColor=C_TEXT, backColor=C_BG)

        s = {}
        s['cover_title'] = ParagraphStyle('cover_title', parent=base,
            fontName=_F_BOLD, fontSize=26, leading=32,
            textColor=C_ACC1, alignment=TA_CENTER, spaceAfter=6)
        s['cover_sub'] = ParagraphStyle('cover_sub', parent=base,
            fontSize=12, leading=17, textColor=C_MUTED,
            alignment=TA_CENTER, spaceAfter=4)
        s['cover_meta'] = ParagraphStyle('cover_meta', parent=base,
            fontSize=9, leading=13, textColor=C_MUTED,
            alignment=TA_CENTER, spaceAfter=2)
        s['h1'] = ParagraphStyle('h1', parent=base,
            fontName=_F_BOLD, fontSize=15, leading=20,
            textColor=C_ACC1, spaceBefore=18, spaceAfter=6,
            borderPad=4, backColor=HexColor("#0a1a2a"),
            borderColor=C_ACC1, borderWidth=0, leftIndent=0)
        s['h2'] = ParagraphStyle('h2', parent=base,
            fontName=_F_BOLD, fontSize=12, leading=16,
            textColor=C_ACC2, spaceBefore=12, spaceAfter=4)
        s['h3'] = ParagraphStyle('h3', parent=base,
            fontName=_F_BOLD, fontSize=10, leading=14,
            textColor=C_ACC4, spaceBefore=8, spaceAfter=3)
        s['body'] = ParagraphStyle('body', parent=base,
            fontSize=9.5, leading=14.5, textColor=C_TEXT,
            alignment=TA_JUSTIFY, spaceAfter=6, leftIndent=6)
        s['math'] = ParagraphStyle('math', parent=base,
            fontName=_F_MONO, fontSize=9, leading=14,
            textColor=C_MATH, leftIndent=24, spaceAfter=4,
            backColor=HexColor("#0a0a20"))
        s['math_label'] = ParagraphStyle('math_label', parent=base,
            fontName=_F_BOLD, fontSize=8.5, leading=12,
            textColor=C_GOLD, leftIndent=6, spaceAfter=2)
        s['caption'] = ParagraphStyle('caption', parent=base,
            fontSize=8, leading=12, textColor=C_MUTED,
            alignment=TA_CENTER, spaceAfter=10)
        s['def_box'] = ParagraphStyle('def_box', parent=base,
            fontSize=9, leading=13.5, textColor=C_TEXT,
            leftIndent=14, rightIndent=10,
            backColor=HexColor("#0c1a30"), spaceAfter=6)
        s['insight'] = ParagraphStyle('insight', parent=base,
            fontSize=9, leading=13.5, textColor=HexColor("#c8f5a0"),
            leftIndent=14, rightIndent=10,
            backColor=HexColor("#071a0f"), spaceAfter=6)
        s['warning'] = ParagraphStyle('warning', parent=base,
            fontSize=9, leading=13.5, textColor=HexColor("#ffd0a0"),
            leftIndent=14, rightIndent=10,
            backColor=HexColor("#1a0e00"), spaceAfter=6)
        s['bullet'] = ParagraphStyle('bullet', parent=base,
            fontSize=9.5, leading=13.5, textColor=C_TEXT,
            leftIndent=18, spaceAfter=3, firstLineIndent=-10)
        s['kpi_val'] = ParagraphStyle('kpi_val', parent=base,
            fontName=_F_BOLD, fontSize=18, leading=22,
            textColor=C_ACC2, alignment=TA_CENTER)
        s['kpi_lbl'] = ParagraphStyle('kpi_lbl', parent=base,
            fontSize=8, leading=11, textColor=C_MUTED,
            alignment=TA_CENTER)
        s['ref'] = ParagraphStyle('ref', parent=base,
            fontSize=7.5, leading=11, textColor=C_MUTED, leftIndent=6)
        return s, {
            'BG': C_BG, 'TEXT': C_TEXT, 'MUTED': C_MUTED,
            'ACC1': C_ACC1, 'ACC2': C_ACC2, 'ACC3': C_ACC3,
            'ACC4': C_ACC4, 'MATH': C_MATH, 'CARD': C_CARD, 'GOLD': C_GOLD
        }

    def _pdf_header_footer(self, canvas_obj, doc, title, subtitle=""):
        """Encabezado y pie de página para todos los reportes."""
        from reportlab.lib.colors import HexColor
        from reportlab.lib.units import cm
        w, h = doc.pagesize
        canvas_obj.saveState()

        # Franja superior
        canvas_obj.setFillColor(HexColor("#0a1a30"))
        canvas_obj.rect(0, h - 28, w, 28, fill=1, stroke=0)
        canvas_obj.setFillColor(HexColor("#00c9e0"))
        canvas_obj.rect(0, h - 30, w, 2, fill=1, stroke=0)
        canvas_obj.setFont('Helvetica-Bold', 9)
        canvas_obj.setFillColor(HexColor("#00c9e0"))
        canvas_obj.drawString(36, h - 20, title)
        canvas_obj.setFont('Helvetica', 8)
        canvas_obj.setFillColor(HexColor("#7070a0"))
        canvas_obj.drawRightString(w - 36, h - 20, subtitle)

        # Franja inferior
        canvas_obj.setFillColor(HexColor("#0a1a30"))
        canvas_obj.rect(0, 0, w, 26, fill=1, stroke=0)
        canvas_obj.setFillColor(HexColor("#00c9e0"))
        canvas_obj.rect(0, 26, w, 1.5, fill=1, stroke=0)
        canvas_obj.setFont('Helvetica', 7.5)
        canvas_obj.setFillColor(HexColor("#7070a0"))
        canvas_obj.drawString(36, 10,
            f"Generado por KEVIN ORTEGA — {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}")
        canvas_obj.drawRightString(w - 36, 10, f"Página {doc.page}")

        canvas_obj.restoreState()

    # ═════════════════════════════════════════════════════════════════════════
    #  REPORTE INTELIGENTE PDF
    # ═════════════════════════════════════════════════════════════════════════
    def _export_report_pdf(self):
        """Genera el Reporte Inteligente con análisis variacional + imágenes."""
        try:
            self._do_export_report_pdf()
        except Exception as _e:
            import traceback as _tb
            _msg = _tb.format_exc()
            print("[REPORT PDF ERROR]\n" + _msg)
            try:
                self.lbl_status.config(text=f"Error: {str(_e)[:90]}")
                messagebox.showerror("Error al generar Reporte PDF",
                    f"{str(_e)}\n\nRevisa la consola para el detalle completo.")
            except Exception:
                pass

    def _do_export_report_pdf(self):
        """Cuerpo real del generador — llamado desde _export_report_pdf."""
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, PageBreak,
            Table, TableStyle, Image as RLImage, HRFlowable
        )
        from reportlab.lib.units import cm
        from reportlab.lib.colors import HexColor

        if self.net_base is None:
            messagebox.showwarning("Sin datos",
                "Entrena la red primero para generar el reporte.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")],
            initialfile=f"reporte_inteligente_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            title="Guardar Reporte Inteligente PDF"
        )
        if not path:
            return

        self.lbl_status.config(text="Generando Reporte PDF...")
        self.update_idletasks()

        W, H = A4
        sty, col = self._pdf_styles()

        def hf_report(c, d):
            self._pdf_header_footer(c, d,
                "Reporte Inteligente — Red Neuronal + Optimizador No Lineal",
                f"Dataset: {self.v_dataset.get()}  |  Activación: {self.v_act.get()}")

        doc = SimpleDocTemplate(
            path, pagesize=A4,
            title="Reporte Inteligente — MLP + Optimizador No Lineal",
            author="Perplexity Computer",
            leftMargin=2.2*cm, rightMargin=2.2*cm,
            topMargin=1.8*cm, bottomMargin=1.8*cm
        )

        story = []

        # ── PORTADA ──────────────────────────────────────────────────────────
        story.append(Spacer(1, 1.8*cm))
        story.append(Paragraph(
            "REPORTE INTELIGENTE", sty['cover_title']))
        story.append(Paragraph(
            "Red Neuronal Profunda con Optimización No Lineal de Memoria",
            sty['cover_sub']))
        story.append(Spacer(1, 0.4*cm))
        story.append(HRFlowable(width="100%", thickness=1.5,
                                color=HexColor("#00c9e0"), spaceAfter=12))
        story.append(Spacer(1, 0.3*cm))

        # Metadata card
        now_str = datetime.datetime.now().strftime("%d de %B de %Y, %H:%M")
        ds = self.v_dataset.get()
        act = self.v_act.get()
        h1 = int(self.v_h1.get()); h2 = int(self.v_h2.get())
        epochs = int(self.v_epochs.get())
        lr_base = self.v_lr.get(); lam_base = self.v_lam.get()

        meta_data = [
            ["Parámetro", "Valor"],
            ["Fecha / Hora", now_str],
            ["Dataset", ds],
            ["Función de activación", act],
            ["Arquitectura", f"MLP  2→{h1}→{h2}→1 (3 capas)"],
            ["Épocas configuradas", str(epochs)],
            ["Learning rate base", f"{lr_base:.5f}"],
            ["λ base (regularización)", f"{lam_base:.6f}"],
            ["γ (peso memoria)", f"{self.v_gamma.get():.3f}"],
        ]
        if self.net_base:
            meta_data.append(["Precisión base", f"{self.net_base.best_acc*100:.2f} %"])
            meta_data.append(["Épocas reales base", str(self.net_base.epoch_count)])
        if self.net_opt:
            meta_data.append(["Precisión optimizada", f"{self.net_opt.best_acc*100:.2f} %"])
        if self.opt_lam is not None:
            meta_data.append(["λ* óptimo", f"{self.opt_lam:.6f}"])
            meta_data.append(["α* óptimo", f"{self.opt_lr:.6f}"])

        tbl_meta = Table(meta_data, colWidths=[5.5*cm, 9.5*cm])
        tbl_meta.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#0c2d5a")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), HexColor("#00c9e0")),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 8.5),
            ("BACKGROUND", (0, 1), (0, -1), HexColor("#0a1a30")),
            ("TEXTCOLOR",  (0, 1), (0, -1), HexColor("#9090b8")),
            ("TEXTCOLOR",  (1, 1), (1, -1), HexColor("#e8e8f0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [HexColor("#0d0d1a"), HexColor("#0f0f22")]),
            ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#0c2d5a")),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ]))
        story.append(tbl_meta)
        story.append(Spacer(1, 0.5*cm))
        story.append(HRFlowable(width="100%", thickness=0.6,
                                color=HexColor("#0c2d5a"), spaceAfter=0))
        story.append(PageBreak())

        # ── §1  PLANTEAMIENTO DEL PROBLEMA ────────────────────────────────────
        story.append(Paragraph("§1  Planteamiento del Problema", sty['h1']))
        story.append(Paragraph(
            "Las redes neuronales profundas (Deep MLP) son aproximadores universales de funciones: "
            "dada una arquitectura con capas ocultas suficientes, pueden aprender cualquier mapeo "
            "continuo entre entradas y salidas. Sin embargo, su entrenamiento por descenso de "
            "gradiente no garantiza utilización eficiente de los pesos (parámetros) almacenados "
            "en memoria. Redes sobreparametrizadas tienden a memorizar el conjunto de entrenamiento "
            "(sobreajuste) con normas de pesos ‖W‖ excesivamente grandes, lo que perjudica la "
            "generalización y el coste computacional de inferencia.",
            sty['body']))
        story.append(Paragraph(
            "Este módulo aborda el problema desde dos frentes simultáneos:",
            sty['body']))
        for txt in [
            "<b>Frente 1 — Clasificación:</b> Entrenar una MLP con Adam y early stopping para "
            "maximizar la precisión de clasificación binaria en el conjunto de datos seleccionado.",
            "<b>Frente 2 — Eficiencia de memoria:</b> Aplicar un optimizador no lineal (L-BFGS-B) "
            "para encontrar los hiperparámetros (λ*, α*) que minimizan simultáneamente la pérdida "
            "de clasificación y la norma cuadrática de los pesos, logrando un compromiso óptimo "
            "entre rendimiento y huella de memoria.",
        ]:
            story.append(Paragraph(f"• {txt}", sty['bullet']))
        story.append(Spacer(1, 0.2*cm))

        story.append(Paragraph("Formulación matemática del problema central:", sty['h3']))
        story.append(Paragraph(
            "min<sub>λ,α</sub> f(λ,α)  =  ℒ(W*(λ,α))  +  γ · ‖W*(λ,α)‖<super>2</super>",
            sty['math']))
        story.append(Paragraph(
            "donde W*(λ,α) = argmin<sub>W</sub> [ BCE(W) + (λ/2)·‖W‖<super>2</super> ] "
            "es la red entrenada con regularización λ y tasa α, y γ es el peso de penalización "
            "de memoria configurado por el usuario.",
            sty['def_box']))
        story.append(Spacer(1, 0.3*cm))

        # ── §2  DESCRIPCIÓN DEL DATASET ───────────────────────────────────────
        story.append(Paragraph("§2  Descripción del Dataset y Normalización", sty['h1']))

        ds_desc = {
            "Espiral":    "Dos clases en forma de espiral entrelazada. Alta no linealidad: "
                          "ningún clasificador lineal puede separar las clases. Requiere "
                          "capas ocultas profundas con activaciones no lineales para "
                          "aprender la estructura rotacional del espacio.",
            "Lunas":      "Dos medias lunas (crescents) parcialmente superpuestas. La frontera "
                          "de decisión óptima es suave y curva. Dataset estándar de referencia "
                          "para validar capacidad de generalización de clasificadores.",
            "Círculos":   "Un círculo interno rodeado por un anillo externo. La frontera de "
                          "decisión es circular (no convexa). Requiere al menos una capa oculta "
                          "para separar correctamente las clases.",
            "XOR":        "Problema clásico XOR extendido a 500 muestras con ruido gaussiano. "
                          "Las cuatro regiones del plano están coloreadas alternadamente. "
                          "El perceptrón simple no puede resolverlo; es el problema canónico "
                          "de no linealidad.",
            "Gaussianas": "Cuatro agrupaciones gaussianas (blobs) en posiciones fijas del plano. "
                          "Relativamente separables, sirve para verificar que el modelo "
                          "puede al menos clasificar problemas simples correctamente antes "
                          "de abordar datasets complejos."
        }
        story.append(Paragraph(
            f"<b>Dataset seleccionado:</b> {ds}", sty['h3']))
        story.append(Paragraph(
            ds_desc.get(ds, "Dataset de clasificación binaria sintético."),
            sty['body']))
        story.append(Paragraph(
            "Todos los datasets se generan con N = 500 muestras balanceadas (250 por clase) "
            "y se normalizan a media μ = 0, desviación estándar σ = 1 por característica, "
            "lo que estabiliza el gradiente y acelera la convergencia del optimizador Adam. "
            "La partición es 80 % entrenamiento / 20 % validación.",
            sty['body']))
        story.append(Spacer(1, 0.3*cm))

        # ── §3  ARQUITECTURA MLP ──────────────────────────────────────────────
        story.append(Paragraph("§3  Arquitectura de la Red Neuronal (MLP)", sty['h1']))
        story.append(Paragraph(
            f"La red implementada es un Perceptrón Multicapa (MLP) de 3 capas con la "
            f"arquitectura <b>2 → {h1} → {h2} → 1</b>. La capa de entrada tiene 2 neuronas "
            f"(coordenadas x₁, x₂ del plano 2D), dos capas ocultas con {h1} y {h2} neuronas "
            f"respectivamente, y una neurona de salida con activación sigmoide para clasificación "
            f"binaria. La activación seleccionada en las capas ocultas es <b>{act}</b>.",
            sty['body']))

        # Tabla de capas
        layer_data = [
            ["Capa", "Neuronas", "Activación", "Parámetros", "Función"],
            ["Entrada (L0)", "2", "—", "—", "Recibe (x₁, x₂)"],
            [f"Oculta 1 (L1)", str(h1), act, str(2*h1 + h1), f"z=W₁x+b₁, a=σ(z)"],
            [f"Oculta 2 (L2)", str(h2), act, str(h1*h2 + h2), f"z=W₂a₁+b₂, a=σ(z)"],
            ["Salida (L3)", "1", "Sigmoid", str(h2*1 + 1), "ŷ = σ(W₃a₂+b₃)"],
            ["TOTAL", "", "", str(2*h1+h1 + h1*h2+h2 + h2+1), ""]
        ]
        tbl_layers = Table(layer_data, colWidths=[2.8*cm, 2.2*cm, 2.4*cm, 2.6*cm, 4.5*cm])
        tbl_layers.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#0c2d5a")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), HexColor("#00c9e0")),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME",   (0, -1), (-1, -1), "Helvetica-Bold"),
            ("BACKGROUND", (0, -1), (-1, -1), HexColor("#0a1a30")),
            ("TEXTCOLOR",  (0, -1), (-1, -1), HexColor("#7bed9f")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -2),
             [HexColor("#0d0d1a"), HexColor("#0f0f22")]),
            ("TEXTCOLOR",  (0, 1), (-1, -2), HexColor("#e8e8f0")),
            ("FONTSIZE",   (0, 0), (-1, -1), 8),
            ("GRID",       (0, 0), (-1, -1), 0.4, HexColor("#0c2d5a")),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
            ("ALIGN",      (1, 0), (3, -1), "CENTER"),
        ]))
        story.append(tbl_layers)
        story.append(Paragraph(
            f"Tabla 1. Estructura de capas de la MLP: 2→{h1}→{h2}→1. "
            f"Total de parámetros: {2*h1+h1 + h1*h2+h2 + h2+1} (pesos + sesgos).",
            sty['caption']))
        story.append(Spacer(1, 0.2*cm))

        act_desc = {
            "ReLU":  "ReLU(z) = max(0, z). Evita el problema de gradiente desvaneciente en "
                     "redes profundas. Computacionalmente eficiente. Puede producir neuronas "
                     'muertas (dead neurons) si la tasa de aprendizaje es muy alta.',
            "Tanh":  "Tanh(z) = (e^z - e^{-z})/(e^z + e^{-z}) ∈ (-1, 1). Centrada en cero, "
                     "produce gradientes más estables que la sigmoide. Recomendada para "
                     "problemas con fronteras de decisión curvas y complejas.",
            "ELU":   "ELU(z) = z si z>0; α(e^z - 1) si z≤0. Permite valores negativos "
                     "suaves que acercan la media de las activaciones a cero, acelerando "
                     "la convergencia sin neuronas muertas.",
            "Swish": "Swish(z) = z · σ(z) = z/(1+e^{-z}). Función suave, no monótona, "
                     "propuesta por Google Brain. Empíricamente supera a ReLU en redes "
                     "profundas al mantener activaciones suaves cerca de cero."
        }
        story.append(Paragraph(f"Función de activación — {act}:", sty['h3']))
        story.append(Paragraph(act_desc.get(act, ""), sty['def_box']))
        story.append(Spacer(1, 0.3*cm))

        # ── §4  OPTIMIZADOR ADAM ──────────────────────────────────────────────
        story.append(Paragraph("§4  Optimizador Adam — Análisis Variacional", sty['h1']))
        story.append(Paragraph(
            "Adam (Adaptive Moment Estimation) combina momentum de primer orden y "
            "estimación del segundo momento de los gradientes para adaptar la tasa de "
            "aprendizaje de cada parámetro individualmente. Es el estándar de facto "
            "en entrenamiento de redes neuronales modernas.",
            sty['body']))
        story.append(Paragraph("Reglas de actualización de Adam:", sty['math_label']))
        for eq in [
            "gₜ  =  ∂ℒ/∂θₜ₋₁           (gradiente de la pérdida en paso t)",
            "mₜ  =  β₁·mₜ₋₁  +  (1−β₁)·gₜ      (momento de 1er orden; β₁ = 0.9)",
            "vₜ  =  β₂·vₜ₋₁  +  (1−β₂)·gₜ²     (momento de 2do orden; β₂ = 0.999)",
            "m̂ₜ  =  mₜ / (1 − β₁ᵗ)              (corrección de sesgo — 1er orden)",
            "v̂ₜ  =  vₜ / (1 − β₂ᵗ)              (corrección de sesgo — 2do orden)",
            "θₜ  =  θₜ₋₁ − α · m̂ₜ / (√v̂ₜ + ε)  (actualización del parámetro; ε=1e-8)",
        ]:
            story.append(Paragraph(eq, sty['math']))
        story.append(Paragraph(
            "La corrección de sesgo en los primeros pasos evita que el optimizador "
            "sobreestime el momento cuando mₜ ≈ 0. La división por √v̂ₜ normaliza "
            "el gradiente según su varianza histórica, logrando pasos adaptativos: "
            "parámetros con gradientes grandes reciben actualizaciones más pequeñas, "
            "y viceversa. Esto permite manejar funciones de pérdida con curvaturas "
            "muy dispares entre diferentes capas de la red.",
            sty['insight']))
        story.append(Spacer(1, 0.3*cm))

        # ── §5  EARLY STOPPING + LR DECAY ────────────────────────────────────
        story.append(Paragraph("§5  Early Stopping y Decaimiento de LR", sty['h1']))
        story.append(Paragraph(
            "Para evitar el sobreajuste sin fijar arbitrariamente el número de épocas, "
            "se implementan dos mecanismos automáticos de regularización implícita:",
            sty['body']))
        story.append(Paragraph(
            f"<b>Early Stopping:</b> Si la pérdida de validación no mejora en más de "
            f"<i>patience</i> = {int(self.v_patience.get())} épocas consecutivas, el "
            f"entrenamiento se detiene y se restauran los pesos del mejor epoch registrado "
            f"(checkpoint óptimo). Esto garantiza que la red entregada es la de mejor "
            f"desempeño, no la del último paso.",
            sty['def_box']))
        story.append(Paragraph(
            "<b>LR Decay:</b> Cuando se activa el early stopping por primera vez, "
            "la tasa de aprendizaje se reduce en un factor de 0.3 (×0.3) y el entrenamiento "
            "continúa, permitiendo que el optimizador explore vecindades más finas del mínimo "
            "local. Este proceso se repite hasta 2 veces antes de detener definitivamente.",
            sty['def_box']))
        story.append(Spacer(1, 0.3*cm))

        # ── §6  RESULTADOS NUMÉRICOS ──────────────────────────────────────────
        story.append(Paragraph("§6  Resultados Numéricos Obtenidos", sty['h1']))

        if self.net_base:
            res_data = [["Métrica", "Red Base", "Red Optimizada", "Mejora"]]

            acc_b = self.net_base.best_acc * 100
            acc_o = (self.net_opt.best_acc * 100) if self.net_opt else None
            loss_b = self.net_base.best_loss
            loss_o = self.net_opt.best_loss if self.net_opt else None

            # Norma de pesos (MLP usa W1/b1, W2/b2, W3/b3 directamente)
            def get_norm(net):
                if net is None: return None
                try:
                    return (np.sum(net.W1**2) + np.sum(net.b1**2) +
                            np.sum(net.W2**2) + np.sum(net.b2**2) +
                            np.sum(net.W3**2) + np.sum(net.b3**2))
                except AttributeError:
                    return None

            norm_b = get_norm(self.net_base)
            norm_o = get_norm(self.net_opt)

            def fmt_delta(a, b, fmt="{:.3f}", better="lower"):
                if b is None: return "—", "—", "—"
                delta = b - a
                pct = (b - a) / (abs(a) + 1e-9) * 100
                arrow = "↓" if delta < 0 else "↑"
                if better == "lower":
                    color_txt = "mejor" if delta < 0 else "peor"
                else:
                    color_txt = "mejor" if delta > 0 else "peor"
                return fmt.format(a), fmt.format(b), f"{arrow} {abs(pct):.1f}% ({color_txt})"

            vb, vo, vd = fmt_delta(acc_b, acc_o, "{:.2f} %", "higher")
            res_data.append(["Precisión", vb, vo if vo != "—" else "—", vd])

            vb, vo, vd = fmt_delta(loss_b, loss_o, "{:.5f}", "lower")
            res_data.append(["Pérdida BCE", vb, vo if vo != "—" else "—", vd])

            vb, vo, vd = fmt_delta(norm_b, norm_o, "{:.4f}", "lower")
            res_data.append(["‖W‖² (memoria)", vb, vo if vo != "—" else "—", vd])

            res_data.append([
                "Épocas",
                str(self.net_base.epoch_count),
                str(self.net_opt.epoch_count) if self.net_opt else "—",
                "—"
            ])
            if self.opt_lam is not None:
                res_data.append(["λ* óptimo", f"{lam_base:.6f}",
                                  f"{self.opt_lam:.6f}", "L-BFGS-B"])
                res_data.append(["α* óptimo", f"{lr_base:.5f}",
                                  f"{self.opt_lr:.5f}", "L-BFGS-B"])

            tbl_res = Table(res_data, colWidths=[3.8*cm, 3.5*cm, 3.5*cm, 4.7*cm])
            tbl_res.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#0c2d5a")),
                ("TEXTCOLOR",  (0, 0), (-1, 0), HexColor("#00c9e0")),
                ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",   (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [HexColor("#0d0d1a"), HexColor("#0f0f22")]),
                ("TEXTCOLOR",  (0, 1), (0, -1), HexColor("#9090b8")),
                ("TEXTCOLOR",  (1, 1), (-1, -1), HexColor("#e8e8f0")),
                ("GRID",       (0, 0), (-1, -1), 0.4, HexColor("#0c2d5a")),
                ("TOPPADDING",    (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ]))
            story.append(tbl_res)
            story.append(Paragraph(
                "Tabla 2. Comparativa cuantitativa entre la red base y la red optimizada.",
                sty['caption']))

            # Diagnóstico automático
            story.append(Paragraph("Diagnóstico automático:", sty['h3']))
            if acc_b >= 95:
                story.append(Paragraph(
                    f"✓ Precisión base excelente ({acc_b:.1f}%). El dataset {ds} con activación "
                    f"{act} está bien configurado para esta arquitectura.",
                    sty['insight']))
            elif acc_b >= 80:
                story.append(Paragraph(
                    f"⚠ Precisión base aceptable ({acc_b:.1f}%). Considera aumentar h1, h2 o "
                    f"el número de épocas para mejorar la separación.",
                    sty['warning']))
            else:
                story.append(Paragraph(
                    f"✗ Precisión base baja ({acc_b:.1f}%). El modelo necesita ajuste. "
                    f"Prueba activación Tanh, aumenta h1=64, h2=32, reduce λ a 0.00001.",
                    sty['warning']))

            if norm_b is not None and norm_b > 50:
                story.append(Paragraph(
                    f"⚠ Norma de pesos alta (‖W‖² = {norm_b:.2f}). Considera aumentar "
                    f"γ para penalizar más la memoria o reducir la arquitectura.",
                    sty['warning']))
            elif norm_b is not None:
                story.append(Paragraph(
                    f"✓ Norma de pesos controlada (‖W‖² = {norm_b:.2f}). "
                    f"La regularización está operando correctamente.",
                    sty['insight']))

        story.append(PageBreak())

        # ── §7  GRÁFICAS CON ANÁLISIS VARIACIONAL ────────────────────────────
        story.append(Paragraph(
            "§7  Análisis Variacional de las Gráficas", sty['h1']))
        story.append(Paragraph(
            "En esta sección se presentan las figuras generadas por el módulo experimental "
            "con una interpretación rigurosa de su comportamiento variacional, identificando "
            "zonas de convergencia, anomalías y el rol de cada curva en la dinámica de "
            "optimización.", sty['body']))
        story.append(Spacer(1, 0.2*cm))

        usable_w = 15.6 * cm
        half_w   = 7.5  * cm

        # -- Figura 1: Training tab
        story.append(Paragraph("Figura 1 — Curvas de Entrenamiento", sty['h2']))
        buf1 = self._fig_to_png_bytes(self.fig1)
        story.append(RLImage(buf1, width=usable_w, height=usable_w*0.38))
        story.append(Paragraph(
            "Fig. 1: Panel de entrenamiento. Izquierda: BCE base y optimizada vs época. "
            "Centro: evolución de la norma ‖W‖ durante el entrenamiento. "
            "Derecha: precisión de clasificación, con línea de referencia al 90%.",
            sty['caption']))
        story.append(Paragraph(
            "<b>Análisis variacional — Pérdida BCE:</b> Una curva de pérdida correctamente "
            "convergente es monótonamente decreciente y cóncava (d²ℒ/dt² > 0). Oscilaciones "
            "de alta frecuencia indican tasa de aprendizaje excesiva; un plateau prematuro "
            "señala mínimo local o gradiente desvaneciente. La brecha entre la curva base y "
            "la optimizada cuantifica la ganancia del optimizador L-BFGS-B.",
            sty['body']))
        story.append(Paragraph(
            "<b>Análisis variacional — Norma ‖W‖:</b> La evolución de la norma revela el "
            "equilibrio entre expresividad (normas grandes) y regularización (normas pequeñas). "
            "Un incremento monótono con pérdida decreciente indica sobreajuste potencial. "
            "El optimizador NL busca reducir esta norma sin comprometer la precisión, "
            "desplazando la solución hacia regiones de menor 'energía' de pesos.",
            sty['body']))
        story.append(Paragraph(
            "<b>Análisis variacional — Precisión:</b> La curva de accuracy exhibe comportamiento "
            "escalonado: períodos de meseta (exploración) seguidos de saltos bruscos "
            "(explotación de gradiente). Una precisión > 90% sostenida confirma que la red "
            "ha aprendido la estructura topológica del dataset.",
            sty['body']))
        story.append(Spacer(1, 0.3*cm))

        # -- Figura 2: Boundary tab
        story.append(Paragraph("Figura 2 — Frontera de Decisión", sty['h2']))
        buf2 = self._fig_to_png_bytes(self.fig2)
        story.append(RLImage(buf2, width=usable_w, height=usable_w*0.46))
        story.append(Paragraph(
            f"Fig. 2: Frontera de decisión en ℝ² para el dataset {ds}. "
            "Izquierda: red base. Derecha: red con hiperparámetros optimizados por L-BFGS-B.",
            sty['caption']))
        story.append(Paragraph(
            "<b>Análisis variacional — Frontera de decisión:</b> La complejidad geométrica "
            "de la frontera refleja directamente la capacidad expresiva de la red. Una "
            "frontera suave y bien definida indica que la red ha aprendido la estructura "
            "real del espacio de entrada. Bordes rugosos o irregulares señalan sobreajuste "
            "(la red memoriza ruido en lugar de generalizar). La región de color uniforme "
            "entre clases es proporcional al margen de clasificación: márgenes amplios "
            "correlacionan con mejor generalización fuera de la muestra.",
            sty['body']))
        story.append(Paragraph(
            "La comparación base vs optimizado permite visualizar el efecto del optimizador "
            "L-BFGS-B sobre la geometría de la frontera. Una frontera más suave tras la "
            "optimización confirma que la reducción de ‖W‖ tiene efecto regularizador: "
            "pesos de menor magnitud producen transiciones más graduales en el espacio de "
            "activación, equivalente a incrementar el margen efectivo del clasificador.",
            sty['body']))
        story.append(PageBreak())

        # -- Figura 3: Optimizer tab
        story.append(Paragraph("Figura 3 — Trayectoria del Optimizador L-BFGS-B", sty['h2']))
        if self.opt_iters:
            buf3 = self._fig_to_png_bytes(self.fig3)
            story.append(RLImage(buf3, width=usable_w, height=usable_w*0.65))
            story.append(Paragraph(
                "Fig. 3: Panel 2×3 del optimizador. (1,1) Convergencia de f(λ,α). "
                "(1,2) Trayectoria de λ coloreada por calidad. "
                "(1,3) Trayectoria de α con eje positivo garantizado. "
                "(2,1) Mapa (λ,α)→f con tamaño proporcional a calidad. "
                "(2,2) Mejora relativa acumulada %. "
                "(2,3) Trade-off BCE vs ‖W‖² — espacio de compromiso.",
                sty['caption']))
            story.append(Paragraph(
                "<b>Análisis variacional — Convergencia f(λ,α):</b> L-BFGS-B utiliza "
                "aproximaciones quasi-Newton de la inversa del Hessiano para calcular "
                "direcciones de descenso. La convergencia cuadrática local garantiza que, "
                "cerca del mínimo, el error se reduce cuadráticamente en cada iteración. "
                "La coloración de la trayectoria (cool colormap: azul→magenta) permite "
                "identificar la velocidad de descenso: colores cercanos al magenta "
                "indican iteraciones tardías donde el optimizador refina la solución.",
                sty['body']))
            story.append(Paragraph(
                "<b>Análisis variacional — Espacio (λ,α)→f:</b> Este diagrama de dispersión "
                "es el más informativo del panel. El tamaño de cada punto es inversamente "
                "proporcional a f (puntos grandes = mejor calidad). Las flechas de trayectoria "
                "revelan la dirección de búsqueda del optimizador en el espacio de "
                "hiperparámetros. Un camino directo hacia la región de puntos grandes indica "
                "que L-BFGS-B identificó correctamente el gradiente del paisaje de pérdida "
                "sobre (λ,α). Bifurcaciones o retrocesos señalan regiones no convexas donde "
                "el optimizador debió explorar.",
                sty['body']))
            story.append(Paragraph(
                "<b>Análisis variacional — Trade-off BCE vs ‖W‖²:</b> Este diagrama captura "
                "la tensión fundamental del problema: minimizar la pérdida de clasificación "
                "tiende a aumentar ‖W‖, mientras que penalizar ‖W‖ puede empeorar BCE. "
                "La solución óptima (★) se ubica en la frontera de Pareto de este "
                "trade-off. La trayectoria coloreada (viridis: oscuro→claro=inicio→fin) "
                "muestra si el optimizador navega hacia o desde la frontera de Pareto.",
                sty['body']))
        else:
            story.append(Paragraph(
                "[Optimizador no ejecutado — corre ⚙ Optimizar Hiperparámetros para "
                "incluir el análisis del optimizador L-BFGS-B en el reporte.]",
                sty['warning']))

        story.append(Spacer(1, 0.3*cm))

        # ── §8  CONCLUSIONES ──────────────────────────────────────────────────
        story.append(Paragraph("§8  Conclusiones y Recomendaciones", sty['h1']))

        concl = []
        if self.net_base:
            acc_b = self.net_base.best_acc * 100
            if acc_b >= 95:
                concl.append(
                    f"La red base alcanzó una precisión de {acc_b:.1f}%, confirmando que la "
                    f"arquitectura 2→{h1}→{h2}→1 con activación {act} es suficientemente "
                    f"expresiva para el dataset {ds}.")
            else:
                concl.append(
                    f"La precisión base del {acc_b:.1f}% sugiere que la configuración actual "
                    f"puede mejorarse. Se recomienda: (1) aumentar h1 a 64, h2 a 32; "
                    f"(2) usar activación Tanh; (3) reducir λ a 1e-5.")
        if self.opt_lam is not None:
            concl.append(
                f"El optimizador L-BFGS-B encontró los hiperparámetros óptimos "
                f"λ* = {self.opt_lam:.6f} y α* = {self.opt_lr:.6f} en "
                f"{len(self.opt_iters)} evaluaciones, logrando el mejor compromiso "
                f"entre clasificación y eficiencia de memoria.")
        concl.append(
            "Para reproducibilidad: ejecuta el módulo con los parámetros registrados "
            "en la Tabla 1 de la portada. La semilla aleatoria no está fijada; variaciones "
            "menores (<2%) entre ejecuciones son normales debido a la inicialización "
            "Xavier estocástica.")

        for c in concl:
            story.append(Paragraph(f"• {c}", sty['bullet']))

        story.append(Spacer(1, 0.4*cm))
        story.append(HRFlowable(width="100%", thickness=0.8,
                                color=HexColor("#0c2d5a"), spaceAfter=6))
        story.append(Paragraph(
            "Generado automáticamente por el Módulo Experimental v4.0 — "
            "Red Neuronal Profunda + Optimización No Lineal de Memoria. "
            "Autor del módulo: Perplexity Computer.",
            sty['ref']))

        try:
            doc.build(story,
                      onFirstPage=hf_report,
                      onLaterPages=hf_report)
        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            self.lbl_status.config(text=f"Error PDF: {str(e)[:80]}")
            messagebox.showerror("Error al generar PDF",
                f"No se pudo crear el reporte:\n\n{str(e)}\n\n"
                f"Ruta intentada: {path}")
            print("[PDF ERROR]\n" + err_msg)
            return

        self.lbl_status.config(
            text=f"✓ Reporte PDF guardado en: {os.path.basename(path)}")
        messagebox.showinfo("PDF generado",
            f"Reporte Inteligente guardado en:\n{path}")

    # ═════════════════════════════════════════════════════════════════════════
    #  MODELO MATEMÁTICO PDF
    # ═════════════════════════════════════════════════════════════════════════
    def _export_math_pdf(self):
        """Genera el PDF formal del Modelo Matemático con todas las secciones."""
        try:
            self._do_export_math_pdf()
        except Exception as _e:
            import traceback as _tb
            _msg = _tb.format_exc()
            print("[MATH PDF ERROR]\n" + _msg)
            try:
                self.lbl_status.config(text=f"Error: {str(_e)[:90]}")
                messagebox.showerror("Error al generar Modelo PDF",
                    f"{str(_e)}\n\nRevisa la consola para el detalle completo.")
            except Exception:
                pass

    def _do_export_math_pdf(self):
        """Cuerpo real del generador — llamado desde _export_math_pdf."""
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, PageBreak,
            Table, TableStyle, HRFlowable, KeepTogether
        )
        from reportlab.lib.units import cm
        from reportlab.lib.colors import HexColor

        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")],
            initialfile=f"modelo_matematico_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            title="Guardar Modelo Matemático PDF"
        )
        if not path:
            return

        self.lbl_status.config(text="Generando Modelo Matemático PDF...")
        self.update_idletasks()

        W, H = A4
        sty, col = self._pdf_styles()

        h1v = int(self.v_h1.get()); h2v = int(self.v_h2.get())
        actv = self.v_act.get()
        lam_v = self.v_lam.get(); lr_v = self.v_lr.get()
        gam_v = self.v_gamma.get()

        def hf_math(c, d):
            self._pdf_header_footer(c, d,
                "Modelo Matemático Formal — MLP + Optimizador No Lineal",
                f"h1={h1v}  h2={h2v}  act={actv}  λ={lam_v:.5f}  γ={gam_v:.3f}")

        doc = SimpleDocTemplate(
            path, pagesize=A4,
            title="Modelo Matemático Formal — MLP + Optimizador No Lineal",
            author="Perplexity Computer",
            leftMargin=2.2*cm, rightMargin=2.2*cm,
            topMargin=1.8*cm, bottomMargin=1.8*cm
        )

        story = []

        # ── PORTADA ──────────────────────────────────────────────────────────
        story.append(Spacer(1, 1.5*cm))
        story.append(Paragraph("MODELO MATEMÁTICO FORMAL", sty['cover_title']))
        story.append(Paragraph(
            "Red Neuronal Profunda (MLP) con Optimización No Lineal de Memoria",
            sty['cover_sub']))
        story.append(Paragraph(
            "Descripción rigurosa de todas las estructuras matemáticas, "
            "operadores, espacios de búsqueda y relaciones funcionales del módulo experimental.",
            sty['cover_meta']))
        story.append(Spacer(1, 0.5*cm))
        story.append(HRFlowable(width="100%", thickness=1.5,
                                color=HexColor("#c77dff"), spaceAfter=12))

        # Card de config actual
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph("Configuración actual del modelo:", sty['h3']))
        cfg_data = [
            ["Símbolo", "Descripción", "Valor actual"],
            ["n0", "Dimensión de entrada", "2 (x₁, x₂)"],
            ["h1", "Neuronas capa oculta 1", str(h1v)],
            ["h2", "Neuronas capa oculta 2", str(h2v)],
            ["n3", "Neuronas capa de salida", "1 (clasificación binaria)"],
            ["σ(·)", "Activación capas ocultas", actv],
            ["λ", "Coeficiente de regularización L2 (base)", f"{lam_v:.6f}"],
            ["α", "Tasa de aprendizaje (base)", f"{lr_v:.5f}"],
            ["γ", "Peso de penalización de memoria", f"{gam_v:.3f}"],
            ["β₁", "Decaimiento 1er momento (Adam)", "0.9"],
            ["β₂", "Decaimiento 2do momento (Adam)", "0.999"],
            ["ε", "Estabilidad numérica (Adam)", "1 × 10⁻⁸"],
            ["N", "Muestras del dataset", "500"],
            ["p", "Patience (early stopping)", str(int(self.v_patience.get()))],
        ]
        if self.opt_lam is not None:
            cfg_data.append(["λ*", "Regularización óptima (L-BFGS-B)", f"{self.opt_lam:.6f}"])
            cfg_data.append(["α*", "Learning rate óptimo (L-BFGS-B)", f"{self.opt_lr:.6f}"])

        tbl_cfg = Table(cfg_data, colWidths=[1.8*cm, 8.5*cm, 4.2*cm])
        tbl_cfg.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a0a3a")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), HexColor("#c77dff")),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 8),
            ("BACKGROUND", (0, 1), (0, -1), HexColor("#120822")),
            ("TEXTCOLOR",  (0, 1), (0, -1), HexColor("#c77dff")),
            ("FONTNAME",   (0, 1), (0, -1), "Courier"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [HexColor("#0d0d1a"), HexColor("#100d20")]),
            ("TEXTCOLOR",  (1, 1), (1, -1), HexColor("#9090b8")),
            ("TEXTCOLOR",  (2, 1), (2, -1), HexColor("#7bed9f")),
            ("GRID",       (0, 0), (-1, -1), 0.4, HexColor("#1a0a3a")),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ]))
        story.append(tbl_cfg)
        story.append(PageBreak())

        # ── §1  ESPACIO DE ENTRADA ────────────────────────────────────────────
        story.append(Paragraph("§1  Espacio de Entrada y Normalización", sty['h1']))
        story.append(Paragraph(
            "Sea el conjunto de datos de entrenamiento:", sty['body']))
        story.append(Paragraph(
            "D = { (x⁽ⁱ⁾, y⁽ⁱ⁾) : x⁽ⁱ⁾ ∈ ℝ², y⁽ⁱ⁾ ∈ {0,1} }  con  |D| = N = 500",
            sty['math']))
        story.append(Paragraph(
            "<b>x⁽ⁱ⁾ ∈ ℝ²</b>: vector de características bidimensional del i-ésimo ejemplo. "
            "Componentes x₁ (abscisa) y x₂ (ordenada) en el plano cartesiano sintético. "
            "<b>y⁽ⁱ⁾ ∈ {0,1}</b>: etiqueta de clase binaria (0=clase negativa, 1=clase positiva).",
            sty['def_box']))
        story.append(Paragraph("Normalización z-score por característica:", sty['math_label']))
        story.append(Paragraph(
            "x̃⁽ⁱ⁾ⱼ  =  (x⁽ⁱ⁾ⱼ − μⱼ) / σⱼ     con  μⱼ = (1/N)·Σᵢx⁽ⁱ⁾ⱼ,  σⱼ = √[(1/N)Σᵢ(x⁽ⁱ⁾ⱼ−μⱼ)²]",
            sty['math']))
        story.append(Paragraph(
            "La normalización garantiza que el gradiente de la pérdida tenga magnitudes "
            "similares en todas las direcciones del espacio de parámetros, evitando que "
            "el optimizador Adam gaste iteraciones corrigiendo diferencias de escala. "
            "El dataset normalizado satisface: E[x̃ⱼ] = 0 y Var[x̃ⱼ] = 1 para j = 1,2.",
            sty['body']))
        story.append(Spacer(1, 0.3*cm))

        # ── §2  ARQUITECTURA MLP ──────────────────────────────────────────────
        story.append(Paragraph(f"§2  Arquitectura MLP: 2→{h1v}→{h2v}→1", sty['h1']))
        story.append(Paragraph(
            "La red neuronal es una función paramétrica "
            "f_W : ℝ² → (0,1) compuesta por transformaciones afines seguidas de "
            "no linealidades punto a punto:", sty['body']))

        layers_math = [
            ("Capa 1 (oculta)",
             f"z⁽¹⁾ = W₁ · x̃ + b₁,    z⁽¹⁾ ∈ ℝ^{h1v}",
             f"W₁ ∈ ℝ^({h1v}×2),  b₁ ∈ ℝ^{h1v}"),
            ("Activación 1",
             f"a⁽¹⁾ = σ(z⁽¹⁾)  [componente a componente],   σ = {actv}",
             f"a⁽¹⁾ ∈ ℝ^{h1v}"),
            ("Capa 2 (oculta)",
             f"z⁽²⁾ = W₂ · a⁽¹⁾ + b₂,    z⁽²⁾ ∈ ℝ^{h2v}",
             f"W₂ ∈ ℝ^({h2v}×{h1v}),  b₂ ∈ ℝ^{h2v}"),
            ("Activación 2",
             f"a⁽²⁾ = σ(z⁽²⁾)  [componente a componente],   σ = {actv}",
             f"a⁽²⁾ ∈ ℝ^{h2v}"),
            ("Capa 3 (salida)",
             f"z⁽³⁾ = W₃ · a⁽²⁾ + b₃,    z⁽³⁾ ∈ ℝ",
             f"W₃ ∈ ℝ^(1×{h2v}),  b₃ ∈ ℝ"),
            ("Salida",
             "ŷ = sigmoid(z⁽³⁾) = 1 / (1 + e^{−z⁽³⁾})  ∈ (0,1)",
             "ŷ ∈ (0,1)"),
        ]
        for lbl, eq, dim in layers_math:
            story.append(Paragraph(lbl + ":", sty['math_label']))
            story.append(Paragraph(eq, sty['math']))
            story.append(Paragraph(
                f"  Dimensiones: {dim}", sty['def_box']))

        story.append(Paragraph(
            "Conjunto completo de parámetros entrenables:", sty['math_label']))
        total_params = 2*h1v + h1v + h1v*h2v + h2v + h2v + 1
        story.append(Paragraph(
            f"W = (W₁, b₁, W₂, b₂, W₃, b₃)  ∈  "
            f"ℝ^{2*h1v} × ℝ^{h1v} × ℝ^{h1v*h2v} × ℝ^{h2v} × ℝ^{h2v} × ℝ",
            sty['math']))
        story.append(Paragraph(
            f"Total de grados de libertad: |W| = {total_params} parámetros "
            f"({2*h1v+h1v} en L1, {h1v*h2v+h2v} en L2, {h2v+1} en L3).",
            sty['def_box']))
        story.append(Spacer(1, 0.3*cm))

        # ── §3  INICIALIZACIÓN XAVIER ─────────────────────────────────────────
        story.append(Paragraph("§3  Inicialización de Pesos — Xavier/Glorot", sty['h1']))
        story.append(Paragraph(
            "Los pesos se inicializan mediante la distribución de Glorot (Xavier) para "
            "mantener la varianza de las activaciones constante a través de las capas, "
            "evitando explosión o desvanecimiento del gradiente en el paso forward:",
            sty['body']))
        story.append(Paragraph(
            "Wₗ ~ Uniforme(−√(6/(nₗ₋₁ + nₗ)),  +√(6/(nₗ₋₁ + nₗ)))",
            sty['math']))
        story.append(Paragraph(
            f"Para L1: rango = ±√(6/(2+{h1v})) = ±{(6/(2+h1v))**0.5:.4f}.  "
            f"Para L2: rango = ±√(6/({h1v}+{h2v})) = ±{(6/(h1v+h2v))**0.5:.4f}.  "
            f"Para L3: rango = ±√(6/({h2v}+1)) = ±{(6/(h2v+1))**0.5:.4f}.",
            sty['def_box']))
        story.append(Paragraph(
            "Los sesgos bₗ se inicializan a cero: bₗ = 0. Esta elección es estándar "
            "porque la simetría ya se rompe por los pesos aleatorios.",
            sty['body']))
        story.append(Spacer(1, 0.3*cm))

        # ── §4  FUNCIÓN DE PÉRDIDA ────────────────────────────────────────────
        story.append(Paragraph("§4  Función de Pérdida con Regularización L2", sty['h1']))
        story.append(Paragraph(
            "La función objetivo que minimiza el entrenamiento es la Entropía Cruzada "
            "Binaria (BCE) con regularización Ridge (L2):",
            sty['body']))
        story.append(Paragraph("Función de pérdida total:", sty['math_label']))
        story.append(Paragraph(
            "ℒ(W; λ) = BCE(W) + (λ/2) · ‖W‖²",
            sty['math']))
        story.append(Paragraph("Descomposición de cada término:", sty['math_label']))
        story.append(Paragraph(
            "BCE(W) = −(1/N) · Σᵢ₌₁ᴺ [ y⁽ⁱ⁾·log(ŷ⁽ⁱ⁾) + (1−y⁽ⁱ⁾)·log(1−ŷ⁽ⁱ⁾) ]",
            sty['math']))
        story.append(Paragraph(
            f"(λ/2)·‖W‖² = (λ/2) · (‖W₁‖²_F + ‖W₂‖²_F + ‖W₃‖²_F + ‖b₁‖² + ‖b₂‖² + ‖b₃‖²)",
            sty['math']))
        story.append(Paragraph(
            "<b>BCE(W):</b> mide el desacuerdo entre las probabilidades predichas ŷ⁽ⁱ⁾ ∈ (0,1) "
            "y las etiquetas binarias y⁽ⁱ⁾ ∈ {0,1}. Alcanza su mínimo (BCE=0) cuando "
            "ŷ⁽ⁱ⁾ = y⁽ⁱ⁾ para todo i.",
            sty['def_box']))
        story.append(Paragraph(
            f"<b>(λ/2)·‖W‖²:</b> penalización Ridge. Encoge todos los pesos hacia cero "
            f"proporcionalmente a λ. Con λ = {lam_v:.6f} (valor base), el término de "
            f"regularización es pequeño, priorizando la clasificación sobre la compresión "
            f"de memoria. El optimizador NL ajusta λ automáticamente.",
            sty['def_box']))
        story.append(Spacer(1, 0.3*cm))

        # ── §5  RETROPROPAGACIÓN ──────────────────────────────────────────────
        story.append(Paragraph("§5  Retropropagación del Gradiente", sty['h1']))
        story.append(Paragraph(
            "El gradiente de ℒ respecto a cada parámetro se obtiene por la regla de la "
            "cadena aplicada capa a capa (backpropagation). Sean δ⁽ˡ⁾ los 'errores locales' "
            "propagados hacia atrás:", sty['body']))
        story.append(Paragraph("Paso forward (ya definido en §2).", sty['math_label']))
        story.append(Paragraph("Gradiente en capa de salida (L3):", sty['math_label']))
        story.append(Paragraph(
            "δ⁽³⁾ = ŷ − y  ∈ ℝ   (error de predicción)",
            sty['math']))
        story.append(Paragraph(
            "∂ℒ/∂W₃ = δ⁽³⁾ · (a⁽²⁾)ᵀ  +  λ·W₃",
            sty['math']))
        story.append(Paragraph(
            "∂ℒ/∂b₃ = δ⁽³⁾",
            sty['math']))
        story.append(Paragraph("Propagación a L2:", sty['math_label']))
        story.append(Paragraph(
            "δ⁽²⁾ = (W₃)ᵀ · δ⁽³⁾  ⊙  σ'(z⁽²⁾)   (⊙ = producto elemento a elemento)",
            sty['math']))
        story.append(Paragraph(
            "∂ℒ/∂W₂ = δ⁽²⁾ · (a⁽¹⁾)ᵀ  +  λ·W₂",
            sty['math']))
        story.append(Paragraph("Propagación a L1:", sty['math_label']))
        story.append(Paragraph(
            "δ⁽¹⁾ = (W₂)ᵀ · δ⁽²⁾  ⊙  σ'(z⁽¹⁾)",
            sty['math']))
        story.append(Paragraph(
            "∂ℒ/∂W₁ = δ⁽¹⁾ · (x̃)ᵀ  +  λ·W₁",
            sty['math']))
        story.append(Paragraph(
            f"Donde σ'(z) es la derivada de la función de activación {actv}: "
            + {"ReLU": "σ'(z) = 1 si z>0, 0 si z≤0 (función escalonada de Heaviside).",
               "Tanh": "σ'(z) = 1 − tanh²(z) = 1 − (a)² (se calcula eficientemente desde a).",
               "ELU":  "σ'(z) = 1 si z>0; α·e^z si z≤0 (continua en z=0 para α=1).",
               "Swish": "σ'(z) = σ(z) + z·σ(z)·(1−σ(z)) (derivada suave, no monótona)."
              }.get(actv, "σ'(z) definida según la función de activación seleccionada."),
            sty['def_box']))
        story.append(PageBreak())

        # ── §6  OPTIMIZADOR ADAM ──────────────────────────────────────────────
        story.append(Paragraph("§6  Optimizador Adam — Especificación Completa", sty['h1']))
        story.append(Paragraph(
            "Adam es un método de descenso de gradiente estocástico de primer orden con "
            "tasas de aprendizaje adaptativas por parámetro. Propuesto por Kingma y Ba "
            "(ICLR 2015). Su complejidad por iteración es O(|W|) en tiempo y espacio.",
            sty['body']))

        adam_rows = [
            ["Variable", "Descripción", "Valor / Dominio"],
            ["gₜ", "Gradiente ∂ℒ/∂θ en paso t", "ℝ^|W|"],
            ["mₜ", "Estimador del 1er momento (media exp.)", "m₀=0, β₁=0.9"],
            ["vₜ", "Estimador del 2do momento (varianza exp.)", "v₀=0, β₂=0.999"],
            ["m̂ₜ", "1er momento con corrección de sesgo", "mₜ/(1−β₁ᵗ)"],
            ["v̂ₜ", "2do momento con corrección de sesgo", "vₜ/(1−β₂ᵗ)"],
            ["α", "Tasa de aprendizaje global", f"{lr_v:.5f}"],
            ["ε", "Estabilizador numérico", "1×10⁻⁸"],
            ["θₜ", "Actualización del parámetro θ", "θₜ₋₁ − α·m̂ₜ/(√v̂ₜ+ε)"],
        ]
        tbl_adam = Table(adam_rows, colWidths=[1.8*cm, 8*cm, 4.7*cm])
        tbl_adam.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a0a3a")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), HexColor("#c77dff")),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME",   (0, 1), (0, -1), "Courier"),
            ("FONTNAME",   (2, 1), (2, -1), "Courier"),
            ("FONTSIZE",   (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [HexColor("#0d0d1a"), HexColor("#100d20")]),
            ("TEXTCOLOR",  (0, 1), (0, -1), HexColor("#c77dff")),
            ("TEXTCOLOR",  (1, 1), (1, -1), HexColor("#9090b8")),
            ("TEXTCOLOR",  (2, 1), (2, -1), HexColor("#7bed9f")),
            ("GRID",       (0, 0), (-1, -1), 0.4, HexColor("#1a0a3a")),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ]))
        story.append(tbl_adam)
        story.append(Paragraph(
            "Tabla A. Diccionario de variables del optimizador Adam con sus valores "
            "de configuración en el módulo experimental.",
            sty['caption']))
        story.append(Paragraph(
            "<b>Propiedad clave:</b> cuando un parámetro recibe gradientes consistentemente "
            "grandes, v̂ₜ crece y la actualización efectiva α/√v̂ₜ disminuye, evitando "
            "divergencia. Cuando los gradientes son pequeños (zona plana de la pérdida), "
            "v̂ₜ es pequeño y la tasa efectiva es mayor, impulsando la exploración. "
            "Esto hace Adam robusto a paisajes de pérdida con curvaturas heterogéneas.",
            sty['insight']))
        story.append(Spacer(1, 0.3*cm))

        # ── §7  EARLY STOPPING ────────────────────────────────────────────────
        story.append(Paragraph("§7  Early Stopping y LR Decay", sty['h1']))
        p_val = int(self.v_patience.get())
        story.append(Paragraph("Algoritmo formal de Early Stopping:", sty['math_label']))
        for line in [
            f"Inicializar: best_loss = +∞,  contador = 0,  W_best = W₀,  decay_count = 0",
            "Para cada época t = 1, 2, ..., T:",
            "  Calcular ℒ_val(Wₜ) en el conjunto de validación",
            f"  Si ℒ_val(Wₜ) < best_loss:",
            "    best_loss ← ℒ_val(Wₜ);  W_best ← Wₜ;  contador ← 0",
            "  Sino:",
            "    contador ← contador + 1",
            f"    Si contador ≥ p = {p_val}  y  decay_count < 2:",
            "      α ← α × 0.3   (decaimiento de LR)",
            "      contador ← 0;  decay_count ← decay_count + 1",
            f"    Si contador ≥ p = {p_val}  y  decay_count = 2:",
            "      Detener entrenamiento",
            "Retornar W_best",
        ]:
            story.append(Paragraph(line, sty['math']))
        story.append(Paragraph(
            "El LR decay ×0.3 es equivalente a una búsqueda de línea gruesa: "
            "cuando el optimizador se atasca en una región plana, reduce el paso "
            "para explorar con mayor resolución. Permite hasta 2 reducciones antes "
            "de declarar convergencia, balanceando exploración y explotación.",
            sty['body']))
        story.append(Spacer(1, 0.3*cm))

        # ── §8  PROBLEMA NL DE OPTIMIZACIÓN DE MEMORIA ────────────────────────
        story.append(Paragraph(
            "§8  Problema No Lineal de Optimización de Memoria", sty['h1']))
        story.append(Paragraph(
            "El optimizador de memoria resuelve un problema de optimización sobre el "
            "espacio de hiperparámetros (λ,α), donde el valor de la función objetivo "
            "requiere entrenar la red completa para cada evaluación:",
            sty['body']))
        story.append(Paragraph("Problema de optimización bilevel:", sty['math_label']))
        story.append(Paragraph(
            "min_{λ,α}  f(λ,α)  =  ℒ(W*(λ,α))  +  γ · ‖W*(λ,α)‖²",
            sty['math']))
        story.append(Paragraph(
            "sujeto a:   λ ∈ [1×10⁻⁶,  0.1]        (regularización no negativa, acotada)",
            sty['math']))
        story.append(Paragraph(
            "            α ∈ [1×10⁻⁴,  0.05]       (tasa de aprendizaje positiva, acotada)",
            sty['math']))
        story.append(Paragraph(
            "donde:  W*(λ,α) = argmin_W [ BCE(W) + (λ/2)·‖W‖² ]  con optimizador Adam",
            sty['math']))
        story.append(Paragraph(
            f"γ = {gam_v:.3f}  (peso de penalización de memoria configurado por el usuario)",
            sty['math']))
        story.append(Paragraph(
            "<b>Naturaleza del problema:</b> f(λ,α) es no lineal, no convexo en general, "
            "y su gradiente ∂f/∂(λ,α) no está disponible analíticamente (depende de la "
            "solución del problema interno de entrenamiento). El uso de L-BFGS-B con "
            "diferencias finitas aproxima numéricamente el gradiente exterior.",
            sty['def_box']))
        story.append(Paragraph(
            "<b>Cotas de la región factible:</b> Las cotas λ ∈ [10⁻⁶, 0.1] y α ∈ [10⁻⁴, 0.05] "
            "fueron calibradas empíricamente para evitar: (a) λ demasiado grande que "
            "anule los pesos (W*→0, precisión→50%); (b) α demasiado grande que cause "
            "divergencia del optimizador interno; (c) valores negativos que carecen de "
            "interpretación en el contexto de regularización y tasa de aprendizaje.",
            sty['insight']))
        story.append(Spacer(1, 0.3*cm))

        # ── §9  ALGORITMO L-BFGS-B ────────────────────────────────────────────
        story.append(Paragraph("§9  Algoritmo L-BFGS-B — Quasi-Newton con Cotas", sty['h1']))
        story.append(Paragraph(
            "L-BFGS-B (Limited-memory Broyden–Fletcher–Goldfarb–Shanno with Bounds) es un "
            "método quasi-Newton de memoria limitada para optimización con restricciones de "
            "caja (box constraints). Aproxima la inversa del Hessiano H⁻¹ usando los últimos "
            "m pares curvatura (sₖ, yₖ) sin almacenar explícitamente la matriz completa.",
            sty['body']))
        story.append(Paragraph("Iteración L-BFGS-B:", sty['math_label']))
        for line in [
            "Dado punto (λₖ, αₖ) en la región factible:",
            "1. Evaluar f(λₖ,αₖ): entrenar W*(λₖ,αₖ) por 200 pasos Adam",
            "2. Estimar gradiente: ∇f ≈ [f(λ+δ,α)−f(λ,α)]/δ  (diferencias finitas)",
            "3. Calcular dirección de descenso: dₖ = −H⁻¹ₖ · ∇fₖ  (aproximación L-BFGS)",
            "4. Proyectar dₖ sobre la región factible (cotas de caja)",
            "5. Búsqueda de línea con condición de Wolfe para αₖ₊₁",
            "6. (λₖ₊₁, αₖ₊₁) = (λₖ, αₖ) + αₖ₊₁ · dₖ",
            "7. Actualizar aproximación H⁻¹ₖ₊₁ con par (sₖ = λₖ₊₁−λₖ, yₖ = ∇fₖ₊₁−∇fₖ)",
            "8. Si ‖∇fₖ‖ < tol o max_iter alcanzado: detener y retornar (λ*, α*)",
        ]:
            story.append(Paragraph(line, sty['math']))
        story.append(Paragraph(
            "La convergencia local del método es superlineal (quasi-cuadrática) bajo "
            "condiciones de suavidad de f. Las cotas de caja se manejan mediante la "
            "proyección de la dirección de búsqueda sobre el conjunto factible, "
            "garantizando λ ≥ 10⁻⁶ y α ≥ 10⁻⁴ en todo momento.",
            sty['body']))
        story.append(Spacer(1, 0.3*cm))

        # ── §10  INTERPRETACIÓN GLOBAL ─────────────────────────────────────────
        story.append(Paragraph("§10  Interpretación Global del Sistema", sty['h1']))
        story.append(Paragraph(
            "El módulo experimental implementa un sistema de optimización bilevel: "
            "el nivel superior minimiza f(λ,α) sobre el espacio de hiperparámetros, "
            "y el nivel inferior entrena W*(λ,α) para cada par (λ,α) evaluado. "
            "Formalmente:",
            sty['body']))
        story.append(Paragraph(
            "Nivel superior (L-BFGS-B):  min_{λ∈Λ, α∈A}  f(λ,α) = ℒ(W*(λ,α)) + γ·‖W*(λ,α)‖²",
            sty['math']))
        story.append(Paragraph(
            "Nivel inferior (Adam):       W*(λ,α) = argmin_W [ BCE(W) + (λ/2)·‖W‖² ]",
            sty['math']))
        story.append(Paragraph(
            "Este acoplamiento bilevel es la clave de la metodología: el optimizador externo "
            "no necesita conocer la estructura interna de la red; solo evalúa pares "
            "(entrada: hiperparámetros) → (salida: calidad de la red entrenada). "
            "Esto hace el método agnóstico a la arquitectura y potencialmente aplicable "
            "a cualquier red entrenable por descenso de gradiente.",
            sty['insight']))
        story.append(Paragraph(
            "<b>Propiedades del espacio de optimización externo:</b> f(λ,α) no es convexa "
            "en general — puede tener múltiples mínimos locales dependiendo del dataset y "
            "la inicialización de la red. Sin embargo, en la práctica, la región de "
            "hiperparámetros explorada por L-BFGS-B es suficientemente pequeña (region "
            "factible acotada) para que el mínimo encontrado sea globalmente bueno "
            "dentro de los bounds configurados.",
            sty['body']))

        if self.opt_lam is not None:
            story.append(Paragraph(
                f"<b>Solución encontrada en esta sesión:</b> (λ*, α*) = "
                f"({self.opt_lam:.6f}, {self.opt_lr:.6f}). "
                f"Evaluaciones del optimizador: {len(self.opt_iters)}. "
                f"Reducción de f: de {self.opt_iters[0]['f']:.5f} a "
                f"{self.opt_iters[-1]['f']:.5f} "
                f"({((self.opt_iters[0]['f']-self.opt_iters[-1]['f'])/max(abs(self.opt_iters[0]['f']),1e-9)*100):.1f}% de mejora).",
                sty['insight']))

        story.append(Spacer(1, 0.4*cm))
        story.append(HRFlowable(width="100%", thickness=0.8,
                                color=HexColor("#1a0a3a"), spaceAfter=6))
        story.append(Paragraph(
            "Documento generado automáticamente por el Módulo Experimental v4.0. "
            "Autor: Perplexity Computer. Las ecuaciones reflejan la implementación "
            "exacta en el código fuente (neural_opt.py).",
            sty['ref']))

        try:
            doc.build(story,
                      onFirstPage=hf_math,
                      onLaterPages=hf_math)
        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            self.lbl_status.config(text=f"Error PDF: {str(e)[:80]}")
            messagebox.showerror("Error al generar PDF",
                f"No se pudo crear el modelo:\n\n{str(e)}\n\n"
                f"Ruta intentada: {path}")
            print("[PDF ERROR]\n" + err_msg)
            return

        self.lbl_status.config(
            text=f"✓ Modelo Matemático PDF: {os.path.basename(path)}")
        messagebox.showinfo("PDF generado",
            f"Modelo Matemático guardado en:\n{path}")


if __name__ == "__main__":
    app = App()
    app.mainloop()

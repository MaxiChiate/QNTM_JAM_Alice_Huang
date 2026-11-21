# Reparto de tareas: Gonza, Fede y Maxi

## La idea para no pisarse

El problema se corta en tres piezas con una **interfaz fija** entre ellas. Cada uno es dueño de sus archivos, y nadie edita los archivos de otro: si algo de otro hace falta cambiar, se le pide.

```
 Gonza: química            Fede: ansatz + optimización       Maxi: medición + entrega
 src/hamiltonian.py  ───►  src/ansatz.py, src/vqe.py   ───►  src/measurement.py, src/analysis.py
 "dame d, te doy H"        "dame H, te doy circuito + θ*"    "dame circuito + θ*, te doy E ± σ"
```

| Archivo | Dueño |
|---|---|
| `src/hamiltonian.py` | Gonza |
| `src/ansatz.py`, `src/vqe.py` | Fede |
| `src/measurement.py`, `src/analysis.py` | Maxi |
| `notebooks/gonza_*.ipynb` | Gonza (pruebas propias) |
| `notebooks/fede_*.ipynb` | Fede (pruebas propias) |
| `notebooks/maxi_*.ipynb` | Maxi (pruebas propias) |
| `notebooks/narrative/*.md` | cada uno escribe el texto de su sección |
| `tools/build_notebook.py` → `notebooks/final.ipynb` | Maxi (el `.ipynb` final se genera, no se edita a mano) |

### La entrega es UN notebook de Colab autocontenido

El jurado abre el notebook en Colab desde la carpeta de Drive y lo corre de punta a punta. **No tiene el repo ni `src/`**, así que el notebook final **no puede hacer `import` de nuestros `.py`**.

- Los `.py` son **solo para desarrollar** sin conflictos de git.
- El notebook final se **genera** con `tools/build_notebook.py`, que mete el contenido de cada `src/*.py` en celdas de código, intercaladas con las celdas de texto de `notebooks/narrative/*.md`. Así hay una sola fuente de verdad y nadie copia y pega a mano.
- Reglas para que esto funcione:
  - Cada `.py` importa solo librerías externas (`qiskit`, `pyscf`, `numpy`...), **nunca** `from src.x import ...`. En el notebook todo queda en el mismo namespace.
  - El orden de las celdas es el de las dependencias: `hamiltonian.py` → `ansatz.py` → `vqe.py` → `measurement.py` → `analysis.py`.
  - `ruido.py` también se pega como celda, o se descarga en la celda de instalación si los organizadores lo publican con una URL.
  - Nada de rutas locales. Si se cachean resultados (θ óptimos), el cache va **dentro del notebook** (un dict en una celda) o se recalcula. Lo más seguro es que el notebook recalcule todo.
- **El tiempo de ejecución en Colab importa**: optimizar 12 a 15 puntos más las mediciones tiene que terminar en un tiempo razonable en el runtime gratis. Midan cuánto tarda cada pieza desde el principio.

**Reglas de git**:
- Una rama por persona (`gonza/...`, `fede/...`, `maxi/...`) y PR a `main`.
- Los notebooks se mergean muy mal: **el código compartido va en `.py`** y cada notebook es de una sola persona.
- Cuando alguien cambia una firma de la interfaz, avisa en el grupo **antes** de mergear.

---

## La interfaz (lo primero que hay que acordar)

```python
# src/hamiltonian.py (Gonza)
DISTANCES = [...]   # la grilla de d en Å, la define Gonza

def build_problem(d: float) -> dict:
    """H4 lineal, STO-3G, separación d (Å)."""
    return {
        "d": d,
        "hamiltonian": SparsePauliOp,  # YA incluye la repulsión nuclear
        "n_qubits": int,
        "hf_bitstring": str,           # convención Qiskit, ej. "00110011"
        "e_hf": float, "e_ccsd": float, "e_fci": float,
        "ccsd_t1": ndarray, "ccsd_t2": ndarray,  # para inicializar el ansatz
        "number_op": SparsePauliOp,    # N (tiene que dar 4)
        "s2_op": SparsePauliOp,        # S² (tiene que dar 0)
    }

# src/ansatz.py / src/vqe.py (Fede)
def build_ansatz(problem: dict, kind: str) -> QuantumCircuit: ...
def optimize(problem: dict, ansatz: QuantumCircuit, theta0=None) -> dict:
    return {"theta": ndarray, "e_exact": float, "n_iters": int}
def count_cz(circuit: QuantumCircuit) -> int:
    """Transpila a ['cz','sx','x','rz'] con conectividad en línea y cuenta cz."""

# src/measurement.py (Maxi)
def measure_energy(circuit, theta, hamiltonian, total_shots: int) -> dict:
    return {"energy": float, "sigma": float, "shots": int}
```

Los resultados de cada punto se guardan en `results/<kind>_d<d>.json` (θ óptimo, energía, σ, shots, cz). Así nadie tiene que re-correr lo del otro.

---

## Fase 0: los tres juntos (~1 h)

- [ ] Instalar el entorno local: `pip install qiskit qiskit-aer pyscf qiskit-nature` (más lo que pida `ruido.py`). Probar que corra `import pyscf, qiskit_aer`. **Anotar las versiones exactas** (`pip freeze`) para fijarlas en la celda de instalación de Colab, así el jurado corre lo mismo que nosotros.
- [ ] Crear la estructura `src/`, `notebooks/`, `results/` y los archivos con las **firmas vacías** de arriba (stubs que devuelvan `NotImplemented`). A partir de acá cada uno trabaja en lo suyo.
- [ ] Leer juntos las secciones 0, 4 y 8 de [theory_notes.md](theory_notes.md).

---

## Gonza: la química y el hamiltoniano

**Leer**: apunte, secciones 3 a 7, y [Química Cuántica, Parte 1](../workshops/Química_Cuántica,_Parte_1.ipynb) completo (sobre todo PySCF y Jordan-Wigner). Las [soluciones](../workshops/Soluciones_Química_Cuántica.ipynb) tienen código reutilizable.

1. [ ] **Entregable urgente (desbloquea a Fede)**: `build_problem(d)` andando para un solo `d`, aunque sea sin CCSD. Avisar apenas funcione.
2. [ ] Validar el hamiltoniano: el autovalor más chico de `hamiltonian.to_matrix()` restringido a 4 electrones tiene que coincidir con `e_fci` de PySCF hasta 1e-8. **Si no coincide, no sigue nada.** El error típico es olvidar la repulsión nuclear.
3. [ ] Agregar CCSD, las amplitudes `t1`/`t2`, `number_op` y `s2_op`.
4. [ ] Definir `DISTANCES`: unos 12 a 15 puntos entre 0,6 y 3,0 Å, con más densidad cerca del mínimo (~0,9 Å) y en la zona ≥ 1,8 Å. Que incluya exactamente 1,8 y 3,0.
5. [ ] Gráfico de referencia: HF, CCSD y FCI vs `d`, y el error de CCSD en mHa vs `d` con la línea de 1,6 mHa. Así sabemos **cuánto hay que ganarle a CCSD** en cada punto.
6. [ ] Tabla de los coeficientes $c_k$ del hamiltoniano: cuántos términos hay, cuántos grupos que conmutan y cómo se distribuyen los $|c_k|$. Se la pasa a Maxi para repartir los shots.

**Después (extras)**: reducir a 6 qubits con el mapeo de paridad y tapering (extra "menos qubits"), y otra geometría de H₄ (extra "otra geometría").

---

## Fede: el ansatz y la optimización (sin ruido)

**Leer**: apunte, secciones 8 y 9; [Química Cuántica, Parte 2](../workshops/Química_Cuántica,_Parte_2.ipynb), sección VQE; [Clase 2, Parte 2](../workshops/Clase_2,_Parte_2_Computación_Cuántica.ipynb), ejercicios 8 y 9 (circuitos parametrizados y transpilación).

Mientras Gonza termina el paso 1, se puede practicar con H₂ del notebook de química.

1. [ ] Circuito del estado HF (aplicar `X` en los qubits ocupados) y `count_cz`.
2. [ ] **Ansatz UCCSD** (`qiskit_nature`) inicializado en HF. Optimizarlo con **statevector exacto** (está permitido para buscar θ) en un `d` cerca del equilibrio y en uno estirado (2,4 Å). Comparar con `e_fci` y `e_ccsd`. **Primer hito: ¿le ganamos a CCSD a 2,4 Å?**
3. [ ] Contar las `cz` de UCCSD transpilado. Van a ser muchas; anotarlo, es el punto de partida.
4. [ ] Barrido de la curva completa con *warm start* (los θ de un `d` como punto inicial del siguiente), guardando en `results/`.
5. [ ] Probar al menos un ansatz más barato y comparar precisión contra `cz`: hardware-efficient, k-UpCCGSD o ADAPT-VQE. **La meta es la misma precisión con muchas menos `cz`.**
6. [ ] Anotar qué optimizadores se probaron y cuál anduvo (el jurado quiere ver "qué no funcionó").

**Después (extras)**: tabla comparativa de ansätze (extra "comparar ansätze").

---

## Maxi: medición, estadística y entrega

**Leer**: apunte, secciones 2, 9, 10 y 11; [Clase 2, Parte 2](../workshops/Clase_2,_Parte_2_Computación_Cuántica.ipynb), ejercicios 11 a 14 (Sampler, Estimator, mitigación).

Para empezar sin esperar a nadie: probar con un circuito cualquiera de 8 qubits y un `SparsePauliOp` inventado.

1. [ ] `measure_energy` con `SamplerV2` de `qiskit_aer` y shots explícitos:
   - agrupar los Paulis que conmutan (`SparsePauliOp.group_commuting(qubit_wise=True)`),
   - un circuito por grupo con las rotaciones de base,
   - calcular $\langle P_k\rangle$ a partir de los bitstrings, y $E$ y $\sigma$.
2. [ ] **Validar $\sigma$**: repetir la medición unas 30 veces con semillas distintas y verificar que la desviación estándar de las 30 energías coincida con la $\sigma$ reportada. Esto es lo que mira el jurado en "barras de error bien calculadas".
3. [ ] Reparto de shots: comparar un reparto uniforme contra uno proporcional a $|c_k|$. Calcular **cuántos shots hacen falta para $\sigma < 1$ mHa**.
4. [ ] Indicadores de confiabilidad: medir con el mismo código $\langle N\rangle$, $\langle S^2\rangle$ y la varianza $\langle H^2\rangle - \langle H\rangle^2$ (sección 11 del apunte).
5. [ ] `analysis.py`: a partir de `results/`, armar la tabla ($d$, $E$, $\sigma$, shots, `cz`), el gráfico de la curva con barras de error contra CCSD y FCI, el gráfico de error en mHa, el ajuste de $R_e$ (parábola o spline alrededor del mínimo), $D_e$ y el chequeo $|E-E_\text{FCI}|+\sigma < |E_\text{CCSD}-E_\text{FCI}|$ para $d \ge 1{,}8$.
6. [ ] `tools/build_notebook.py` (con `nbformat`): genera `notebooks/final.ipynb` con la celda de instalación (`!pip install qiskit qiskit-aer pyscf ...` con versiones fijadas), el código de cada `src/*.py` en orden y las celdas de texto de `notebooks/narrative/`.
7. [ ] **Probarlo en Colab de verdad lo antes posible**, aunque sea con los stubs: subir el `.ipynb` generado, `Entorno de ejecución > Reiniciar y ejecutar todo`, y anotar el tiempo total. Repetirlo cada vez que se integra algo grande.
8. [ ] Al final: carpeta de Google Drive compartida con el jurado, con el notebook dentro y probado desde esa copia.

**Después (extras)**: curva con el ruido de `ruido.py`, y ZNE más mitigación de readout.

---

## Fase 2: integración (los tres)

- [ ] Correr toda la cadena para 2 puntos (equilibrio y 2,4 Å): `build_problem` → `optimize` → `measure_energy` → tabla.
- [ ] Si anda, correr la curva completa.
- [ ] Decidir juntos el ansatz final y el presupuesto de shots por punto.
- [ ] Escribir la parte de "qué decidimos, por qué y qué no funcionó". Cada uno escribe su parte y Maxi la junta.

## Fase 3: extras (elegir según el tiempo)

| Extra | Quién lo agarra naturalmente |
|---|---|
| Escribir `src/ruido.py` (modelo de ruido con `qiskit_aer.noise`: `cz`, `sx`, `x`, `rz`, línea, 0,1 % por `cz`, 0,005 % por compuerta de un qubit) y correr la curva con él y con `FakeFez` | Maxi |
| ZNE y mitigación de readout | Maxi + Fede (necesita plegar `cz` en el circuito) |
| Menos qubits (tapering a 6) | Gonza |
| Comparar ansätze | Fede |
| Otra geometría | Gonza |
| VQE vs SQD | quien termine primero |

## Puntos de control

1. **Interfaz acordada y stubs en `main`.** Fin de la fase 0.
2. **`build_problem` validado contra FCI.** Gonza → desbloquea a Fede.
3. **UCCSD le gana a CCSD a 2,4 Å sin ruido.** Fede → confirma que el plan anda.
4. **$E \pm \sigma$ con $\sigma$ validada.** Maxi.
5. **Curva completa con tabla.** Los tres → el desafío central está cumplido; de ahí en más todo es mejora.

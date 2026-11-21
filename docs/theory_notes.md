# Apunte: la teoría detrás de la consigna

Para entender [CONSIGNA.md](../CONSIGNA.md) sin saber química ni cuántica, y con poca álgebra lineal.
La idea es explicar todo con analogías de programación. Al final de cada sección dice qué notebook de [workshops/](../workshops/) la desarrolla en serio.

---

## 0. El desafío en una pantalla

```
 geometría (4 H en fila, separados d Å)
        │
        ▼
 PySCF: Hartree-Fock  ──►  integrales h_pq, h_pqrs  ──►  Hamiltoniano fermiónico (a†, a)
        │                                                     │
        │                                                     ▼ Jordan-Wigner
        │                                        H = Σ c_k P_k   (suma de cadenas de Pauli, 8 qubits)
        │                                                     │
        ▼                                                     ▼
 referencias clásicas:                      VQE:  circuito |ψ(θ)⟩  ──►  Estimator mide ⟨H⟩ ± σ
 E_HF, E_CCSD, E_FCI (exacta)                        ▲                       │
                                                     └──── optimizador ◄─────┘ (ajusta θ)
        │                                                     │
        └──────────────► comparar para cada d ◄───────────────┘
                         curva E(d), R_e, D_e, errores vs FCI
```

Traducido a programación:

- Hay una matriz gigante $H$ (el **hamiltoniano**). La energía de la molécula es su **autovalor más chico**.
- Para 4 hidrógenos esa matriz es de $256 \times 256$, y la computadora clásica la diagonaliza en milisegundos (eso es **FCI**, la "respuesta exacta"). El desafío es encontrar ese número **con un algoritmo cuántico (VQE)**, como si la matriz fuera demasiado grande para diagonalizarla.
- VQE es un **loop de optimización**: un circuito con parámetros $\theta$ prepara un estado, se mide su energía (con ruido estadístico) y un optimizador clásico (tipo `scipy.optimize.minimize`) mueve $\theta$ para bajar la energía.
- Hay que hacerlo para al menos 10 valores de $d$ y armar una curva.
- Se gana si en la zona estirada ($d \ge 1{,}8$ Å) el error de VQE (más su barra de error) es menor que el de **CCSD**, un método clásico aproximado que ahí falla.
- También se evalúa cuánto cuesta: cantidad de compuertas `cz` y cantidad de mediciones (shots).

---

## 1. Kit mínimo de álgebra lineal (versión numpy)

### Números complejos
$z = a + bi$, con $i^2 = -1$. El **conjugado** es $z^* = a - bi$. El **módulo al cuadrado** es $|z|^2 = z^* z = a^2 + b^2$, que siempre es real y no negativo.
En cuántica los números son complejos, pero lo que se termina midiendo siempre sale de un $|z|^2$, así que es real.

### Vectores y matrices
- Un **vector columna** es un `np.array` de forma `(n, 1)`.
- La **transpuesta** $A^T$ intercambia filas y columnas.
- **El superíndice de la cruz, $A^\dagger$** (se lee "A daga"): **transpuesta + conjugar cada elemento**. Es `A.conj().T`. También se llama *conjugado hermítico* o *adjunto*. Si todo es real, $A^\dagger = A^T$.

  $$\begin{pmatrix} 1 & i \\ 2 & 3-i \end{pmatrix}^\dagger = \begin{pmatrix} 1 & 2 \\ -i & 3+i \end{pmatrix}$$

  Ojo: en $a_p^\dagger$ (sección 5) es la misma operación aplicada a un operador. $a_p$ saca un electrón y su daga $a_p^\dagger$ hace lo inverso: lo pone.

### Notación de Dirac (bra-ket)
Es solo notación para vectores:

| Notación | Qué es | numpy |
|---|---|---|
| $\lvert\psi\rangle$ ("ket psi") | vector columna | `psi` |
| $\langle\psi\rvert$ ("bra psi") | su daga: vector fila conjugado | `psi.conj().T` |
| $\langle\phi\vert\psi\rangle$ ("braket") | producto interno, un número | `np.vdot(phi, psi)` |
| $\langle\psi\rvert A \lvert\psi\rangle$ | "sándwich": un número | `psi.conj().T @ A @ psi` |
| $\lvert\psi\rangle\langle\phi\rvert$ | producto externo, una matriz | `np.outer(psi, phi.conj())` |

### Autovalores y autovectores
$A v = \lambda v$: hay vectores $v$ que la matriz no rota, solo los estira por un factor $\lambda$. `np.linalg.eigh(A)` los encuentra todos.
**La energía de la molécula es el autovalor más chico del hamiltoniano.** Toda la consigna gira alrededor de esta frase.

### Dos tipos de matrices que importan
- **Hermítica**: $A = A^\dagger$. Sus autovalores son **reales**, así que representan cosas medibles: los **observables** (energía, espín, etc.). El hamiltoniano es hermítico.
- **Unitaria**: $U^\dagger U = I$. Conserva la norma (no "pierde probabilidad"). **Toda compuerta cuántica es una matriz unitaria**, y un circuito es un producto de unitarias.

### Producto tensorial $\otimes$
Combina sistemas: $A \otimes B$ es `np.kron(A, B)`. Con 2 qubits de dimensión 2 se obtiene uno de dimensión 4, con $n$ qubits uno de dimensión $2^n$. **Por esto la memoria clásica explota**: el tamaño crece exponencialmente.
La notación $\{I,X,Y,Z\}^{\otimes n}$ significa "un producto tensorial de $n$ matrices, cada una elegida de entre $I, X, Y, Z$", por ejemplo $X \otimes Z \otimes I \otimes Y$.

> Notebooks: Clase 1, Parte 1 (kets, bras, operadores) y Clase 2, Parte 1 (producto tensorial).

---

## 2. Kit mínimo de cuántica

### Qubit y superposición
Un qubit es un vector de 2 componentes complejas: $\lvert\psi\rangle = \alpha\lvert 0\rangle + \beta\lvert 1\rangle$, con $|\alpha|^2 + |\beta|^2 = 1$.
- $\lvert 0\rangle = (1, 0)^T$ y $\lvert 1\rangle = (0, 1)^T$.
- Al **medir** sale 0 con probabilidad $|\alpha|^2$ y 1 con probabilidad $|\beta|^2$. La medición destruye la superposición, así que para estimar probabilidades hay que **preparar y medir muchas veces**. Cada repetición es un **shot**.
- $n$ qubits: un vector de $2^n$ amplitudes, una por cada bitstring.

### Las matrices de Pauli
$$I=\begin{pmatrix}1&0\\0&1\end{pmatrix},\quad X=\begin{pmatrix}0&1\\1&0\end{pmatrix},\quad Y=\begin{pmatrix}0&-i\\i&0\end{pmatrix},\quad Z=\begin{pmatrix}1&0\\0&-1\end{pmatrix}$$
- $X$ es el NOT: intercambia $\lvert 0\rangle$ y $\lvert 1\rangle$.
- $Z$ deja $\lvert 0\rangle$ igual y le cambia el signo a $\lvert 1\rangle$.
- Son hermíticas **y** unitarias, y sus autovalores son $\pm 1$.

Una **cadena de Pauli** es un producto tensorial de Paulis, como `"XZZY"` en Qiskit (`SparsePauliOp`).
⚠️ En Qiskit el string se lee **de derecha a izquierda**: en `"XIZ"` la $Z$ actúa sobre el qubit 0.

### Valor esperado: lo que mide el Estimator
$\langle P\rangle = \langle\psi\rvert P\lvert\psi\rangle$ es el **promedio** de lo que daría medir $P$ muchas veces.
Ejemplo con $Z$: cada shot da $+1$ (salió 0) o $-1$ (salió 1), así que $\langle Z\rangle \approx (N_0 - N_1)/N$.
Para una cadena como $Z\otimes Z$ se mide la **paridad** del bitstring: $+1$ si hay una cantidad par de unos y $-1$ si es impar.
Para medir $X$ o $Y$ en vez de $Z$ se agrega una rotación antes de medir ($H$ para $X$, $S^\dagger H$ para $Y$). El Estimator lo hace solo.

**Linealidad**: si $H = \sum_k c_k P_k$, entonces
$$\langle H\rangle = \sum_k c_k \langle P_k\rangle.$$
Por eso se escribe el hamiltoniano como suma de Paulis: **cada término se puede medir en un circuito** y después se suman.

### Error estadístico (la $\sigma$ de la consigna)
Con $N$ shots, un promedio tiene error estándar $\propto 1/\sqrt{N}$. Para un Pauli (valores $\pm1$), la varianza de un shot es $1 - \langle P\rangle^2$. Si se mide cada término por separado con $N_k$ shots:
$$\sigma_E^2 \approx \sum_k c_k^2\,\frac{1 - \langle P_k\rangle^2}{N_k}$$
Consecuencias prácticas:
- **Para bajar $\sigma$ a la mitad hacen falta 4 veces más shots.**
- Conviene darle **más shots a los términos con $|c_k|$ grande**. El reparto óptimo es $N_k \propto |c_k|\sqrt{1-\langle P_k\rangle^2}$. Esto es "manejo del presupuesto" y el jurado lo mira.
- Los Paulis que **conmutan** (por ejemplo, todos los que solo tienen $I$ y $Z$) se pueden medir **con el mismo circuito**: agruparlos ahorra shots. Ojo, dentro de un grupo hay covarianzas, así que la fórmula de arriba pasa a ser una aproximación.

> Notebooks: Clase 1, Parte 2 (mediciones, valores medios, Pauli) y Clase 2, Parte 2 (Sampler, Estimator, ejercicios 11 a 13).

---

## 3. Kit mínimo de química

### Unidades
| Símbolo | Qué es | Equivalencias |
|---|---|---|
| **Å** (la "A con el círculo", *ångström*) | unidad de distancia atómica | $1\text{ Å} = 10^{-10}$ m $\approx 1{,}89$ bohr. Un enlace típico mide alrededor de 1 Å. |
| bohr ($a_0$) | unidad "natural" de distancia | $\approx 0{,}529$ Å. PySCF usa Å por defecto, pero algunos códigos usan bohr: **revisen siempre la unidad.** |
| **Ha** (Hartree) | unidad natural de energía | $1\text{ Ha} \approx 27{,}2$ eV $\approx 627{,}5$ kcal/mol |
| **mHa** | miliHartree, $10^{-3}$ Ha | — |
| **precisión química** | 1,6 mHa $\approx$ 1 kcal/mol | El error por debajo del cual un cálculo es útil para predecir experimentos. |

Las energías de la molécula salen **negativas** (unos −2 Ha para H₄) porque se mide respecto de "todo separado en el infinito". Lo que importa son las **diferencias**.

### La molécula como problema de física
- **Núcleos quietos (Born-Oppenheimer)**: los núcleos son miles de veces más pesados que los electrones, así que se fijan en su lugar y solo se resuelve el problema de los electrones. Cada distancia $d$ es entonces **un problema independiente**, con su propio hamiltoniano. Por eso la consigna dice "construirlo una vez por punto de la curva".
- **Curva de energía**: $E(d)$. Si los átomos están muy juntos se repelen y la energía sube; si están lejos no hay enlace y la curva se aplana; en el medio está el mínimo.
  - $R_e$: la $d$ del mínimo (**distancia de equilibrio**).
  - $D_e = E(3{,}0\text{ Å}) - E(R_e)$: cuánto cuesta romper la molécula (**energía de disociación**). Es positiva.
  - Para sacar un $R_e$ preciso hay que poner varios puntos cerca del mínimo y **ajustar una parábola o spline**, no quedarse con el punto más bajo de la grilla.

### Orbitales, espín y la base STO-3G
- Un **orbital** es una "zona" donde puede estar un electrón; matemáticamente es una función.
- Cada electrón tiene **espín** ↑ o ↓ (también se escriben $\alpha$ y $\beta$). Un **espín-orbital** es un orbital junto con un espín, y por el **principio de Pauli** puede tener **0 o 1 electrones**. **Es un bit.**
- La **base** es el "vocabulario" de funciones con que se describen los orbitales. **STO-3G** es la más chica posible: una función por orbital atómico (cada una aproxima un orbital tipo Slater con 3 gaussianas, de ahí el nombre).

**Las cuentas para H₄ en STO-3G**:
- 4 átomos × 1 orbital = **4 orbitales espaciales** = **8 espín-orbitales** = **8 qubits** con Jordan-Wigner.
- 4 electrones: 2 ↑ y 2 ↓.
- Espacio total: $2^8 = 256$ bitstrings, pero solo $\binom{4}{2}\cdot\binom{4}{2} = 36$ tienen el número correcto de electrones de cada espín (esto es lo que se aprovecha en "menos qubits" o "simetrías").

> Notebook: Química Cuántica, Parte 1 (secciones "Átomos, electrones y orbitales", "Nuestra primera molécula" y "Electrones como bitstrings").

---

## 4. El hamiltoniano $\hat H$

**Qué es**: el operador (una matriz hermítica) que codifica **toda la energía** del sistema: la cinética de los electrones, la atracción electrón-núcleo, la repulsión electrón-electrón y la repulsión núcleo-núcleo.
El **sombrero** en $\hat H$ solo indica "esto es un operador (una matriz), no un número".

**Para qué sirve**:
- Sus **autovalores** son las energías posibles de la molécula.
- El más chico, $E_0$, es la energía del **estado fundamental**, el estado en que la molécula "quiere" estar. Es el número que se busca para cada $d$.
- Su **autovector** asociado $\lvert\psi_0\rangle$ es ese estado.

**Leyendo la fórmula de la consigna**:
$$\hat{H} = \underbrace{\sum_{pq} h_{pq}\, a_p^\dagger a_q}_{\text{1 electrón}} + \underbrace{\frac{1}{2}\sum_{pqrs} h_{pqrs}\, a_p^\dagger a_q^\dagger a_r a_s}_{\text{2 electrones}} \;\; (+\, E_\text{nuc})$$

- $p, q, r, s$ son índices de espín-orbital (0 a 7). Las sumas son for anidados.
- $h_{pq}$ (**integrales de un electrón**) son números que PySCF calcula a partir de Hartree-Fock. Miden cuánto le "cuesta" a un electrón estar en $q$ o pasar de $q$ a $p$ (energía cinética + atracción a los núcleos).
- $h_{pqrs}$ (**integrales de dos electrones**; en los notebooks se llaman $g_{pqrs}$) miden la repulsión entre pares de electrones.
- $a_p^\dagger a_q$ se lee "**sacá un electrón de $q$ y ponelo en $p$**" (sección 5). El primer término mueve electrones de a uno; el segundo, de a dos.
- $E_\text{nuc}$: la repulsión entre los núcleos es una constante para cada $d$. **No se olviden de sumarla** (Qiskit/PySCF la reportan aparte; sin ella la curva sale mal).

**Por qué es difícil**: con $n$ espín-orbitales la matriz es $2^n \times 2^n$. Para H₄ son 256, nada. Para la nitrogenasa serían cientos de orbitales, y $2^{\text{cientos}}$ no entra en ninguna computadora clásica.

---

## 5. Segunda cuantización: electrones como bits

Un estado "clásico" de los electrones (una **configuración** o *determinante de Slater*) es un **bitstring de ocupación**: el bit $p$ vale 1 si el espín-orbital $p$ tiene un electrón.
Para H₄, el estado de Hartree-Fock (los 2 orbitales más bajos llenos, para ↑ y para ↓) es, con la convención de Qiskit (↑ a la derecha, ↓ a la izquierda, orbitales más bajos a la derecha):
```
 ↓↓↓↓ ↑↑↑↑
 0011 0011
```

- **Operador de creación $a_p^\dagger$**: pone el bit $p$ en 1. Si ya estaba en 1, el resultado es el vector cero (no se pueden poner dos electrones en el mismo lugar).
- **Operador de aniquilación $a_p$**: pone el bit $p$ en 0. Si ya estaba en 0, da cero.
- Son daga uno del otro, de ahí la cruz.
- **El detalle cuántico**: los electrones son *fermiones*, y al reordenarlos aparece un **signo −**. Concretamente, al aplicar $a_p^\dagger$ el resultado se multiplica por $(-1)^{\text{cantidad de unos antes de } p}$. Ese signo es lo único raro, y es lo que obliga a hacer Jordan-Wigner con cuidado.

El **estado exacto** (FCI) no es un solo bitstring sino una **superposición** de los 36 bitstrings válidos con distintos pesos. Es exactamente un estado de 8 qubits. Por eso esto encaja tan bien en una computadora cuántica.

---

## 6. Jordan-Wigner: de fermiones a Paulis

Se asigna **un qubit por espín-orbital**: $\lvert 1\rangle$ = ocupado y $\lvert 0\rangle$ = vacío. Hay que traducir $a_p$ y $a_p^\dagger$ a matrices de Pauli.
- Con un solo qubit: $a = \lvert 0\rangle\langle 1\rvert = \frac{X + iY}{2}$ y $a^\dagger = \lvert 1\rangle\langle 0\rvert = \frac{X - iY}{2}$.
- Con varios qubits, para reproducir el signo fermiónico se agregan $Z$ en todos los qubits anteriores (cada $Z$ aporta un −1 si ese qubit está ocupado, y eso cuenta la paridad):

$$a_p^\dagger = Z_0 \otimes Z_1 \otimes \cdots \otimes Z_{p-1} \otimes \tfrac{X_p - iY_p}{2} \otimes I \otimes \cdots$$

Si se reemplaza esto en la fórmula de la sección 4 y se expande, queda
$$\hat H = \sum_k c_k P_k, \qquad P_k \in \{I,X,Y,Z\}^{\otimes 8}$$
una lista de pares (coeficiente real $c_k$, cadena de Pauli $P_k$). Para H₄ son del orden de cien o doscientos términos (chequéenlo con `len(H)`). Cada uno se mide con un circuito.

**Alternativas**: el mapeo de **paridad** junto con la eliminación de qubits por simetría (*qubit tapering*) baja de 8 a 6 qubits, porque el número de electrones ↑ y ↓ es fijo. Esto es el extra "menos qubits".

> Notebook: Química Cuántica, Parte 1 (sección "De electrones a qubits").

---

## 7. Los métodos clásicos de referencia

| Método | Idea | Exacto | Costo | Comportamiento |
|---|---|---|---|---|
| **HF** (Hartree-Fock) | Cada electrón siente solo el *promedio* de los otros. El resultado es **un solo bitstring**. | No | barato | bien cerca del equilibrio, muy mal estirado |
| **CCSD** | HF corregido con excitaciones simples y dobles: $\lvert\psi\rangle = e^{T_1+T_2}\lvert\text{HF}\rangle$ | No | medio, polinomial | muy bien cerca del equilibrio, falla estirado |
| **FCI** (Full Configuration Interaction) | Diagonalizar $\hat H$ completo en todos los bitstrings | **Sí** (dentro de la base) | exponencial | la referencia contra la que se compara todo |

- **Energía de correlación** $= E_\text{FCI} - E_\text{HF}$: lo que HF se pierde por ignorar que los electrones se esquivan entre sí.
- **CCSD en detalle**: *Coupled Cluster Singles and Doubles*. $T_1$ genera todas las formas de mover **1** electrón de un orbital ocupado a uno vacío, y $T_2$ todas las formas de mover **2**, cada una con un coeficiente (amplitud). La exponencial genera además combinaciones de esos movimientos.
- **Por qué falla al estirar**: cerca del equilibrio, el estado exacto es "HF + pequeñas correcciones" (*correlación dinámica*), y CCSD es excelente para eso. Al separar los átomos, cada electrón se queda con "su" hidrógeno y el estado pasa a ser una mezcla **pareja** de muchos bitstrings (*correlación estática*). "HF + correcciones chicas" deja de ser una buena descripción.
- **CCSD no es variacional**: puede dar una energía **por debajo** de la exacta, y en cadenas de H estiradas a veces lo hace. Por eso la consigna compara **valores absolutos** $|E - E_\text{FCI}|$. "Más bajo" no significa "mejor" para CCSD.
- Según las reglas, estos métodos **solo sirven para comparar y para elegir puntos de partida**: por ejemplo, usar las amplitudes de CCSD o MP2 como parámetros iniciales del ansatz UCCSD.

> Notebook: Química Cuántica, Parte 1 (secciones "Calculando energías con PySCF" y "La respuesta exacta y la energía de correlación").

---

## 8. VQE (Variational Quantum Eigensolver)

### El principio variacional
Para **cualquier** estado $\lvert\psi\rangle$ normalizado,
$$E(\psi) = \langle\psi\rvert\hat H\lvert\psi\rangle \;\ge\; E_0 .$$
**Por qué**: se escribe $\lvert\psi\rangle = \sum_i c_i \lvert E_i\rangle$ en la base de autovectores. Entonces $\langle\hat H\rangle = \sum_i |c_i|^2 E_i$, que es un **promedio ponderado** de autovalores (los pesos $|c_i|^2$ suman 1), y un promedio nunca es menor que el mínimo.
**Consecuencia**: la energía es una función de costo, y **cuanto más baja, mejor**. Nunca se puede "pasar" del valor real, salvo por ruido estadístico (y ahí entra $\sigma$).

### El algoritmo
```
θ = θ_inicial
repetir:
    preparar |ψ(θ)⟩ con el circuito (ansatz)
    E, σ = Estimator(ansatz(θ), H)       # mide Σ c_k ⟨P_k⟩ con shots
    θ = optimizador.paso(E)              # clásico: COBYLA, SPSA, L-BFGS-B...
hasta converger
```

### El ansatz: el "modelo" del estado
Un **ansatz** (en alemán, "planteo"; plural *ansätze*) es un circuito con parámetros, el equivalente a la arquitectura de una red neuronal. Es la decisión más importante del desafío.

| Ansatz | Idea | A favor | En contra |
|---|---|---|---|
| **Hardware-efficient** (Kandala 2017) | capas de rotaciones $R_y, R_z$ y CZ entre vecinos | pocas `cz`, se adapta al hardware | no respeta el número de electrones, mínimos locales, *barren plateaus* (gradientes que desaparecen) |
| **UCCSD** (*Unitary* CCSD) | $e^{T - T^\dagger}\lvert\text{HF}\rangle$: la versión unitaria de CCSD, así que es un circuito válido | químicamente motivado, respeta simetrías, se inicializa con amplitudes clásicas | **muchas** `cz` (cientos a miles para 8 qubits) |
| Otros (k-UpCCGSD, LUCJ, ADAPT-VQE, *givens rotations*) | términos de UCC elegidos o reducidos | compromiso entre precisión y profundidad | más trabajo de implementación |

- El circuito **arranca en el estado de HF** (aplicar `X` en los qubits ocupados) y el ansatz agrega correlación encima.
- **UCCSD no es CCSD**: UCCSD es unitario y variacional (nunca baja de $E_0$), y en la zona estirada suele andar bien justamente donde CCSD falla. Ese es el corazón de "ganarle a CCSD".
- **ADAPT-VQE** construye el ansatz de a un operador por vez, eligiendo el que más baja la energía. Suele dar muchas menos `cz` que UCCSD con la misma precisión.

### Ruido estadístico y la regla de medir de verdad
- La regla dice: las energías tienen que salir de **medir** con `EstimatorV2`/`SamplerV2` de `qiskit_aer`, **no** de leer el vector de estado.
- **Pueden optimizar $\theta$ como quieran**, incluso con statevector exacto. Pero la energía **final reportada** de cada punto tiene que salir de una medición con shots, con su $\sigma$.
- ⚠️ Revisen bien cómo genera `EstimatorV2` de Aer la $\sigma$: el parámetro `precision` controla la precisión objetivo, y con precisión 0 el resultado puede ser exacto y no venir de shots reales. Para que la $\sigma$ sea honesta, una opción es usar `SamplerV2` con shots explícitos y calcular ustedes los $\langle P_k\rangle$ y $\sigma$ (sección 2). Si hay dudas, pregunten al jurado.

> Notebook: Química Cuántica, Parte 2 (principio variacional, VQE, ansatz, optimizador, curva completa).

---

## 9. Costo: compuertas `cz` y shots

- **`cz`** (*controlled-Z*) es la compuerta de dos qubits del hardware de IBM: aplica $Z$ al segundo qubit si el primero está en 1. Las compuertas de dos qubits son las que **más errores** tienen (0,1 % por `cz` en `ruido.py`, contra 0,005 % para las de un qubit), así que **contar `cz` ≈ medir qué tan ruidoso va a ser el circuito**.
  - Cómo contarlas: transpilar el circuito a la base `['cz','sx','x','rz']` con conectividad en línea y leer `qc.count_ops()['cz']`.
  - **Conectividad en línea**: el qubit $i$ solo interactúa con $i\pm1$. Si el ansatz necesita acoplar qubits lejanos, el transpilador agrega **SWAPs** (3 `cz` cada uno). Ordenar bien los qubits ayuda.
  - Regla rápida: un circuito con $N$ `cz` sobrevive sin errores con probabilidad $\approx 0{,}999^N$. Con 1000 `cz` queda $\approx 37\%$: **basura**.
- **Shots**: cada uno es una ejecución. El jurado quiere ver cuántos se usaron por punto y cómo se repartieron (sección 2).

---

## 10. Ruido y mitigación (extras)

- **Modelo de ruido**: el simulador agrega errores aleatorios con las probabilidades de `ruido.py`. **`FakeFez`** es una copia simulada del ruido de un procesador real de IBM (Fez).
- **ZNE (Zero Noise Extrapolation)**: se corre el circuito con el ruido **amplificado** a propósito (por ejemplo, reemplazando cada `cz` por `cz cz cz`, que es lógicamente lo mismo pero tiene el triple de error), se mide $E$ con ruido ×1, ×3, ×5, y se **extrapola** la recta o curva a ruido 0. Es como un ajuste de regresión.
- **Mitigación de readout**: la medición final también se equivoca (lee 0 cuando era 1). Se calibra preparando estados conocidos, se arma la "matriz de confusión" y se invierte.
- **Post-selección por simetría**: si un bitstring medido tiene un número incorrecto de electrones, seguro es un error y se puede descartar.

> Notebook: Clase 2, Parte 2, ejercicio 14 (mitigación de errores).

---

## 11. "Argumenten por qué su resultado es confiable" (sin mirar FCI)

En una computadora cuántica real no se tiene FCI para comparar. Algunos argumentos que no lo necesitan:
1. **Cota variacional**: el ansatz es unitario, así que $E_\text{VQE} \ge E_0$ (salvo $\sigma$). Si se prueban mejoras y la energía no baja más, probablemente se llegó al mínimo.
2. **Varianza de la energía**: $\text{Var}(H) = \langle H^2\rangle - \langle H\rangle^2$ es **cero si y solo si el estado es un autovector**. Medirla y mostrar que es chica es una evidencia fuerte de que se encontró un autoestado.
3. **Simetrías**: medir el número de electrones $\langle N\rangle$ (tiene que dar 4) y el espín total $\langle S^2\rangle$ (tiene que dar 0 para el estado fundamental). Si dan mal, el estado está contaminado.
4. **Convergencia**: más capas o más operadores en el ansatz no cambian la energía, y varias semillas o puntos iniciales distintos llegan al mismo valor.
5. **Continuidad de la curva**: usar los $\theta$ óptimos de $d$ como punto inicial de $d + \Delta d$ (*warm start*) y verificar que la curva sale suave.

---

## 12. SQD (para el extra)

**Sample-based Quantum Diagonalization**: se usa la computadora cuántica solo para **muestrear bitstrings**, se arma la matriz $\hat H$ restringida a ese subconjunto y se diagonaliza clásicamente. Con H₄ hay solo 36 bitstrings válidos, así que casi cualquier muestreo los encuentra todos y da la respuesta exacta. Por eso **no vale para el desafío central**.

---

## 13. Glosario de símbolos

| Símbolo | Se lee | Significa |
|---|---|---|
| $\hat H$ | "hache sombrero" | hamiltoniano (operador de energía, una matriz) |
| $A^\dagger$ | "A daga" | transpuesta conjugada: `A.conj().T` |
| $a_p^\dagger,\ a_p$ | creación y aniquilación | poner y sacar un electrón en el espín-orbital $p$ |
| $\lvert\psi\rangle$, $\langle\psi\rvert$ | ket, bra | vector columna y vector fila conjugado |
| $\langle\psi\rvert\hat H\lvert\psi\rangle$ | valor esperado | energía promedio del estado $\psi$ |
| $\otimes$ | producto tensorial | `np.kron`, juntar sistemas |
| $\{I,X,Y,Z\}^{\otimes n}$ | — | cadenas de Pauli de $n$ qubits |
| $c_k$, $P_k$ | — | coeficiente y cadena de Pauli del término $k$ |
| $\theta$ | theta | parámetros del circuito (ángulos de rotación) |
| $E_0$ | — | energía del estado fundamental (el autovalor más chico) |
| $\sigma$ | sigma | error estadístico (desviación estándar de la estimación) |
| Å | ångström | $10^{-10}$ m |
| Ha, mHa | Hartree, miliHartree | unidad de energía; 1,6 mHa = precisión química |
| $R_e$, $D_e$ | — | distancia de equilibrio y energía de disociación |
| $d$ | — | distancia entre hidrógenos vecinos |
| STO-3G | — | base mínima de orbitales |
| HF, CCSD, FCI | — | métodos clásicos: aproximado, mejor aproximado, exacto |
| VQE, ansatz, UCCSD | — | algoritmo cuántico, circuito parametrizado, ansatz químico |
| ZNE | — | extrapolación a ruido cero |
| `cz` | controlled-Z | compuerta de 2 qubits, la cara y ruidosa |

---

## 14. Checklist de la consigna, traducido

- [ ] Para cada $d$ en al menos 10 valores entre 0,6 y 3,0 Å (con más densidad cerca del mínimo, ~0,9 Å, y en la zona ≥ 1,8 Å):
  - [ ] armar H₄ lineal en STO-3G con PySCF → $E_\text{HF}$, $E_\text{CCSD}$, $E_\text{FCI}$
  - [ ] armar $\hat H$ como `SparsePauliOp` (Jordan-Wigner), **sumando la repulsión nuclear**
  - [ ] optimizar el ansatz (puede ser sin ruido, con *warm start* desde el punto anterior)
  - [ ] medir la energía final con Estimator o Sampler de Aer → $E \pm \sigma$, anotando shots y cantidad de `cz` transpiladas
- [ ] Graficar $E(d)$ de VQE con barras de error, CCSD y FCI, y aparte el **error vs FCI en mHa** con la línea de 1,6 mHa
- [ ] $R_e$ (ajuste alrededor del mínimo) y $D_e = E(3{,}0) - E(R_e)$
- [ ] Verificar $|E_\text{VQE}-E_\text{FCI}| + \sigma < |E_\text{CCSD}-E_\text{FCI}|$ para $d \ge 1{,}8$
- [ ] Argumento de confiabilidad (sección 11)
- [ ] Tabla: $d$, $E$, $\sigma$, shots, `cz`
- [ ] Texto: qué decidimos, por qué, qué no funcionó
- [ ] Notebook de Colab que corra de punta a punta, en una carpeta de Drive compartida con el jurado

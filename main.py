# main.py
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import math


# =========================
# Parámetros globales
# =========================
DAY_START = 0               # 06:00 (minuto 0)
DAY_END = 18 * 60           # 18 horas = 1080 min
MIN_SEPARACION = 4.0        # separación mínima en minutos
SEPARACION_OBJETIVO = 5.0   # separación deseada
RADAR_DIST = 100.0          # distancia inicial en millas náuticas
VEL_RETROCESO = 200.0       # velocidad en turnaround (nudos)
RNG_SEED = 42               # semilla reproducible

TRAMOS = [
    (100.0, float("inf"), 300.0, 500.0),
    (50.0, 100.0, 250.0, 300.0),
    (15.0, 50.0, 200.0, 250.0),
    (5.0, 15.0, 150.0, 200.0),
    (0.0, 5.0, 120.0, 150.0),
]

# =========================
# Utilidades
# =========================
def kts_to_nm_min(kts: float) -> float:
    """Convierte de nudos a millas náuticas por minuto."""
    return kts / 60.0

def rango_velocidad(dist_nm: float) -> Tuple[float, float]:
    """Devuelve el rango (vmin, vmax) válido para una distancia dada."""
    for low, high, vmin, vmax in TRAMOS:
        if low < dist_nm <= high:   # 👈 cambiamos los signos
            return vmin, vmax
    return TRAMOS[0][2], TRAMOS[0][3]


def tiempo_estimado(dist_nm: float, vel_kts: float) -> float:
    """
    Calcula el tiempo hasta AEP si se vuela a velocidad constante vel_kts,
    ajustando tramo por tramo.
    """
    tiempo, dist, v = 0.0, dist_nm, vel_kts
    while dist > 0:
        for low, high, vmin, vmax in TRAMOS:
            if low <= dist < high:
                restante = min(dist, dist - low)
                if restante <= 0:
                    dist -= 1e-6
                    continue
                tiempo += restante / kts_to_nm_min(v)
                dist -= restante
                v = vmin
                break
    return tiempo

# =========================
# Clase Avión
# =========================
@dataclass
class Avion:
    id: int
    minuto_inicio: int
    distancia: float = RADAR_DIST
    velocidad: float = 300.0
    estado: str = "approach"   # approach | turnaround | diverted | landed
    congestionado: bool = False   # si alguna vez estuvo en congestión
    min_congestion: int = 0       # minutos acumulados en congestión
    lider: Optional[int] = None
    t_aterrizaje: Optional[int] = None
    t_ideal: Optional[float] = None      # tiempo de aterrizaje ideal (siempre a vmax)
    atraso: Optional[float] = None       # diferencia entre real e ideal
    registro_t: List[int] = field(default_factory=list)
    registro_d: List[float] = field(default_factory=list)
    registro_v: List[float] = field(default_factory=list)
    registro_estado: List[str] = field(default_factory=list)


    def limites_vel(self) -> Tuple[float, float]:
        return rango_velocidad(self.distancia)

    def tiempo_a_aep(self, vel: Optional[float] = None) -> float:
        v = self.velocidad if vel is None else vel
        return tiempo_estimado(self.distancia, v)
    
    def tiempo_a_aep_libre(self) -> float:
        """Tiempo hasta AEP volando siempre al vmax de cada tramo."""
        tiempo, dist = 0.0, self.distancia
        for low, high, vmin, vmax in TRAMOS:
            if dist > low:
                # nm dentro de este tramo
                restante = min(dist, high) - low
                tiempo += restante / kts_to_nm_min(vmax)
                dist = low
        return tiempo






    def avanzar(self, minuto: int, paso: int = 1):
        """Avanza `paso` minutos según la velocidad actual y guarda historial."""
        # cuánto avanza en este paso
        delta = kts_to_nm_min(self.velocidad) * paso
        self.distancia = max(0.0, self.distancia - delta)

        # Si llegó a tierra → aterriza en el minuto entero
        if self.distancia == 0.0 and self.estado != "landed":
            self.estado = "landed"
            self.t_aterrizaje = minuto  # 🚨 ahora se usa el minuto entero, sin decimales

        # Guardar en historial
        self.registro_t.append(minuto)
        self.registro_d.append(self.distancia)
        self.registro_v.append(self.velocidad)
        self.registro_estado.append(self.estado)








            

# =========================
# Clase Simulador
# =========================
class Simulador:
    def __init__(self, seed: int = RNG_SEED):
        self.rng = np.random.default_rng(seed)
        self.aviones: Dict[int, Avion] = {}
        self.tiempos_aterrizaje: List[float] = []

    def arribos_bernoulli(self, lam: float, t_ini: int = DAY_START, t_fin: int = DAY_END) -> List[int]:
        """
        Genera los minutos en que aparecen aviones,
        siguiendo un proceso Bernoulli minuto a minuto.
        """
        tiempos = []
        for minuto in range(t_ini, t_fin):
            if self.rng.random() < lam:
                tiempos.append(minuto)
        return tiempos

    def cargar_aviones(self, arribos: List[int]):
        """
        Crea instancias de Avion a partir de los minutos de arribo,
        y calcula el ETA ideal suponiendo velocidad máxima.
        """
        for idx, minuto in enumerate(arribos, start=1):
            avion = Avion(id=idx, minuto_inicio=minuto)

            # calcular ETA ideal desde el inicio
            _, vmax = avion.limites_vel()
            avion.t_ideal = math.ceil(minuto + avion.tiempo_a_aep_libre())


            self.aviones[idx] = avion


    def step(self, minuto: int):
        # seleccionar aviones activos en aproximación o turnaround
        activos = [a for a in self.aviones.values()
                if a.estado in ("approach", "turnaround") and a.minuto_inicio <= minuto]

        # ordenar por distancia (más cerca de AEP primero)
        activos = sorted(activos, key=lambda av: av.distancia)

        for i, avion in enumerate(activos):
            if minuto == avion.minuto_inicio:
                print(f"[t={minuto}] Avión {avion.id} aparece en radar (100 nm).")
                vmin, vmax = avion.limites_vel()
                avion.velocidad = vmax
                continue

            if avion.estado == "approach":
                if i == 0:
                    # 🚨 líder siempre al vmax de su tramo
                    vmin, vmax = avion.limites_vel()
                    avion.velocidad = vmax
                    avion.congestionado = False
                else:
                    lider = activos[i-1]
                    eta_self = minuto + avion.tiempo_a_aep(avion.velocidad)
                    eta_lider = minuto + lider.tiempo_a_aep(lider.velocidad)
                    diff = eta_self - eta_lider

                    if diff < MIN_SEPARACION:
                        nueva_vel = lider.velocidad - 20
                        vmin, vmax = avion.limites_vel()
                        if nueva_vel >= vmin:
                            avion.velocidad = nueva_vel
                            avion.congestionado = True
                            avion.min_congestion += 1
                            print(f"[t={minuto}] Avión {avion.id} reduce velocidad a {avion.velocidad} kts (congestión).")
                        else:
                            avion.estado = "turnaround"
                            avion.velocidad = VEL_RETROCESO
                            print(f"[t={minuto}] Avión {avion.id} entra en TURNAROUND 🚨")
                    elif diff >= SEPARACION_OBJETIVO:
                        vmin, vmax = avion.limites_vel()
                        avion.velocidad = vmax

            elif avion.estado == "turnaround":
                if avion.distancia >= RADAR_DIST:
                    avion.estado = "diverted"
                    print(f"[t={minuto}] Avión {avion.id} se fue a MONTEVIDEO ✈️")
                    continue
                etas = sorted([a.t_aterrizaje for a in self.aviones.values() if a.t_aterrizaje])
                for t1, t2 in zip(etas, etas[1:]):
                    if t2 - t1 >= 10 and avion.distancia > 5:
                        avion.estado = "approach"
                        _, vmax = avion.limites_vel()
                        avion.velocidad = vmax
                        print(f"[t={minuto}] Avión {avion.id} REJOIN entre {t1:.1f} y {t2:.1f}.")
                        break

            # avanzar
            avion.avanzar(minuto, paso=1)

            if avion.distancia <= 0.0 and avion.estado != "landed":
                avion.estado = "landed"
                avion.t_aterrizaje = minuto
                self.tiempos_aterrizaje.append(minuto)
                print(f"[t={minuto}] Avión {avion.id} aterrizó en AEP ✅")








    def simular_dia(self, lam: float):
        """Corre la simulación de un día completo con tasa de arribo lam."""
        arribos = self.arribos_bernoulli(lam, 0, DAY_END)
        self.cargar_aviones(arribos)
        self.tiempos_aterrizaje.clear()

        for minuto in range(DAY_END):
            self.step(minuto)
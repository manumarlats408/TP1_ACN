import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from dataclasses import dataclass, field

# Parámetros globales
RADAR_DIST = 100.0
MIN_SEPARACION = 4.0  # minutos
VEL_RETROCESO = 200.0  # knots (velocidad de regreso / desvío)
TIEMPO_IDEAL = 23.4  # minutos de aproximación ideal (para cálculo de retraso)

# Funciones auxiliares
def vel_maxima_permitida_por_tramo(dist_nm: float) -> int:
    if dist_nm > 50:
        return 300
    elif dist_nm > 15:
        return 250
    elif dist_nm > 5:
        return 200
    else:
        return 150

def velocidad_minima_permitida_por_tramo(distancia):
    if distancia > 50:
        return 250
    elif distancia > 15:
        return 200
    elif distancia > 5:
        return 150
    else:
        return 120

def tiempo_entre_aviones(avion_trasero, avion_delantero):
    delta_dist = avion_trasero.distancia - avion_delantero.distancia
    if avion_trasero.velocidad <= 0:
        return float("inf")
    return (delta_dist / avion_trasero.velocidad) * 60.0

def aterrizaje_libre(dist_inicial, minutos) -> float:
    v_max = vel_maxima_permitida_por_tramo(dist_inicial)
    return (v_max / 60.0) * minutos

def hay_gap_disponible(avion, cola, minutos_gap=10):
    # Verifica si el avión puede volver con al menos 'minutos_gap' de separación.
    pos = avion.distancia
    avance_estimado = aterrizaje_libre(pos, minutos=minutos_gap/2)
    for otro in cola:
        if otro.estado in ("APROXIMANDO", "AJUSTANDO"):
            if pos - avance_estimado <= otro.distancia <= pos + avance_estimado:
                return False
    return True

# Clase Avión
class Avion:
    def __init__(self, id_avion, minuto_actual, will_interrupt=False):
        self.id = id_avion
        self.distancia = 100.0
        self.velocidad = 0.0
        self.estado = "APROXIMANDO"
        # Estados: APROXIMANDO | AJUSTANDO | REGRESANDO | REGRESANDO_VIENTO | DESVIADO | ATERRIZADO
        self.tiempo_llegada = minuto_actual
        self.retraso = 0.0
        self.t_aterrizaje = None
        self.will_interrupt = will_interrupt
        self.interrupted = False
        self.diverted_to_mvd = False
        self.congestionado = False

    def __repr__(self):
        return (f"Avión {self.id} - Dist {self.distancia:.1f} nm - Vel {self.velocidad:.0f} kts - Estado {self.estado}")


    def controlar_aproximacion(self, lider=None, minuto_actual=None):
        if lider is None:
            self.estado = "APROXIMANDO"
            self.velocidad = vel_maxima_permitida_por_tramo(self.distancia)
            return

        separacion = tiempo_entre_aviones(self, lider)
        if separacion < MIN_SEPARACION:
            nueva_vel = lider.velocidad - 20
            if nueva_vel < velocidad_minima_permitida_por_tramo(self.distancia):
                self.estado = "REGRESANDO"
                self.velocidad = VEL_RETROCESO
            else:
                self.estado = "AJUSTANDO"
                self.velocidad = nueva_vel
                self.congestionado = True
        else:
            self.estado = "APROXIMANDO"
            self.velocidad = vel_maxima_permitida_por_tramo(self.distancia)




# ======================
# Clase Simulador
# ======================
class Simulador:
    def __init__(self, seed=50):
        self.rng = np.random.default_rng(seed)
        self.aviones: dict[int, Avion] = {}
        self.historial = []
        self.finalizados: List[Avion] = []

    def generar_nuevo_avion(self, minuto: int, next_id: int, lam: float) -> int:
        if self.rng.random() < lam:
            avion = Avion(next_id, minuto)
            self.aviones[next_id] = avion
            return next_id + 1
        return next_id

    def actualizar_estados(self, minuto: int):
        activos = [a for a in self.aviones.values() if a.estado not in ("ATERRIZADO", "DESVIADO")]
        activos.sort(key=lambda av: av.distancia)

        for i, avion in enumerate(activos):
            if avion.estado == "REGRESANDO":
                self.controlar_regreso(avion, activos)
            else:
                lider = activos[i-1] if i > 0 and activos[i-1].estado != "REGRESANDO" else None
                avion.controlar_aproximacion(lider)

    def controlar_regreso(self, avion, cola):
        if avion.distancia > RADAR_DIST:
            avion.estado = "DESVIADO"
            self.finalizados.append(avion)
        elif hay_gap_disponible(avion, cola):
            avion.estado = "APROXIMANDO"

    def mover_aviones(self):
        for avion in self.aviones.values():
            if avion.estado in ("ATERRIZADO", "DESVIADO"):
                continue
            delta = avion.velocidad / 60.0
            if avion.estado == "REGRESANDO":
                avion.distancia += delta
            else:
                avion.distancia -= delta

    def gestionar_finalizados(self, minuto: int, tiempo_ideal=23.4):
        for avion in self.aviones.values():
            if avion.estado in ("ATERRIZADO", "DESVIADO"):
                continue
            if avion.distancia <= 0:
                avion.estado = "ATERRIZADO"
                avion.t_aterrizaje = minuto
                tiempo_real = minuto - avion.tiempo_llegada
                avion.retraso = max(0, tiempo_real - tiempo_ideal)
                self.finalizados.append(avion)

    def guardar_estado(self, minuto: int):
        for avion in self.aviones.values():
            self.historial.append({
                "minuto": minuto,
                "id": avion.id,
                "distancia": avion.distancia,
                "velocidad": avion.velocidad,
                "estado": avion.estado,
            })

    def simular_dia(self, lam: float, minutos=18*60):
        next_id = 1
        for minuto in range(minutos):
            next_id = self.generar_nuevo_avion(minuto, next_id, lam)
            self.actualizar_estados(minuto)
            self.mover_aviones()
            self.gestionar_finalizados(minuto)
            self.guardar_estado(minuto)



# ======================
# Clase Simulador con Viento
# ======================
class SimuladorViento(Simulador):
    def __init__(self, seed=50, p_interrupt=0.1):
        self.rng = np.random.default_rng(seed)
        self.aviones: Dict[int, Avion] = {}
        self.historial = []
        self.finalizados: List[Avion] = []
        self.p_interrupt = p_interrupt

    def generar_nuevo_avion(self, minuto: int, next_id: int, lam: float) -> int:
        if self.rng.random() < lam:
            will_interrupt = self.rng.random() < self.p_interrupt
            avion = Avion(next_id, minuto, will_interrupt=will_interrupt)
            avion.velocidad = vel_maxima_permitida_por_tramo(avion.distancia)
            self.aviones[next_id] = avion
            return next_id + 1
        return next_id

    def actualizar_estados(self, minuto: int):
        activos = [a for a in self.aviones.values() if a.estado not in ("ATERRIZADO", "DESVIADO")]
        activos.sort(key=lambda av: av.distancia)

        for i, avion in enumerate(activos):
            # Interrupción por viento: si está predispuesto y alcanza <=5 nm pasa a REGRESANDO_VIENTO
            if (avion.will_interrupt and not avion.interrupted and avion.distancia <= 5.0 and
                avion.estado in ("APROXIMANDO", "AJUSTANDO")):
                avion.interrupted = True
                avion.estado = "REGRESANDO_VIENTO"
                avion.velocidad = VEL_RETROCESO
                continue

            if avion.estado in ("REGRESANDO", "REGRESANDO_VIENTO"):
                self.controlar_regreso(avion, activos)
            else:
                lider = activos[i-1] if i > 0 and activos[i-1].estado not in ("REGRESANDO", "REGRESANDO_VIENTO") else None
                avion.controlar_aproximacion(lider, minuto_actual=minuto)

    def controlar_regreso(self, avion, cola):
        if avion.distancia > RADAR_DIST:
            avion.estado = "DESVIADO"
            avion.diverted_to_mvd = True
            self.finalizados.append(avion)
            return
        if avion.estado == "REGRESANDO_VIENTO":
            if hay_gap_disponible(avion, cola, minutos_gap=10):
                avion.estado = "APROXIMANDO"
                avion.velocidad = velocidad_minima_permitida_por_tramo(avion.distancia)
                return
            else:
                avion.velocidad = VEL_RETROCESO
                return

        if hay_gap_disponible(avion, cola):
            avion.estado = "APROXIMANDO"
            avion.velocidad = vel_maxima_permitida_por_tramo(avion.distancia)
        else:
            avion.velocidad = VEL_RETROCESO

    def mover_aviones(self, delta_min=1.0):
        for avion in self.aviones.values():
            if avion.estado in ("ATERRIZADO", "DESVIADO"):
                continue
            delta = avion.velocidad / 60.0 * delta_min
            if avion.estado in ("REGRESANDO", "REGRESANDO_VIENTO"):
                avion.distancia += delta
            else:
                avion.distancia -= delta

    def gestionar_finalizados(self, minuto: int, tiempo_ideal=23.4):
        for avion in list(self.aviones.values()):
            if avion.estado in ("ATERRIZADO", "DESVIADO"):
                continue
            if avion.distancia <= 0.0:
                avion.estado = "ATERRIZADO"
                avion.t_aterrizaje = minuto
                tiempo_real = minuto - avion.tiempo_llegada
                avion.retraso = max(0.0, tiempo_real - tiempo_ideal)
                self.finalizados.append(avion)

    def guardar_estado(self, minuto: int):
        for avion in self.aviones.values():
            self.historial.append({
                "minuto": minuto,
                "id": avion.id,
                "distancia": avion.distancia,
                "velocidad": avion.velocidad,
                "estado": avion.estado,
                "will_interrupt": getattr(avion, "will_interrupt", False),
                "interrupted": getattr(avion, "interrupted", False),
                "diverted_to_mvd": getattr(avion, "diverted_to_mvd", False),
            })






# ======================
# Clase Simulador con Tormenta
# ======================

class SimuladorTormenta(Simulador):
    def __init__(self, seed=50, cierre_inicio=None, cierre_duracion=30):
        self.rng = np.random.default_rng(seed)
        self.aviones: Dict[int, Avion] = {}
        self.historial = []
        self.finalizados: List[Avion] = []
        self.cierre_inicio = cierre_inicio
        self.cierre_duracion = cierre_duracion

    def generar_nuevo_avion(self, minuto: int, next_id: int, lam: float) -> int:
        if self.rng.random() < lam:
            avion = Avion(next_id, minuto)
            self.aviones[next_id] = avion
            return next_id + 1
        return next_id

    def actualizar_estados(self, minuto: int):
        activos = [a for a in self.aviones.values() if a.estado not in ("ATERRIZADO", "DESVIADO")]
        activos.sort(key=lambda av: av.distancia)

        for i, avion in enumerate(activos):
            if avion.estado == "REGRESANDO":
                self.controlar_regreso(avion, activos)
            else:
                lider = activos[i-1] if i > 0 and activos[i-1].estado != "REGRESANDO" else None
                avion.controlar_aproximacion(lider)

    def controlar_regreso(self, avion, cola):
        if avion.distancia > RADAR_DIST:
            avion.estado = "DESVIADO"
            avion.diverted_to_mvd = True
            self.finalizados.append(avion)
        elif hay_gap_disponible(avion, cola):
            avion.estado = "APROXIMANDO"

    def mover_aviones(self):
        for avion in self.aviones.values():
            if avion.estado in ("ATERRIZADO", "DESVIADO"):
                continue
            delta = avion.velocidad / 60.0
            if avion.estado == "REGRESANDO":
                avion.distancia += delta
            else:
                avion.distancia -= delta

    def gestionar_finalizados(self, minuto: int, tiempo_ideal=TIEMPO_IDEAL):
        en_cierre = self.cierre_inicio is not None and self.cierre_inicio <= minuto < self.cierre_inicio + self.cierre_duracion
        for avion in self.aviones.values():
            if avion.estado in ("ATERRIZADO", "DESVIADO"):
                continue

            if avion.distancia <= 0:
                if en_cierre:
                    avion.estado = "DESVIADO"
                    avion.diverted_to_mvd = True
                    self.finalizados.append(avion)
                else:
                    avion.estado = "ATERRIZADO"
                    avion.t_aterrizaje = minuto
                    tiempo_real = minuto - avion.tiempo_llegada
                    avion.retraso = max(0, tiempo_real - tiempo_ideal)
                    self.finalizados.append(avion)

    def guardar_estado(self, minuto: int):
        for avion in self.aviones.values():
            self.historial.append({
                "minuto": minuto,
                "id": avion.id,
                "distancia": avion.distancia,
                "velocidad": avion.velocidad,
                "estado": avion.estado,
                "congestionado": avion.congestionado,
                "diverted_to_mvd": avion.diverted_to_mvd,
            })

    def simular_dia(self, lam: float, minutos=18*60):
        next_id = 1
        for minuto in range(minutos):
            next_id = self.generar_nuevo_avion(minuto, next_id, lam)
            self.actualizar_estados(minuto)
            self.mover_aviones()
            self.gestionar_finalizados(minuto)
            self.guardar_estado(minuto)

# ======================
# Simulador sin congestion
# ======================

class SimuladorSinCongestion(Simulador):
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    def simular_dia(self, lam: float, minutos=18*60):
        next_id = 1
        for minuto in range(minutos):
            # Generar nuevos arribos
            next_id = self.generar_nuevo_avion(minuto, next_id, lam)

            # Mover todos los aviones sin lógica de congestión
            for avion in self.aviones.values():
                if avion.estado in ("ATERRIZADO", "DESVIADO"):
                    continue
                # velocidad máxima permitida según tramo
                avion.velocidad = vel_maxima_permitida_por_tramo(avion.distancia)
                avion.distancia -= avion.velocidad / 60.0

                # Verificar aterrizaje
                if avion.distancia <= 0:
                    avion.estado = "ATERRIZADO"
                    avion.t_aterrizaje = minuto
                    # Sin congestión → retraso = 0
                    avion.retraso = 0
                    self.finalizados.append(avion)

            # Guardar estado
            self.guardar_estado(minuto)


# ======================
# Clase Simulador con reintentos (para ej7parte1)
# ======================
class Simulador_con_reintentos:
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.aviones: dict[int, Avion_con_reintentos] = {}
        self.historial = []
        self.finalizados: List[Avion_con_reintentos] = []
        self.no_aterriza : List[int] = []
        self.intentos_permitidos: int = 1 #modificar para experimentar con más intentos. No es relevante ya que aumentaría mucho el tiempo de atraso

    def generar_nuevo_avion(self, minuto: int, next_id: int, lam: float) -> int:
        if self.rng.random() < lam:
            avion = Avion_con_reintentos(next_id, minuto)
            self.aviones[next_id] = avion
            return next_id + 1
        return next_id

    def actualizar_estados(self, minuto: int):
        activos = [a for a in self.aviones.values() if a.estado not in ("ATERRIZADO", "DESVIADO")]
        activos.sort(key=lambda av: av.distancia)

        for i, avion in enumerate(activos):
            if avion.estado == "REGRESANDO":
                self.controlar_regreso(avion, activos)
            else:
                lider = activos[i-1] if i > 0 and activos[i-1].estado != "REGRESANDO" else None
                avion.controlar_aproximacion(lider)

    def controlar_regreso(self, avion, cola): #MODIFICACIÓN PARA EJERCICIO 7
        if avion.distancia > RADAR_DIST and avion.intentos < self.intentos_permitidos:
            #vuelve a intentar 1 sola vez. 
            avion.estado = "ACERCANDOSE"
            avion.intentos += 1
        elif avion.distancia > RADAR_DIST:
            #si ya probó las veces permitidas. Se va Montevideo.
            avion.estado = "DESVIADO"
        elif hay_gap_disponible(avion, cola):
            avion.estado = "APROXIMANDO"

    def mover_aviones(self):
        for avion in self.aviones.values():
            if avion.estado in ("ATERRIZADO", "DESVIADO"):
                continue
            delta = avion.velocidad / 60.0
            if avion.estado == "REGRESANDO":
                avion.distancia += delta
            else:
                avion.distancia -= delta

    def gestionar_finalizados(self, minuto: int, tiempo_ideal=23.4):
        for avion in self.aviones.values():
            if avion.estado in ("ATERRIZADO", "DESVIADO"):
                continue
            if avion.distancia <= 0:
                avion.estado = "ATERRIZADO"
                avion.t_aterrizaje = minuto
                tiempo_real = minuto - avion.tiempo_llegada
                avion.retraso = max(0, tiempo_real - tiempo_ideal)
                self.finalizados.append(avion)

    def guardar_estado(self, minuto: int):
        for avion in self.aviones.values():
            self.historial.append({
                "minuto": minuto,
                "id": avion.id,
                "distancia": avion.distancia,
                "velocidad": avion.velocidad,
                "estado": avion.estado,
            })

    def simular_dia(self, lam: float, minutos=18*60):
        next_id = 1
        for minuto in range(minutos):
            next_id = self.generar_nuevo_avion(minuto, next_id, lam)
            self.actualizar_estados(minuto)
            self.mover_aviones()
            self.gestionar_finalizados(minuto)
            self.guardar_estado(minuto)

# ======================
# Clase Avión con reintentos para ej7parte1
# ======================
class Avion_con_reintentos:
    def __init__(self, id_avion, minuto_actual):
        self.id = id_avion
        self.distancia = 100.0
        self.velocidad = 0.0
        self.estado = "APROXIMANDO"
        # Estados posibles: APROXIMANDO | AJUSTANDO | REGRESANDO | DESVIADO | ATERRIZADO 
        self.tiempo_llegada = minuto_actual
        self.retraso = 0
        self.t_aterrizaje = None
        self.intentos = 0 #Se agrega para ej7

    def __repr__(self):
        return (f" Avión {self.id}\n"
                f" - Distancia: {self.distancia:.1f} mn\n"
                f" - Velocidad: {self.velocidad:.0f} kts\n"
                f" - Estado: {self.estado}")

    def controlar_aproximacion(self, lider=None):
        if lider is None:
            self.estado = "APROXIMANDO"
            self.velocidad = vel_maxima_permitida_por_tramo(self.distancia)
            return

        separacion = tiempo_entre_aviones(self, lider)
        if separacion < MIN_SEPARACION:
            nueva_vel = lider.velocidad - 20
            if nueva_vel < velocidad_minima_permitida_por_tramo(self.distancia):
                self.estado = "REGRESANDO" 
                self.velocidad = VEL_RETROCESO
    
            else:
                self.estado = "AJUSTANDO" 
                self.velocidad = nueva_vel
        else:
            self.estado = "APROXIMANDO"
            self.velocidad = vel_maxima_permitida_por_tramo(self.distancia)


# ======================
# Video de una simulación: reconstruir las trayectorias de cada avión a partir de simulador.historial
# ======================
import itertools
def animate_simulation(simulador: Simulador, titulo_video: str):
    """
    Genera un video con las trayectorias de los aviones en la simulación.
    - Eje X: tiempo [min]
    - Eje Y: distancia a la pista [mn]
    - Cada avión = un punto con color propio.
    """

    # reconstruimos trayectorias por avión
    aviones = {}
    for registro in simulador.historial:
        aid = registro["id"]
        if aid not in aviones:
            aviones[aid] = {"t": [], "d": []}
        aviones[aid]["t"].append(registro["minuto"])
        aviones[aid]["d"].append(registro["distancia"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel("Tiempo [min]")
    ax.set_ylabel("Distancia a pista [mn]")
    ax.set_ylim(0, 100)   # 0 = pista, 100 = radar
    ax.grid(True, linestyle="--", alpha=0.6)

    tiempo_max = max(r["minuto"] for r in simulador.historial)
    ax.set_xlim(0, tiempo_max)

    # paleta de colores (cíclica si hay muchos aviones)
    colors = itertools.cycle(plt.cm.tab20.colors)

    scatters = {}
    for aid in aviones.keys():
        scatters[aid] = ax.plot([], [], "o", ms=6, color=next(colors))[0]

    def update(frame):
        for aid, datos in aviones.items():
            t_arr = np.array(datos["t"])
            d_arr = np.array(datos["d"])

            if frame >= t_arr[0] and frame <= t_arr[-1]:
                y = np.interp(frame, t_arr, d_arr)
                scatters[aid].set_data([frame], [y])
            else:
                scatters[aid].set_data([], [])
        return list(scatters.values())

    anim = FuncAnimation(
        fig, update,
        frames=np.arange(0, tiempo_max+1, 1),
        interval=100,
        blit=False
    )

    writer = FFMpegWriter(fps=20, metadata=dict(artist='Simulación AEP'))
    anim.save(f"{titulo_video}.mp4", writer=writer)
    plt.close(fig)
    return

# ======================
# Segundo video
# ======================
def animate_simulation_aviones(simulador: Simulador, titulo_video: str, Y_MAX=200):
    """
    Visualización tipo planeo ideal:
    - X = distancia real a la pista [mn]
    - Y = altura ficticia: depende solo de la distancia (100 -> Y_MAX, 0 -> 0)
    - Si el avión retrocede en X, su altura sube de nuevo.
    """

    aviones = {}
    for registro in simulador.historial:
        aid = registro["id"]
        if aid not in aviones:
            aviones[aid] = {"t": [], "d": [], "estado": []}
        aviones[aid]["t"].append(registro["minuto"])
        aviones[aid]["d"].append(registro["distancia"])
        aviones[aid]["estado"].append(registro["estado"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel("Distancia a pista [mn]")
    ax.set_xlim(100, 0)          # radar a pista
    ax.set_ylim(0, Y_MAX)        # altura ficticia

    ax.set_yticks([])
    ax.axhline(Y_MAX, color="black", linewidth=2)
    ax.text(101, Y_MAX, "Radar", va="bottom", ha="left", fontsize=10, color="black")

    ax.axhspan(0, Y_MAX, facecolor="skyblue", alpha=0.2)   # todo celeste
   
    tiempo_max = max(r["minuto"] for r in simulador.historial)
    colors = itertools.cycle(plt.cm.tab20.colors)
    scatters = {}

    for aid in aviones:
        color = next(colors)
        scatters[aid] = ax.plot([], [], marker="$✈$", ms=12,
                                color=color, linestyle="")[0]

    # --- Contadores ---
    aterrizados = set()
    desviados = set()

    # Usamos fig.text (coordenadas relativas a la figura completa)
    txt_aterrizados = fig.text(0.95, 0.035, "Aterrizados: 0",
                               ha="right", va="bottom",
                               fontsize=12, color="green", weight="bold")
    txt_desviados = fig.text(0.05, 0.95, "Desviados: 0",
                             ha="left", va="top",
                             fontsize=12, color="red", weight="bold")
    txt_tiempo = fig.text(0.5, 0.97, "0 min",
                          ha="center", va="top",
                          fontsize=13, color="blue", weight="bold")

    def update(frame):
        # --- actualizar posiciones de aviones ---
        for aid, datos in aviones.items():
            t_arr = np.array(datos["t"])
            d_arr = np.array(datos["d"])
            estado_arr = np.array(datos["estado"])

            if frame >= t_arr[0] and frame <= t_arr[-1]:
                x = np.interp(frame, t_arr, d_arr)
                y = (x / 100) * Y_MAX
                scatters[aid].set_data([x], [y])

                # actualizar estado
                idx = np.searchsorted(t_arr, frame, side="right") - 1
                estado_actual = estado_arr[idx]
                if estado_actual == "ATERRIZADO":
                    aterrizados.add(aid)
                elif estado_actual == "DESVIADO":
                    desviados.add(aid)
            else:
                scatters[aid].set_data([], [])

        # --- actualizar contadores ---
        txt_aterrizados.set_text(f"Aterrizados: {len(aterrizados)}")
        txt_desviados.set_text(f"Desviados: {len(desviados)}")

        horas, minutos = divmod(frame, 60)
        if horas > 0:
            txt_tiempo.set_text(f"{horas}h {minutos}min")
        else:
            txt_tiempo.set_text(f"{minutos}min")

        return list(scatters.values()) + [txt_aterrizados, txt_desviados, txt_tiempo]
    
    anim = FuncAnimation(
        fig, update,
        frames=np.arange(0, tiempo_max+1, 1),
        interval=200,
        blit=False
    )

    writer = FFMpegWriter(fps=15, metadata=dict(artist='Simulación AEP'))
    anim.save(f"{titulo_video}.mp4", writer=writer)
    plt.close(fig)
    return

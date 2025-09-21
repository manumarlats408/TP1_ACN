import numpy as np
from typing import List

# ======================
# Parámetros globales
# ======================
RADAR_DIST = 100
MIN_SEPARACION = 4
VEL_RETROCESO = 200.0


# ======================
# Funciones auxiliares
# ======================
def vel_maxima_permitida_por_tramo(dist_nm: float) -> int:
    """Velocidad máxima permitida según distancia a AEP."""
    if dist_nm > 50:
        return 300
    elif dist_nm > 15:
        return 250
    elif dist_nm > 5:
        return 200
    else:
        return 150


def velocidad_minima_permitida_por_tramo(distancia):
    """Velocidad mínima permitida según distancia a AEP."""
    if distancia > 50:
        return 250
    elif distancia > 15:
        return 200
    elif distancia > 5:
        return 150
    else:
        return 120


def tiempo_entre_aviones(avion_trasero, avion_delantero):
    """Devuelve la separación temporal (min) entre dos aviones."""
    delta_dist = avion_trasero.distancia - avion_delantero.distancia
    if avion_trasero.velocidad <= 0:
        return float("inf")
    return (delta_dist / avion_trasero.velocidad) * 60


def aterrizaje_libre(dist_inicial, minutos) -> float:
    """Distancia que se cubriría a velocidad máxima en un lapso dado."""
    v_max = vel_maxima_permitida_por_tramo(dist_inicial)
    return (v_max / 60.0) * minutos


def hay_gap_disponible(avion, cola) -> bool:
    """Verifica si el avión puede volver con al menos 10 minutos de separación."""
    pos = avion.distancia
    avance_estimado = aterrizaje_libre(pos, minutos=5)

    for otro in cola:
        if otro.estado in ("APROXIMANDO", "AJUSTANDO"):
            if pos - avance_estimado <= otro.distancia <= pos + avance_estimado:
                return False
    return True


# ======================
# Clase Avión
# ======================
class Avion:
    def __init__(self, id_avion, minuto_actual):
        self.id = id_avion
        self.distancia = 100.0
        self.velocidad = 0.0
        self.estado = "APROXIMANDO"
        # Estados posibles: APROXIMANDO | AJUSTANDO | REGRESANDO | REGRESANDO_VIENTO | DESVIADO | ATERRIZADO
        self.tiempo_llegada = minuto_actual
        self.retraso = 0
        self.t_aterrizaje = None

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
# Clase Simulador
# ======================
class Simulador:
    def __init__(self, seed=42):
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
    def __init__(self, seed=42):
        super().__init__(seed)

    def controlar_regreso(self, avion, cola):
        # --- congestión normal ---
        if avion.distancia > RADAR_DIST:
            avion.estado = "DESVIADO"
            print(f" ❌ Avión {avion.id} se desvía a Montevideo.")
        elif hay_gap_disponible(avion, cola):
            avion.estado = "APROXIMANDO"
            print(f" ✅ Avión {avion.id} encuentra hueco y regresa a la fila.")

    def mover_aviones(self):
        for avion in self.aviones.values():
            if avion.estado in ("ATERRIZADO", "DESVIADO"):
                continue
            delta = avion.velocidad / 60.0
            if avion.estado in ("REGRESANDO_DISTANCIA", "REGRESANDO_VIENTO"):
                avion.distancia += delta
            else:
                avion.distancia -= delta

    def gestionar_finalizados(self, minuto: int, tiempo_ideal=23.4):
        for avion in self.aviones.values():
            if avion.estado in ("ATERRIZADO", "DESVIADO"):
                continue

            if avion.distancia <= 0:
                # --- 10% chance de go-around por viento ---
                if self.rng.random() < 0.1:
                    avion.estado = "REGRESANDO_VIENTO"
                    avion.velocidad = 200  # kts
                    avion.distancia = 5    # reingresa a 5 nm
                    print(f" 🌬️ Avión {avion.id} aborta aterrizaje por viento y regresa.")
                else:
                    avion.estado = "ATERRIZADO"
                    avion.t_aterrizaje = minuto
                    tiempo_real = minuto - avion.tiempo_llegada
                    avion.retraso = max(0, tiempo_real - tiempo_ideal)
                    self.finalizados.append(avion)



# ======================
# Clase Simulador con Tormenta
# ======================
# ======================
# Clase Simulador con Tormenta
# ======================
class SimuladorTormenta(Simulador):
    def __init__(self, *, seed=42, t_inicio=600, duracion=30):
        super().__init__(seed=seed)
        self.t_inicio = t_inicio
        self.t_fin = t_inicio + duracion

    def actualizar_estados(self, minuto: int):
        """Igual que Simulador, pero mantiene congestión"""
        activos = [a for a in self.aviones.values() if a.estado not in ("ATERRIZADO", "DESVIADO")]
        activos.sort(key=lambda av: av.distancia)

        for i, avion in enumerate(activos):
            if avion.estado.startswith("REGRESANDO"):
                self.controlar_regreso(avion, activos)
            else:
                lider = activos[i-1] if i > 0 and not activos[i-1].estado.startswith("REGRESANDO") else None
                avion.controlar_aproximacion(lider)

    def gestionar_finalizados(self, minuto: int, tiempo_ideal=23.4):
        """Si la tormenta está activa, no permite aterrizar."""
        for avion in self.aviones.values():
            if avion.estado in ("ATERRIZADO", "DESVIADO"):
                continue

            if avion.distancia <= 0:
                # Caso 1: aeropuerto cerrado → aborta
                if self.t_inicio <= minuto < self.t_fin:
                    avion.estado = "REGRESANDO_TORMENTA"
                    avion.velocidad = 200  # nudos
                    avion.distancia = 5    # reinsertado a 5 mn
                    # print(f"⛈️ Avión {avion.id} aborta aterrizaje por tormenta (t={minuto}).")
                else:
                    # Caso 2: aeropuerto abierto → aterriza normal
                    avion.estado = "ATERRIZADO"
                    avion.t_aterrizaje = minuto
                    tiempo_real = minuto - avion.tiempo_llegada
                    avion.retraso = max(0, tiempo_real - tiempo_ideal)
                    self.finalizados.append(avion)

    def mover_aviones(self):
        """Mueve también los que regresan por tormenta."""
        for avion in self.aviones.values():
            if avion.estado in ("ATERRIZADO", "DESVIADO"):
                continue
            delta = avion.velocidad / 60.0
            if avion.estado.startswith("REGRESANDO"):
                avion.distancia += delta
            else:
                avion.distancia -= delta

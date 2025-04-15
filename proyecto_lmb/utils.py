import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import StandardScaler
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import re
import joblib
import math
from sklearn.preprocessing import MinMaxScaler
pd.options.display.max_columns = None
import joblib

modelo_kmeans_bateadores = joblib.load("jlazo3010/lmb-dashboard/main/proyecto_lmb/modelos/modelo_kmeans_bateadores.joblib")
modelo_kmeans_lanzadores = joblib.load("jlazo3010/lmb-dashboard/main/proyecto_lmb/modelos/modelo_kmeans_lanzadores.joblib")

def proceso_total():
    
    # URL de la página de estadísticas
    url = 'https://www.milb.com/es/mexican/stats/games?playerPool=ALL'

    # Realizar la solicitud GET a la página
    response = requests.get(url)
    response.raise_for_status()  # Verificar que la solicitud fue exitosa

    # Analizar el contenido HTML de la página
    soup = BeautifulSoup(response.text, 'html.parser')

    # Encontrar la tabla principal por su clase y atributo 'aria-label'
    tabla_principal = soup.find("table", {"class": ["bui-table", "is-desktop-sKqjv9Sb"], "aria-label": "Tabla de Estadísticas"})

    # Extraer las filas de la tabla
    rows = tabla_principal.find_all("tr")

    # Procesar encabezados correctamente
    header_row = rows[0]  # Primera fila de la tabla contiene los encabezados
    headers = [header.text.strip() for header in header_row.find_all("th")]

    lista_jugadores = []

    for i in range(1,26):
      jugadores = rows[i]
      jugadores1 = [jugadores1.text.strip() for jugadores1 in jugadores.find_all("th")]
      lista_jugadores.append(jugadores1[0])

    # Asegurarnos de agregar la columna 'Jugador' manualmente si no está clara
    if headers[0] != "JUGADORJUGADOR":
        headers.insert(0, "JUGADORJUGADOR")

    # Procesar filas de datos
    data = []
    for row in rows[1:]:  # Omitir la fila de encabezados
        cells = row.find_all("td")
        if cells:
            # Extraer el nombre del jugador
            jugador = cells[0].text.strip()
            lista_jugadores.append(jugador)  # Agregar a la lista de nombres

            # Extraer el resto de los datos
            other_data = [cell.text.strip() for cell in cells[1:]]

            # Agregar el nombre al inicio de la fila de datos
            data.append([jugador] + other_data)


    # Agregar cada nombre al inicio de la fila correspondiente
    data_con_nombres = [[nombre] + fila for nombre, fila in zip(lista_jugadores, data)]

    salida = pd.DataFrame(data_con_nombres, columns= headers)

    url = 'https://www.milb.com/es/mexican/stats/games?'
    for i in range(2,17):
      # URL de la página de estadísticas
      url1 = url + "page="+str(i)+"&playerPool=ALL"

      # Realizar la solicitud GET a la página
      response = requests.get(url1)
      response.raise_for_status()  # Verificar que la solicitud fue exitosa

      # Analizar el contenido HTML de la página
      soup = BeautifulSoup(response.text, 'html.parser')

      # Encontrar la tabla principal por su clase y atributo 'aria-label'
      tabla_principal = soup.find("table", {"class": ["bui-table", "is-desktop-sKqjv9Sb"], "aria-label": "Tabla de Estadísticas"})

      # Extraer las filas de la tabla
      rows = tabla_principal.find_all("tr")

      # Procesar encabezados correctamente
      header_row = rows[0]  # Primera fila de la tabla contiene los encabezados
      headers = [header.text.strip() for header in header_row.find_all("th")]

      lista_jugadores = []

      for j in range(1,26):
        jugadores = rows[j]
        jugadores1 = [jugadores1.text.strip() for jugadores1 in jugadores.find_all("th")]
        lista_jugadores.append(jugadores1[0])

      # Asegurarnos de agregar la columna 'Jugador' manualmente si no está clara
      if headers[0] != "JUGADORJUGADOR":
          headers.insert(0, "JUGADORJUGADOR")

      # Procesar filas de datos
      data = []
      for row in rows[1:]:  # Omitir la fila de encabezados
          cells = row.find_all("td")
          if cells:
              # Extraer el nombre del jugador
              jugador = cells[0].text.strip()
              lista_jugadores.append(jugador)  # Agregar a la lista de nombres

              # Extraer el resto de los datos
              other_data = [cell.text.strip() for cell in cells[1:]]

              # Agregar el nombre al inicio de la fila de datos
              data.append([jugador] + other_data)


      # Agregar cada nombre al inicio de la fila correspondiente
      data_con_nombres = [[nombre] + fila for nombre, fila in zip(lista_jugadores, data)]

      base = pd.DataFrame(data_con_nombres, columns= headers)

      salida = pd.concat([salida, base], ignore_index=True)

    def limpiar_nombre(nombre):
        # Eliminar caracteres invisibles y espacios extras
        nombre = re.sub(r"[\u200b\u200c\u200d]", "", nombre).strip()

        # Eliminar basura al principio/final (números, letras tipo CF, etc.)
        nombre = re.sub(r"^\d+|[A-Z]{1,3}\d*$", "", nombre)

        # Regex mejorada: permite apellidos con apóstrofe
        mayus = "A-ZÁÉÍÓÚÑÜ"
        minus = "a-záéíóúñü"
        letras = f"{mayus}{minus}'"  # incluye apóstrofes

        # Patrón: Nombre + letra suelta, Apellido repetido (posible apóstrofe)
        patron = fr"([{mayus}][{minus}]+)[{mayus}]?\s([{letras}]+)\2"

        match = re.search(patron, nombre)
        if match:
            return f"{match.group(1)} {match.group(2)}"

        return nombre.strip()

    salida["JUGADORJUGADOR"] = salida["JUGADORJUGADOR"].apply(limpiar_nombre)

    salida.set_index("JUGADORJUGADOR", inplace=True)
    salida.iloc[:, 1:] = salida.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

    salida = salida[salida.iloc[:,1] > 11]
    salida = salida[salida["TBTB"] > 0]

    salida = salida.assign(CXJ=salida["CC"] / salida["TBTB"], HRXJ=salida["HRHR"] / salida["TBTB"],CIXJ=salida["CICI"] / salida["TBTB"],PXJ=salida["PP"] / salida["caret-upcaret-downJcaret-upcaret-downJ"],HXJ=salida["HH"] / salida["TBTB"], TBXJ = salida["TBTB"] / salida["caret-upcaret-downJcaret-upcaret-downJ"])

    columnas_bat = salida[["PROPRO","OPSOPS","TBXJ"]]
    columnas_bat.iloc[:, 0:] = columnas_bat.iloc[:, 0:].apply(pd.to_numeric, errors='coerce')

    TBXJ = pd.DataFrame(columnas_bat.iloc[:,2])

    estandarizar = StandardScaler()
    TBXJ_std = pd.DataFrame(estandarizar.fit_transform(TBXJ), columns=TBXJ.columns, index=TBXJ.index)

    # Si solo estás estandarizando una columna y quieres renombrarla
    TBXJ_std = TBXJ_std.rename(columns={TBXJ.columns[0]: 'TBXJ_std'})

    datos_std_bateadores = pd.concat([columnas_bat.loc[:,["PROPRO","OPSOPS"]], TBXJ_std], axis = 1)

    # Implementación del análisis de clusters con KMeans
    modelo_kmeans_bateadores.fit(datos_std_bateadores)

    # Añadimos los etiquetas de cluster a los datos preprocesados para su análisis
    salida['Cluster'] = modelo_kmeans_bateadores.labels_
    mapping = {0: 1, 1: 3}
    salida['Cluster'] = salida['Cluster'].replace(mapping)

    ################################################################################################################ Se generan los datos de los lanzadores
    # URL de la página de estadísticas
    urllanzadores = 'https://www.milb.com/es/mexican/stats/pitching/games?playerPool=ALL'

    # Realizar la solicitud GET a la página
    response = requests.get(urllanzadores)
    response.raise_for_status()  # Verificar que la solicitud fue exitosa

    # Analizar el contenido HTML de la página
    soup = BeautifulSoup(response.text, 'html.parser')

    # Encontrar la tabla principal por su clase y atributo 'aria-label'
    tabla_principal = soup.find("table", {"class": ["bui-table", "is-desktop-sKqjv9Sb"], "aria-label": "Tabla de Estadísticas"})

    # Extraer las filas de la tabla
    rows = tabla_principal.find_all("tr")

    # Procesar encabezados correctamente
    header_row_lanzadores = rows[0]  # Primera fila de la tabla contiene los encabezados
    headers_lanzadores = [header.text.strip() for header in header_row_lanzadores.find_all("th")]

    lista_jugadores_lanzadores = []

    for i in range(1,26):
      jugadores_lanzadores = rows[i]
      jugadores1_lanzadores = [jugadores1_lanzadores.text.strip() for jugadores1_lanzadores in jugadores_lanzadores.find_all("th")]
      lista_jugadores_lanzadores.append(jugadores1_lanzadores[0])

    # Asegurarnos de agregar la columna 'Jugador' manualmente si no está clara
    if headers_lanzadores[0] != "JUGADORJUGADOR":
        headers_lanzadores.insert(0, "JUGADORJUGADOR")

    # Procesar filas de datos
    data = []
    for row in rows[1:]:  # Omitir la fila de encabezados
        cells = row.find_all("td")
        if cells:
            # Extraer el nombre del jugador
            jugador_lanzadores = cells[0].text.strip()
            lista_jugadores_lanzadores.append(jugador_lanzadores)  # Agregar a la lista de nombres

            # Extraer el resto de los datos
            other_data = [cell.text.strip() for cell in cells[1:]]

            # Agregar el nombre al inicio de la fila de datos
            data.append([jugador_lanzadores] + other_data)


    # Agregar cada nombre al inicio de la fila correspondiente
    data_con_nombres_lanzadores = [[nombre] + fila for nombre, fila in zip(lista_jugadores_lanzadores, data)]

    salida_lanzadores = pd.DataFrame(data_con_nombres_lanzadores, columns= headers_lanzadores)

    # URL de la página de estadísticas
    url = 'https://www.milb.com/es/mexican/stats/pitching/games?'

    for j in range(2,18):
      # Realizar la solicitud GET a la página
      urllanzadores2 = url + "page="+str(j)+"&playerPool=ALL"
      response = requests.get(urllanzadores2)
      response.raise_for_status()  # Verificar que la solicitud fue exitosa

      # Analizar el contenido HTML de la página
      soup = BeautifulSoup(response.text, 'html.parser')

      # Encontrar la tabla principal por su clase y atributo 'aria-label'
      tabla_principal = soup.find("table", {"class": ["bui-table", "is-desktop-sKqjv9Sb"], "aria-label": "Tabla de Estadísticas"})

      # Extraer las filas de la tabla
      rows2 = tabla_principal.find_all("tr")

      # Procesar encabezados correctamente
      header_row_lanzadores2 = rows2[0]  # Primera fila de la tabla contiene los encabezados
      headers_lanzadores2 = [header.text.strip() for header in header_row_lanzadores2.find_all("th")]

      lista_jugadores_lanzadores2 = []

      for i in range(1,26):
        jugadores_lanzadores2 = rows2[i]
        jugadores1_lanzadores2 = [jugadores1_lanzadores2.text.strip() for jugadores1_lanzadores2 in jugadores_lanzadores2.find_all("th")]
        lista_jugadores_lanzadores2.append(jugadores1_lanzadores2[0])

      # Asegurarnos de agregar la columna 'Jugador' manualmente si no está clara
      if headers_lanzadores2[0] != "JUGADORJUGADOR":
          headers_lanzadores2.insert(0, "JUGADORJUGADOR")

      # Procesar filas de datos
      data = []
      for row in rows2[1:]:  # Omitir la fila de encabezados
          cells = row.find_all("td")
          if cells:
              # Extraer el nombre del jugador
              jugador_lanzadores2 = cells[0].text.strip()
              lista_jugadores_lanzadores2.append(jugador_lanzadores2)  # Agregar a la lista de nombres

              # Extraer el resto de los datos
              other_data = [cell.text.strip() for cell in cells[1:]]

              # Agregar el nombre al inicio de la fila de datos
              data.append([jugador_lanzadores2] + other_data)


    # Agregar cada nombre al inicio de la fila correspondiente
      data_con_nombres = [[nombre] + fila for nombre, fila in zip(lista_jugadores_lanzadores2, data)]

      base = pd.DataFrame(data_con_nombres, columns= headers_lanzadores2)

      salida_lanzadores = pd.concat([salida_lanzadores, base], ignore_index=True)

    salida_lanzadores["JUGADORJUGADOR"] = salida_lanzadores["JUGADORJUGADOR"].apply(limpiar_nombre)
    salida_lanzadores.set_index("JUGADORJUGADOR", inplace=True)
    salida_lanzadores.iloc[:, 1:] = salida_lanzadores.iloc[:, 1:].astype(str).apply(pd.to_numeric, errors='coerce')
    salida_lanzadores = salida_lanzadores[salida_lanzadores.iloc[:,4] > 11]

    salida_lanzadores['JGJG'] = pd.to_numeric(salida_lanzadores['JGJG'], errors='coerce')
    salida_lanzadores['JPJP'] = pd.to_numeric(salida_lanzadores['JPJP'], errors='coerce')
    salida_lanzadores['ILIL'] = pd.to_numeric(salida_lanzadores['ILIL'], errors='coerce')
    salida_lanzadores['PP'] = pd.to_numeric(salida_lanzadores['PP'], errors='coerce')
    salida_lanzadores['HRHR'] = pd.to_numeric(salida_lanzadores['HRHR'], errors='coerce')
    salida_lanzadores['HH'] = pd.to_numeric(salida_lanzadores['HH'], errors='coerce')
    salida_lanzadores['CC'] = pd.to_numeric(salida_lanzadores['CC'], errors='coerce')
    salida_lanzadores['BBBB'] = pd.to_numeric(salida_lanzadores['BBBB'], errors='coerce')

    salida_lanzadores = salida_lanzadores.assign(
        PXJ=salida_lanzadores["PP"] / salida_lanzadores["ILIL"],
        HRXJ=salida_lanzadores["HRHR"] / salida_lanzadores["ILIL"],
        HXJ=salida_lanzadores["HH"] / salida_lanzadores["ILIL"],
        CXJ=salida_lanzadores["CC"] / salida_lanzadores["ILIL"],
        BBXJ=salida_lanzadores["BBBB"] / salida_lanzadores["ILIL"],
        JGP=np.where(
            (salida_lanzadores["JGJG"] + salida_lanzadores["JPJP"]).fillna(0) == 0,
            0,
            salida_lanzadores["JGJG"] / (salida_lanzadores["JGJG"] + salida_lanzadores["JPJP"])
        )
    )

    lanzadores_analisis = salida_lanzadores[["JGP","PXJ","WHIPWHIP","PROPRO","EFEEFE"]]

    datos_std_lanzadores = estandarizar.fit_transform(lanzadores_analisis)

    # Implementación del análisis de clusters con KMeans
    modelo_kmeans_lanzadores.fit(datos_std_lanzadores)

    # Añadimos los etiquetas de cluster a los datos preprocesados para su análisis
    salida_lanzadores['Cluster'] = modelo_kmeans_lanzadores.labels_
    mapping_lanzadores = {0:2,2:3}
    salida_lanzadores['Cluster'] = salida_lanzadores['Cluster'].replace(mapping_lanzadores)

    ########################################################################################################################################## Se trabajan los datos y se calcula la probabilidad

    salida = salida.reset_index().rename(columns={'index': 'index_bateador'})
    salida_lanzadores = salida_lanzadores.reset_index().rename(columns={'index': 'index_lanzador'})

    claves = pd.DataFrame({
        'Clave': [
            'MEX', 'QRO', 'OAX', 'AGS', 'MTY', 'PUE', 'MVA', 'JAL', 'LAG', 'YUC',
            'LAR', 'TIJ', 'LEO', 'SLT', 'DUR', 'VER', 'TAB', 'CHI', 'CAM', 'TIG'
        ],
        'Nombre_equipo': [
            'Diablos Rojos del Mexico', 'Conspiradores de Queretaro', 'Guerreros de Oaxaca',
            'Rieleros de Aguascalientes', 'Sultanes de Monterrey', 'Pericos de Puebla',
            'Acereros de Monclova', 'Mariachis de Guadalajara', 'Algodoneros de Union Laguna',
            'Leones de Yucatan', 'Tecolotes de los Dos Laredos', 'Toros de Tijuana',
            'Bravos de Leon', 'Saraperos de Saltillo', 'Generales de Durango',
            'El Aguila de Veracruz', 'Olmecas de Tabasco', 'Dorados de Chihuahua',
            'Piratas de Campeche', 'Tigres de Quintana Roo'
        ]
    })
    salida = salida.merge(claves, how='left', left_on='EQUIPOEQUIPO', right_on='Clave')
    salida_lanzadores = salida_lanzadores.merge(claves, how='left', left_on='EQUIPOEQUIPO', right_on='Clave')

    """# Se unen las bases"""

    # Hacemos el cross join
    combinacion = pd.merge(salida, salida_lanzadores, how='cross')

    combinacion = combinacion.assign(
        PROPRO_lanzadores= 1/ combinacion["PROPRO_y"],
        Hits_lanzadores= 1/ combinacion["HXJ_y"])

    columnas_a_escalar = ["PXJ_x", "OPSOPS", "WHIPWHIP", "PXJ_y", "PROPRO_lanzadores", "PROPRO_x", "Hits_lanzadores", "HXJ_x"]
    df_escalar = combinacion[columnas_a_escalar].copy()
    scaler = MinMaxScaler()
    combinacion[columnas_a_escalar] = scaler.fit_transform(df_escalar)

    combinacion['Score1'] = np.sqrt(
        (combinacion["PXJ_x"] - combinacion["PXJ_x"])**2 +
        (combinacion["OPSOPS"] - combinacion["WHIPWHIP"])**2 +
        (combinacion["PROPRO_lanzadores"] - combinacion["PROPRO_x"])**2 +
        (combinacion["Hits_lanzadores"] - combinacion["HXJ_x"])**2
    )

    mxscore1 = combinacion["Score1"].max()

    combinacion = combinacion.assign(Proba = 1 - (combinacion["Score1"] / mxscore1))

    def ajustar_proba(row):
        if row['Cluster_x'] == 2:
            if row['Proba'] - 0.07 < 0:
                return row['Proba']
            else:
                return row['Proba'] - 0.07
        elif row['Cluster_x'] == 3:
            if row['Proba'] - 0.12 < 0:
                return row['Proba']
            else:
                return row['Proba'] - 0.12
        else:
            return row['Proba']

    combinacion['Proba_ajustada'] = combinacion.apply(ajustar_proba, axis=1)

    equipo_local = pd.DataFrame({
        'Equipo': ['Mexico', 'Queretaro', 'Oaxaca', 'Aguascalientes', 'Monterrey', 'Puebla', 'Monclova', 'Jalisco', 'Laguna', 'Yucatan',
                  'DosLaredos', 'Tijuana', 'Leon', 'Saltillo', 'Durango', 'Veracruz', 'Tabasco', 'Chihuahua', 'Campeche', 'QuintanaRoo'],
        'Clave': ['MEX', 'QRO', 'OAX', 'AGS', 'MTY', 'PUE', 'MVA', 'JAL', 'LAG', 'YUC',
                  'LAR', 'TIJ', 'LEO', 'SLT', 'DUR', 'VER', 'TAB', 'CHI', 'CAM', 'TIG'],
        'Nombre_equipo': ['Diablos Rojos del Mexico', 'Conspiradores de Queretaro', 'Guerreros de Oaxaca', 'Rieleros de Aguascalientes',
                          'Sultanes de Monterrey', 'Pericos de Puebla', 'Acereros de Monclova', 'Mariachis de Guadalajara',
                          'Algodoneros de Union Laguna', 'Leones de Yucatan', 'Tecolotes de los Dos Laredos', 'Toros de Tijuana',
                          'Bravos de Leon', 'Saraperos de Saltillo', 'Generales de Durango', 'El Aguila de Veracruz',
                          'Olmecas de Tabasco', 'Dorados de Chihuahua', 'Piratas de Campeche', 'Tigres de Quintana Roo'],
        'CXJ': [7.28, 6.99, 6.31, 5.90, 5.88, 5.52, 5.79, 5.37, 5.69, 5.37,
                5.62, 5.49, 5.41, 5.52, 5.08, 4.82, 4.70, 4.60, 4.32, 3.76],
        'PXJ': [7.56, 6.88, 7.66, 5.76, 6.73, 7.13, 7.38, 7.50, 7.65, 7.19,
                6.72, 8.10, 8.09, 6.61, 8.22, 7.00, 6.68, 7.19, 7.37, 8.10],
        'PROPRO': [0.32, 0.31, 0.30, 0.32, 0.30, 0.28, 0.29, 0.28, 0.28, 0.28,
                  0.29, 0.27, 0.28, 0.29, 0.28, 0.27, 0.27, 0.29, 0.25, 0.24],
        'OBPOBP': [0.40, 0.39, 0.37, 0.38, 0.37, 0.36, 0.37, 0.36, 0.37, 0.36,
                  0.37, 0.36, 0.35, 0.36, 0.36, 0.35, 0.34, 0.34, 0.32, 0.31],
        'SLGSLG': [0.50, 0.50, 0.49, 0.46, 0.47, 0.47, 0.45, 0.45, 0.44, 0.44,
                  0.43, 0.44, 0.44, 0.43, 0.40, 0.41, 0.41, 0.40, 0.39, 0.36],
        'Cluster': [3, 3, 3, 2, 2, 2, 2, 2, 2, 2,
                    2, 1, 1, 2, 1, 1, 1, 1, 1, 1]
    })

    equipo_visitante = pd.DataFrame({
        'Equipo': ['Mexico', 'Queretaro', 'Oaxaca', 'Aguascalientes', 'Monterrey', 'Puebla', 'Monclova', 'Jalisco', 'Laguna', 'Yucatan',
                  'DosLaredos', 'Tijuana', 'Leon', 'Saltillo', 'Durango', 'Veracruz', 'Tabasco', 'Chihuahua', 'Campeche', 'QuintanaRoo'],
        'Clave': ['MEX', 'QRO', 'OAX', 'AGS', 'MTY', 'PUE', 'MVA', 'JAL', 'LAG', 'YUC',
                  'LAR', 'TIJ', 'LEO', 'SLT', 'DUR', 'VER', 'TAB', 'CHI', 'CAM', 'TIG'],
        'Nombre_equipo': ['Diablos Rojos del Mexico', 'Conspiradores de Queretaro', 'Guerreros de Oaxaca', 'Rieleros de Aguascalientes',
                          'Sultanes de Monterrey', 'Pericos de Puebla', 'Acereros de Monclova', 'Mariachis de Guadalajara',
                          'Algodoneros de Union Laguna', 'Leones de Yucatan', 'Tecolotes de los Dos Laredos', 'Toros de Tijuana',
                          'Bravos de Leon', 'Saraperos de Saltillo', 'Generales de Durango', 'El Aguila de Veracruz',
                          'Olmecas de Tabasco', 'Dorados de Chihuahua', 'Piratas de Campeche', 'Tigres de Quintana Roo'],
        'CXJ': [7.28, 6.99, 6.31, 5.90, 5.88, 5.52, 5.79, 5.37, 5.69, 5.37,
                5.62, 5.49, 5.41, 5.52, 5.08, 4.82, 4.70, 4.60, 4.32, 3.76],
        'PXJ': [7.56, 6.88, 7.66, 5.76, 6.73, 7.13, 7.38, 7.50, 7.65, 7.19,
                6.72, 8.10, 8.09, 6.61, 8.22, 7.00, 6.68, 7.19, 7.37, 8.10],
        'PROPRO': [0.32, 0.31, 0.30, 0.32, 0.30, 0.28, 0.29, 0.28, 0.28, 0.28,
                  0.29, 0.27, 0.28, 0.29, 0.28, 0.27, 0.27, 0.29, 0.25, 0.24],
        'OBPOBP': [0.40, 0.39, 0.37, 0.38, 0.37, 0.36, 0.37, 0.36, 0.37, 0.36,
                  0.37, 0.36, 0.35, 0.36, 0.36, 0.35, 0.34, 0.34, 0.32, 0.31],
        'SLGSLG': [0.50, 0.50, 0.49, 0.46, 0.47, 0.47, 0.45, 0.45, 0.44, 0.44,
                  0.43, 0.44, 0.44, 0.43, 0.40, 0.41, 0.41, 0.40, 0.39, 0.36],
        'Cluster': [3, 3, 3, 2, 2, 2, 2, 2, 2, 2,
                    2, 1, 1, 2, 1, 1, 1, 1, 1, 1]
    })
    
    salida_lanzadores = salida_lanzadores.assign()

    bateo_local = salida[["JUGADORJUGADOR","EQUIPOEQUIPO","Nombre_equipo","CXJ","HXJ","HRXJ","CIXJ","PXJ","PROPRO","OBPOBP",'Cluster']]
    bateo_visita = salida[["JUGADORJUGADOR","EQUIPOEQUIPO","Nombre_equipo","CXJ","HXJ","HRXJ","CIXJ","PXJ","PROPRO","OBPOBP",'Cluster']]
    lanzamiento_local = salida_lanzadores[["JUGADORJUGADOR","EQUIPOEQUIPO","Nombre_equipo","EFEEFE","BBXJ","HRXJ","HXJ","CXJ","PROPRO","WHIPWHIP","Cluster"]]
    lanzamiento_visita = salida_lanzadores[["JUGADORJUGADOR","EQUIPOEQUIPO","Nombre_equipo","EFEEFE","BBXJ","HRXJ","HXJ","CXJ","PROPRO","WHIPWHIP","Cluster"]]

    bateo_local.columns = [
        'Bateador', 'Clave', 'Nombre_equipo', 'Carreras', 'Hits', 'Jonrons',
        'Carreras.impulsadas', 'Ponches', 'PRO', 'OBP', 'Cluster'
    ]

    bateo_visita.columns = [
        'Bateador', 'Clave', 'Nombre_equipo', 'Carreras', 'Hits', 'Jonrons',
        'Carreras.impulsadas', 'Ponches', 'PRO', 'OBP', 'Cluster'
    ]

    lanzamiento_local.columns = [
        'Lanzador', 'Clave', 'Nombre_equipo', 'EFE', 'BB',
        'Jonrons', 'Hits', 'Carreras','PRO','WHIP', 'Cluster'
    ]


    lanzamiento_visita.columns = [
        'Lanzador', 'Clave', 'Nombre_equipo', 'EFE', 'BB',
        'Jonrons', 'Hits', 'Carreras','PRO','WHIP', 'Cluster'
    ]

    columnas_ele = [
        'JUGADORJUGADOR_x', 'EQUIPOEQUIPO_x', 'Nombre_equipo_x', 'Cluster_x',
        'HXJ_x', 'CC_x',
        'PP_x', 'HRHR_x', 'PXJ_x',
        'JUGADORJUGADOR_y', 'EQUIPOEQUIPO_y', 'Nombre_equipo_y', 'Cluster_y',
        'PROPRO_lanzadores','Hits_lanzadores','Score1','Proba','Proba_ajustada'
    ]

    Cruces = combinacion[columnas_ele]

    # Se escriben las bases del powerBi

    return equipo_local, equipo_visitante, bateo_local, bateo_visita, lanzamiento_local, lanzamiento_visita, Cruces

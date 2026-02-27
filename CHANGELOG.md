# Changelog de [/solo_colectivo/main.py]

Todas las novedades notables de este proyecto se documentarán en este archivo.

## [1.4.5] - 2026-02-27

 ### Removed

 - Función importar_csv (ahora solo se utiliza importar_datos()).

### Added

- Se agregó control de valores ingresados en configuración.
- Se agregó manual de usuario.
- Se agregó informe con resultados y conclusiones.

### Changed

- En gráfico barras días se cambiaron los colores de las barras.
- Se modificó el título de la ventana root.

### Fixed

- Se corrigió evento <MouseScroll> en clase VisorPdf, bind_all generaba conflicto se se creaban
  varias instancias de la clase, se reemplazo por bind.
- En procesar_df() se podaron filas con valor 0 en campo correspondiente a cantidad.

## [1.4.4] - 2026-02-22

### Added
- función leer_archivo_config, chequeo y lectura del archivo config.json.

### Changed
- Los valores del config.json creado en caso de error se definen al inicio de código en un diccionario default_config.


## [1.4.3] - 2026-02-20

### Added
- Función crear_archivo_config(), crea un archivo config.json con valores por default.

### Fixed
- Contempla el caso que no exista config.json.
- Contempla el caso que config.json no contenga claves o valores inconsistentes.

## [1.4.2] - 2026-02-13

### Added

- Control de datos ingresados en configuracion(), sólo se inicializa/entrena model si hubo cambio
  en valores y hay un DataFrame cargado.
- Se actualiza de manera dinámica componente text_output si se cambia chk_preview_var en configuracion().

### Changed

- Se cambio del uso de variables max_iter_ini,test_size_ini,random_state_ini y mostrar_preview_ini por
  diccionario config.
- Se cambió key "mostrar_previsualizacion" por "mostrar_preview" en ./config.json.

### Fixed

- Error "failed to open file..." al inicio del programa.
- Error "procesando datos...", si se intenta guardar configuración sin tener cargado un DataFrame
- En verificar() se cambió messagebox.showerror por messagebox.showinfo


## [1.4.0] - 2026-02-11

### Added

- Archivo temporal /solo_colectivo/main_con_ini.py.
  - Varios datos de configuración se leen desde ./config.json.
  - Ventana de configuración para actualizar config.json.
  - Entrada en menu hacer.

 ### Removed

 - Función custom_message.
 - Función estadisticas_texto.

## [1.3.1] - 2026-02-11

### Changed

- Se pasó la clase VisorPdf a un archivo separado visorpdf.py y se importa en main.py.

## [1.3.0] - 2026-01-09

### Added

- Solapas para aplicación, manual e informe.
- Clase VisorPdf y se abrió manual.pdf e informe en solapas correspondientes.
- Etiqueta con versión en footer.

## [1.2.0] - 2025-12-29

## [1.1.0] - 2025-12-29

- Aplicación: /solo_colectivo/main.py

## [1.0.0] - 2025-12-29

_First release._

[1.4.4]: https://github.com/UrielCasas/Practica_Profesionalizante/tree/main/_vers/solo_colectivo/v1.4.4
[1.4.3]: https://github.com/UrielCasas/Practica_Profesionalizante/tree/main/_vers/solo_colectivo/v1.4.3
[1.4.2]: https://github.com/UrielCasas/Practica_Profesionalizante/tree/main/_vers/solo_colectivo/v1.4.2
[1.4.0]: https://github.com/UrielCasas/Practica_Profesionalizante/tree/main/_vers/solo_colectivo/v1.4.0
[1.3.1]: https://github.com/UrielCasas/Practica_Profesionalizante/tree/main/_vers/solo_colectivo/v1.3.1
[1.3.0]: https://github.com/UrielCasas/Practica_Profesionalizante/tree/main/_vers/solo_colectivo/v1.3.0
[1.2.0]: https://github.com/UrielCasas/Practica_Profesionalizante/tree/main/_vers/solo_colectivo/v1.2.0





















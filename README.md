# EcoVision

Aplicación web académica para clasificar residuos a partir de fotografías con un modelo MobileNetV2 entrenado. La interfaz permite cargar imágenes o capturarlas con cámara, consultar los registros persistidos y explorar estadísticas calculadas desde la base de datos.

## Tecnologías y arquitectura

- Python 3.11, Flask, TensorFlow/Keras, Pillow y NumPy.
- Flask-SQLAlchemy con Supabase PostgreSQL como única base de datos de ejecución.
- HTML, CSS y JavaScript sin framework frontend.
- Docker Compose y pytest.

`backend/prediction.py` valida y prepara imágenes; `database/` contiene SQLAlchemy; `app.py` integra API y páginas; `visualizacion/` contiene plantillas y recursos estáticos. La URL PostgreSQL es obligatoria. Crear la aplicación o importar `app.py` no ejecuta DDL ni migraciones. Flask-Migrate/Alembic versionan el esquema; los cambios se aplican mediante comandos explícitos.

## Modelo y clases

Se utiliza `modelo/waste_classifier.keras` tal como fue proporcionado, con entrada RGB de 224 × 224 y salida de 12 probabilidades. El modelo declara Keras 3.13.2 en sus metadatos; el artefacto no conserva la versión exacta de TensorFlow con que fue guardado. El contenedor fija TensorFlow CPU 2.20.0 y Keras 3.15.1, combinación recomendada actualmente en la matriz oficial de compatibilidad de Keras; el modelo guardado en Keras 3.13.2 aún requiere la prueba real para confirmar la carga de este artefacto específico. La inferencia entrega valores RGB [0, 255] y no vuelve a aplicar `preprocess_input`, pues el preprocesamiento se encuentra dentro del modelo.

Las clases se cargan exclusivamente desde `modelo/classes.txt`: battery, biological, brown-glass, cardboard, clothes, green-glass, metal, paper, plastic, shoes, trash y white-glass. `MODEL_DIR`, `MODEL_FILENAME` y `CLASSES_FILENAME` permiten configurar las rutas sin cambiar código. No se entrena ni descarga nada al iniciar.

## Entrenamiento y evaluación

El modelo se entrenó con el dataset Garbage Classification de Mostafa Abla, dividido de forma estratificada en entrenamiento, validación y prueba. El entrenamiento tuvo etapas de Transfer Learning y Fine-tuning con MobileNetV2 e ImageNet. En la evaluación final sobre las 2.328 imágenes de prueba obtuvo **95,27 % de accuracy**, **95,28 % de F1 ponderado** y **93,58 % de F1 macro**. Las métricas completas, configuración de entrenamiento y análisis por categoría están en [Entrenamiento y evaluación del modelo](docs/model-training-evaluation.md).

## Requisitos

- Docker Desktop con Docker Compose, recomendado para Windows/macOS/Linux.
- Para instalación local: Python 3.11 y las dependencias de `requirements.txt`. Para ejecutar pruebas, instala también `requirements-dev.txt`. TensorFlow puede requerir recursos significativos.
- Para cámara, abrir la web en localhost o bajo HTTPS y permitir acceso al dispositivo.

## Ejecución con Docker

1. Copia .env.example como .env; configura DATABASE_URL y SUPABASE_CA_FILE con la URL privada y la ruta host al CA oficial de Supabase; no publiques ninguno.
2. Para uso local ejecuta `docker compose -f docker-compose.yml -f docker-compose.local.yml up --build` y abre `http://localhost:5000`; el puerto se publica solo en 127.0.0.1.
3. Docker ejecuta únicamente Flask/Gunicorn y el modelo. No contiene una base de datos local ni monta un volumen para persistencia; las clasificaciones, el historial y las estadísticas usan Supabase.

El archivo `.env` se usa por Compose en el host y no se copia a la imagen. Para construir sin Compose, pasa `DATABASE_URL` como variable de entorno al contenedor. La aplicación valida la URL al iniciar y rechaza esquemas distintos de PostgreSQL con psycopg 3.

El servidor no crea ni modifica el esquema al arrancar. Las migraciones se aplican manualmente después de revisar el destino y el script. No ejecutes comandos de escritura contra producción hasta confirmar que la revisión coincide con el esquema existente.

El Dockerfile incluye el repositorio completo `migrations/`. Compose exige `DATABASE_URL` y la pasa al contenedor. Puedes ejecutar comandos Flask explícitamente con `docker compose run --rm app`; no forman parte del arranque de Gunicorn:

```powershell
docker compose run --rm app flask --app app db history
docker compose run --rm app flask --app app db heads
docker compose run --rm app flask --app app db upgrade
```

`history` lista las revisiones y `heads` muestra la revisión principal disponible. `upgrade` aplica las revisiones pendientes y **modifica el esquema de la base de datos**. No se ejecuta automáticamente al iniciar la aplicación. El baseline `85b59e312c4e` se registró en Supabase tras una comparación autorizada del esquema; esta fase no volvió a conectarse ni comprobó el destino. Antes de ejecutar `upgrade`, compara el esquema actual y autoriza expresamente la operación.

## Ejecución local

La aplicación carga `.env` automáticamente desde la raíz. Configura una `DATABASE_URL` PostgreSQL válida con el dialecto `postgresql+psycopg://...`; si falta, está vacía o usa otro dialecto, el inicio falla sin imprimir la URL ni la contraseña. Instala con `python -m pip install -r requirements.txt` y arranca con `python app.py`.

## Migraciones PostgreSQL

Las migraciones son revisiones versionadas del esquema, almacenadas en `migrations/` y `migrations/versions/`. Flask-Migrate expone los comandos Alembic mediante `flask db`. La fábrica Flask solo registra la extensión: ni la importación, ni `python app.py`, ni Gunicorn ejecutan `upgrade` o crean tablas.

Para generar una revisión futura, apunta `DATABASE_URL` a una base PostgreSQL de desarrollo aislada (no a producción) y ejecuta:

```powershell
python -m flask --app app db migrate -m "describe el cambio"
```

`migrate` compara los modelos con la base configurada y puede conectarse a ella. Inspecciona el archivo creado en `migrations/versions/`, corrige cualquier operación incompleta o incorrecta y confirma que solo contiene los cambios previstos antes de guardarlo en Git. La revisión inicial de `predictions` se escribió desde los modelos inspeccionados y se registró como baseline en Supabase tras la inspección autorizada informada para el proyecto. No se usó autogenerate contra Supabase.

Comandos para revisar el historial y la revisión disponible sin aplicar cambios:

```powershell
python -m flask --app app db history
python -m flask --app app db heads
```

`current` consulta la versión aplicada en la base indicada y requiere conexión:

```powershell
python -m flask --app app db current
```

Para aplicar una revisión, configura primero la URL del entorno objetivo, revisa otra vez el SQL/código de migración y ejecuta explícitamente:

```powershell
python -m flask --app app db upgrade
```

Hazlo primero en una base de pruebas y verifica los datos y el esquema después. No se aplica una migración automáticamente al iniciar el servidor. El baseline `85b59e312c4e` figura registrado en Supabase tras una inspección autorizada previa. Esta fase no volvió a consultar la base; confirma el estado actual y revisa la migración antes de cualquier `upgrade`. El `downgrade` de esta revisión inicial está bloqueado para evitar que una reversión accidental elimine registros; usa una migración correctiva hacia delante.

## Funcionalidades y API

- Panel con total, categoría más frecuente, último registro y distribución.
- Clasificación de JPG/JPEG/PNG mediante carga o cámara; la ráfaga está limitada a tres fotos, analizadas en secuencia.
- Historial reciente paginado y filtros de categoría/fecha.
- Estadísticas de categoría, confianza promedio y evolución diaria.
- Se guardan metadatos, nunca las fotografías.

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/api/health` | Estado de Flask, modelo y base de datos |
| POST | `/api/predict` | Recibe campo multipart `image`, predice y persiste |
| GET | `/api/predictions` | Historial; admite `category`, `date_from`, `date_to`, `page`, `per_page` |
| GET | `/api/predictions/<id>` | Detalle persistido |
| GET | `/api/statistics` | Totales y agregaciones reales |

## Pruebas

Instala dependencias con `python -m pip install -r requirements-dev.txt`. `python -m pytest` ejecuta las pruebas unitarias y de API con modelo/persistencia simulados; nunca usa Supabase ni SQLite. Las pruebas PostgreSQL reales llevan la marca `postgres_integration` y se ejecutan aparte contra el contenedor efímero descrito a continuación. El test del modelo real requiere `ECOVISION_REAL_MODEL=1`.

## Integración PostgreSQL aislada

Requisitos: Docker Engine disponible, la imagen `postgres:16-alpine` descargable y las dependencias de `requirements-dev.txt`. Este entorno solo contiene PostgreSQL de pruebas: base y usuario `ecovision_test`, puerto local `55432`, contraseña de prueba `ecovision_test_only`, sin volúmenes persistentes y con datos en `tmpfs`. No comparte red de aplicación, volúmenes ni credenciales con Supabase. La contraseña es fija y pública a propósito; nunca la reutilices.

En PowerShell, inicia exclusivamente el servicio de pruebas con un nombre de proyecto Compose propio:

```powershell
$env:TEST_POSTGRES_PORT = "55432"
docker compose -p ecovision-pg-it -f docker-compose.test.yml up -d --wait
$env:ECOVISION_TEST_DATABASE_URL = "postgresql+psycopg://ecovision_test:ecovision_test_only@127.0.0.1:55432/ecovision_test"
.\.venv\Scripts\python.exe -m pytest -m postgres_integration tests/postgres
```

La suite solo acepta `ECOVISION_TEST_DATABASE_URL` con driver psycopg 3, host `127.0.0.1`, puerto de pruebas, base y usuario exactos. Antes de DDL consulta `current_database()` y `current_user`; solo aplica `85b59e312c4e` si el esquema está vacío. Si ya existe, exige exactamente `predictions` y `alembic_version`, la revisión correcta y cero filas. Ante cualquier otra situación se detiene sin modificar la base. Las pruebas no borran registros. Para iniciar otra ejecución limpia, elimina el contenedor efímero con `docker compose -p ecovision-pg-it -f docker-compose.test.yml down` y vuelve a iniciarlo; no uses el Compose normal de EcoVision para esta suite.

La fixture ejecuta Alembic (`85b59e312c4e`) sobre ese contenedor como parte de la suite, solo después de validar el destino. No uses `flask db upgrade` con la URL del `.env` de la aplicación para preparar esta suite; así se evita apuntar por error a Supabase. Para ejecutar solo las pruebas reales: `python -m pytest -m postgres_integration tests/postgres`. Si `ECOVISION_TEST_DATABASE_URL` no está definido, pytest las omite; si apunta a un host, puerto, usuario o base distinto, falla antes de conectar. Comprueba siempre que el Compose de pruebas esté arriba y que la URL conserve el host `127.0.0.1` y la base `ecovision_test`. No copies ni derives esta URL desde `.env`.

## TLS de Supabase

La aplicación impone sslmode=verify-full y sslrootcert a PostgreSQL remoto. El archivo CA lo monta Compose como secreto de solo lectura; la ruta host se configura con SUPABASE_CA_FILE.

Esta fase no cambió `DATABASE_URL`, no leyó sus credenciales ni se conectó a Supabase. La política TLS efectiva del proyecto y el modo negociado no fueron comprobados. Referencias: [conexión PostgreSQL de Supabase](https://supabase.com/docs/guides/database/connecting-to-postgres), [SSL enforcement de Supabase](https://supabase.com/docs/guides/platform/ssl-enforcement) y [modos SSL de libpq](https://www.postgresql.org/docs/current/libpq-ssl.html).

## Seguridad y límites conocidos

- `MAX_IMAGE_MB` limita el cuerpo de imagen (1–25 MB; predeterminado 10 MB). Solo se aceptan JPG/JPEG/PNG con extensión y contenido real válidos; se rechazan archivos corruptos y dimensiones sobre 20 millones de píxeles. No se aceptan videos. La imagen se decodifica y redimensiona en memoria.
- Límites por IP remota: 5 predicciones/minuto, 30 consultas de historial/minuto, 60 detalles/minuto, 20 estadísticas/minuto y 30 healthchecks/minuto; las demás rutas tienen un máximo predeterminado de 120/minuto. Se usa `memory://` por defecto, por lo que los contadores son por proceso y se reinician al reiniciar. Configure `RATE_LIMIT_STORAGE_URI` con almacenamiento compartido si ejecuta varios workers o réplicas.
- La IP se toma de `request.remote_addr`; `X-Forwarded-For` se ignora por defecto (`TRUSTED_PROXY_HOPS=0`). Detrás de un proxy, déjelo así si no necesita IPs individuales. Para confiar en proxies, primero haga que el proxy elimine la cabecera recibida del cliente y escriba su propia cadena, y configure `TRUSTED_PROXY_HOPS` con el número exacto (1 a 10); no exponga el contenedor a tráfico directo. `ProxyFix` solo usa ese número de saltos para `X-Forwarded-For`. Una topología mal configurada puede permitir eludir límites o agrupar visitantes bajo una IP.
- CORS usa `CORS_ALLOWED_ORIGINS`, lista separada por comas de orígenes exactos, sin credenciales. En ejecución local se permiten localhost/127.0.0.1 en puertos 3000 y 5000; para otro origen, configure su valor exacto. Compose pasa el valor configurado y no habilita comodines.
- Las inferencias se serializan dentro de cada proceso para limitar memoria y evitar llamadas concurrentes al mismo modelo. Gunicorn usa un worker y cuatro hilos. Cada worker/réplica adicional cargará su propia copia del modelo.
- El historial limita `per_page` a 100 y `page` a 10000; las estadísticas solo devuelven los últimos 90 días de serie diaria. Totales y agregaciones históricas siguen consultando los registros existentes; define límites operativos y una política de retención antes de uso público sostenido. No hay borrado automático.
- El pool PostgreSQL habilita comprobación previa de conexiones, reciclaje a 30 minutos y máximo de cinco conexiones sin overflow por proceso. Los errores del API no devuelven trazas ni detalles de PostgreSQL/TensorFlow.
- La confianza Softmax es una probabilidad del modelo, no garantía de acierto.
- Las fotos se procesan en memoria y no se conservan.
- La cámara requiere permiso del navegador y contexto seguro.
- La conexión Supabase se configura en el `.env` local, excluido de Git; verifica el destino antes de ejecutar migraciones.
- Docker se validó parcialmente en Fase 8.1; el E2E quedó pendiente por un error read-only del almacenamiento de Docker Desktop.
- No hay autenticación, como requiere el producto público. El rate limiting no sustituye cuotas globales ni protección DDoS del proveedor.

## Fase 8.1: usuario, puertos y TLS

El contenedor ejecuta Gunicorn como usuario sin privilegios UID/GID 10001; los artefactos del modelo son legibles y de solo lectura. `/tmp` y la caché Keras se mantienen como ubicaciones temporales escribibles.

`docker-compose.yml` no publica puertos del host y deja que el proveedor de hosting enrute al puerto interno 5000. Para uso local, combina el Compose base con `docker-compose.local.yml`: `docker compose -f docker-compose.yml -f docker-compose.local.yml up --build`. La publicación local se limita a `127.0.0.1` y no impone esa restricción al despliegue.

Para una base remota, la aplicación exige `SUPABASE_CA_FILE`, establece `sslmode=verify-full` y configura `sslrootcert`. Compose monta el CA como secreto de solo lectura en `/run/secrets/supabase-ca.crt`. Define `SUPABASE_CA_FILE` en `.env` como una ruta del host al certificado descargado desde Database Settings del panel oficial Supabase. No se descarga ni se copia el certificado a la imagen. Las pruebas PostgreSQL locales usan su URL dedicada y no requieren CA de producción.

## Prueba E2E aislada

La prueba E2E usa `docker-compose.test.yml` y `docker-compose.e2e.yml`, con el proyecto `ecovision-e2e`, PostgreSQL efímero, usuario/base `ecovision_test` y publicación en `127.0.0.1:5002`. No carga `.env` ni usa Supabase. En PowerShell:

```powershell
$env:TEST_POSTGRES_PORT = "55432"
$env:ECOVISION_TEST_DATABASE_URL = "postgresql+psycopg://ecovision_test:ecovision_test_only@127.0.0.1:55432/ecovision_test"
docker compose -p ecovision-e2e -f docker-compose.test.yml up -d --wait
python -m pytest -m postgres_integration tests/postgres -v
docker compose -p ecovision-e2e -f docker-compose.test.yml -f docker-compose.e2e.yml up -d --build --wait app-e2e
$env:ECOVISION_E2E_BASE_URL = "http://127.0.0.1:5002"
python -m pytest -m e2e tests/e2e -v
docker compose -p ecovision-e2e -f docker-compose.test.yml -f docker-compose.e2e.yml down
```

La E2E verifica la identidad y revisión antes de la escritura y luego comprueba Gunicorn, disponibilidad del modelo, una predicción real, el registro persistido, historial y estadísticas. `down` detiene exclusivamente los recursos de ese proyecto; PostgreSQL reside en tmpfs.

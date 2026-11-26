# EcoVision

Aplicación web académica para clasificar residuos a partir de fotografías con un modelo MobileNetV2 entrenado. La interfaz permite cargar imágenes o capturarlas con cámara, consultar los registros persistidos y explorar estadísticas calculadas desde la base de datos.

## Tecnologías y arquitectura

- Flask, TensorFlow/Keras, Pillow y NumPy. Docker usa Python 3.11; Vercel fija Python 3.12 mediante `.python-version`.
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
- Para instalacion local usa Python 3.11 y `requirements.txt`; para pruebas, instala `requirements-dev.txt`. Vercel instala las dependencias de `requirements.txt` con Python 3.12. TensorFlow puede requerir recursos significativos.
- Para cámara, abrir la web en localhost o bajo HTTPS y permitir acceso al dispositivo.

## Configuración de Vercel

`app.py` exporta la aplicación WSGI de Flask. `.python-version` fija Python 3.12 y `requirements.txt` contiene las dependencias de la función. `vercel.json` activa Fluid Compute, fija `maxDuration` a 60 segundos y coloca la función en Portland (`pdx1`, AWS `us-west-2`), la región indicada para Supabase. El bundle incluye módulos Python, migraciones, plantillas, estáticos y los dos recursos del modelo activo. Las reglas del bundle y `.vercelignore` excluyen artefactos de entrenamiento, archivos locales, pruebas y secretos. [Runtime Python](https://vercel.com/docs/functions/runtimes/python) · [Regiones de Vercel](https://vercel.com/docs/regions)

El límite estándar documentado para Python es 500 MB descomprimidos. Large Functions está en beta pública y permite hasta 5 GB descomprimidos con Fluid Compute y Active CPU. El build anterior reportó 1418,31 MB y avanzó hasta la configuración SSL al activar la opción en Preview. Comprueba el tamaño de la función y el indicador de Large Functions en el siguiente Preview. [Límites y Large Functions](https://vercel.com/docs/functions/limitations#large-functions-beta)

| Variable | Uso en Vercel |
| --- | --- |
| `DATABASE_URL` | Obligatoria y sensible. URL de PostgreSQL con `postgresql+psycopg://`, host del Transaction Pooler y puerto 6543. |
| `SUPABASE_CA_CERT` | Obligatoria y sensible. PEM completo del CA oficial que corresponda al host. |
| `VERCEL_SUPPORT_LARGE_FUNCTIONS` | Configura `1` para los entornos de este proyecto que usen el bundle grande. |
| `MAX_IMAGE_MB` | Opcional. Límite de archivo entre 1 y 4 MiB, por defecto 4. |
| `PREDICTION_RATE_LIMIT` | Opcional. Por defecto `10 per minute`. |
| `RATE_LIMIT_STORAGE_URI` | Opcional. Por defecto `memory://`, con contadores por proceso. |
| `TRUSTED_PROXY_HOPS` | Opcional. Mantén el valor predeterminado `0` durante la verificación inicial. |
| `CORS_ALLOWED_ORIGINS` | Opcional. Solo configura orígenes exactos separados por comas cuando haya un frontend de otro origen. |
| `VERCEL_ANALYZE_BUILD_OUTPUT` | Opcional para Preview. Usa `1` si necesitas el desglose de tamaño en los logs de build. |

### Preview y Production

1. En Vercel, abre el proyecto. En **Settings → Build and Deployment**, comprueba que la raíz contenga `app.py` y `vercel.json`, y que el preset detecte Flask. Usa los comandos de instalación y build predeterminados. En **Settings → Functions**, comprueba Fluid Compute y Active CPU. [Configuración de Flask](https://vercel.com/docs/frameworks/backend/flask) · [Fluid Compute](https://vercel.com/docs/fluid-compute)
2. En **Settings → Environment Variables**, añade `DATABASE_URL` y `SUPABASE_CA_CERT` como sensibles en **Preview**, y `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` en **Preview**. Usa el host exacto de **Connect → Transaction pooler** de Supabase y conserva el usuario que corresponde a ese rol y proyecto. Si existe una base aislada con el esquema preparado para validar Preview, usa su URL y CA. Configura la protección de Preview en **Settings → Deployment Protection** según el plan. [Entornos de Vercel](https://vercel.com/docs/environment-variables) · [Conexión de Supabase](https://supabase.com/docs/guides/database/connecting-to-postgres)
3. En **Settings → Functions → Function CPU**, usa inicialmente **Standard**. Hobby dispone de 2 GB y 1 vCPU. Pro y Enterprise también ofrecen **Performance** con 4 GB y 2 vCPU desde el dashboard. Usa los logs de Preview para decidir si necesitas elevar memoria o los 60 segundos de duración. Con Fluid Compute, el máximo es 300 s en Hobby y 800 s en Pro/Enterprise, con 1800 s en beta para runtimes elegibles. [Memoria y CPU](https://vercel.com/docs/functions/configuring-functions/memory) · [Duración](https://vercel.com/docs/functions/configuring-functions/duration)
4. En el siguiente Preview autorizado, verifica Python 3.12, la región de ejecución, el tamaño de la función, el indicador de Large Functions y la carga en frío. Comprueba `/api/health`, `/static/js/app.js`, `/static/css/app.css` y las cuatro páginas. La guía de Flask recomienda `public/` para servir estáticos por CDN. La configuración actual incluye los estáticos que sirve la ruta de Flask, cuyos HTTP 200 se deben confirmar en Preview. [Flask y archivos estáticos](https://vercel.com/docs/frameworks/backend/flask#serving-static-assets)
5. Desde HTTPS, clasifica un JPG y un PNG que originalmente supere 4 MiB, una captura y una ráfaga. En la pestaña **Network**, comprueba que cada POST tiene una parte `image`, devuelve 201 y que la ráfaga crea tres registros. Confirma el detalle, los filtros de historial y los totales de estadísticas. Verifica las respuestas para imagen corrupta y límites de archivo y cuerpo.
6. Para validar la IP en el Preview protegido, envía 11 solicitudes sin imagen desde la misma red en menos de un minuto. Cambia `X-Forwarded-For` entre solicitudes y repite desde una segunda red con otra IP pública. Correlaciona el comportamiento con la instancia si hay escalado. Si hace falta observar el entorno WSGI, usa diagnóstico temporal en logs privados de Preview y compara presencia de `REMOTE_ADDR`, longitud de la cadena y coincidencias con el salto esperado, sin registrar IPs completas. Repite el control con el dominio y cualquier proxy que vaya a usar Production. Ajusta los saltos solo tras verificar la topología y elimina el diagnóstico antes de publicar. [Cabeceras de Vercel](https://vercel.com/docs/headers/request-headers#x-forwarded-for)
7. Tras aprobar Preview, añade las variables sensibles y Large Functions a **Production**. Comprueba los permisos del rol y la decisión del propietario sobre historial público y retención. La revisión `85b59e312c4e` ya registrada en Supabase se reutiliza. Valida salud, una predicción de prueba, historial, estadísticas, tiempos y logs en el primer Production autorizado.

## Ejecución con Docker

1. Copia .env.example como .env; configura DATABASE_URL y SUPABASE_CA_FILE con la URL privada y la ruta host al CA oficial de Supabase; no publiques ninguno.
2. Para uso local ejecuta `docker compose -f docker-compose.yml -f docker-compose.local.yml up --build` y abre `http://localhost:5000`; el puerto se publica solo en 127.0.0.1.
3. Docker ejecuta únicamente Flask/Gunicorn y el modelo. No contiene una base de datos local ni monta un volumen para persistencia; las clasificaciones, el historial y las estadísticas usan Supabase.
   La imagen instala `requirements-docker.txt`, que agrega Gunicorn a las dependencias comunes de `requirements.txt`. Vercel usa `requirements.txt` sin instalar Gunicorn.

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

Para PostgreSQL remoto, `SUPABASE_CA_CERT` es la fuente principal cuando esta configurada; `SUPABASE_CA_FILE` se conserva como alternativa secundaria para Docker/local. La aplicacion valida el certificado X.509 y fuerza `sslmode=verify-full` con `sslrootcert`, sin permitir que opciones de `DATABASE_URL` debiliten la validacion. Compose conserva el montaje de solo lectura en `/run/secrets/supabase-ca.crt`.

En Vercel, crea `SUPABASE_CA_CERT` como variable **secreta** en cada entorno que corresponda y configura el PEM completo del CA oficial, incluyendo `-----BEGIN CERTIFICATE-----` y `-----END CERTIFICATE-----`. Puede tener saltos de linea reales o secuencias `\n`. No incluyas el certificado en Git, `.env.example`, logs ni en la imagen. La aplicacion valida el PEM con OpenSSL mediante `ssl` de Python y lo escribe una vez por proceso en un archivo temporal de `/tmp` con permisos `0600`; se conserva durante la vida del proceso para futuras conexiones psycopg. `/tmp` es efimero y no se comparte entre instancias.

Si `SUPABASE_CA_CERT` esta configurada, se valida y utiliza antes de consultar `SUPABASE_CA_FILE`; si esta presente pero es invalida, el inicio falla claramente y no recurre al archivo. Si no esta configurada, puede utilizarse un `SUPABASE_CA_FILE` valido. Para PEM, la aplicacion escribe una copia temporal en `/tmp` con permisos `0600` y la reutiliza durante la vida del proceso. La URL final fuerza `sslmode=verify-full` (incluida la validacion del hostname) y reemplaza cualquier `sslmode` o `sslrootcert` contradictorio de `DATABASE_URL`.

Esta fase no cambió `DATABASE_URL`, no leyó sus credenciales ni se conectó a Supabase. La política TLS efectiva del proyecto y el modo negociado no fueron comprobados. Referencias: [conexión PostgreSQL de Supabase](https://supabase.com/docs/guides/database/connecting-to-postgres), [SSL enforcement de Supabase](https://supabase.com/docs/guides/platform/ssl-enforcement) y [modos SSL de libpq](https://www.postgresql.org/docs/current/libpq-ssl.html).

## Seguridad y límites conocidos

- Flask acepta archivos de hasta 4 MiB y un cuerpo de hasta 4 MiB más 64 KiB para multipart. Pillow verifica JPG/JPEG/PNG y rechaza contenido corrupto y dimensiones superiores a 20 millones de píxeles. El navegador admite originales de hasta 25 MiB, 40 millones de píxeles y 16 384 píxeles por lado. La subida, la cámara y la ráfaga envían JPEG con lado máximo inicial de 1024 píxeles y un objetivo de hasta el 95 % del límite de archivo.
- El límite de predicción predeterminado es 10 por minuto por IP remota. Historial permite 30 por minuto, detalle 60, estadísticas 20 y salud 30. Las demás rutas tienen un límite predeterminado de 120 por minuto. `memory://` guarda contadores por proceso y se reinicia con él. Cada instancia serverless mantiene su propio contador.
- La IP se toma de `request.remote_addr`. Con `TRUSTED_PROXY_HOPS=0`, Flask no vuelve a interpretar cabeceras de proxy. El adaptador WSGI publicado por Vercel asigna `REMOTE_ADDR` a partir de `X-Forwarded-For` o `X-Real-IP`, después de la normalización de la plataforma. Comprueba qué dirección recibe Flask en el Preview concreto. [Cabeceras de Vercel](https://vercel.com/docs/headers/request-headers#x-forwarded-for) · [Adaptador WSGI](https://github.com/vercel/vercel/blob/main/python/vercel-runtime/src/vercel_runtime/vc_init.py#L1031)
- CORS se aplica a la API y permite únicamente los orígenes exactos de `CORS_ALLOWED_ORIGINS`, sin credenciales. Vercel usa el mismo origen por defecto. En local se permiten localhost y 127.0.0.1 en los puertos 3000 y 5000.
- Flask carga una copia del modelo por proceso al inicializar. La inferencia se serializa dentro del proceso para acotar el uso de memoria y reutilizar el modelo mientras la instancia está activa.
- El historial limita `per_page` a 100 y `page` a 10000; las estadísticas solo devuelven los últimos 90 días de serie diaria. Totales y agregaciones históricas siguen consultando los registros existentes; define límites operativos y una política de retención antes de uso público sostenido. No hay borrado automático.
- SQLAlchemy usa `NullPool` con el Transaction Pooler de Supabase. Flask-SQLAlchemy retira la sesión al terminar la solicitud y se libera la conexión local. Las conexiones al puerto 6543 desactivan las sentencias preparadas automáticas de psycopg. Los errores del API devuelven mensajes genéricos.
- La confianza Softmax es una probabilidad del modelo, no garantía de acierto.
- Las fotos se procesan durante la solicitud. Werkzeug puede usar un archivo temporal del sistema para las cargas grandes. Se guardan los metadatos de la predicción. El historial, el detalle y la última predicción de estadísticas son públicos. La interfaz envía nombres genéricos. Los registros anteriores y los clientes directos de la API aún pueden exponer nombres reconocibles.
- La cámara requiere permiso del navegador y contexto seguro.
- La conexión Supabase se configura en el `.env` local, excluido de Git; verifica el destino antes de ejecutar migraciones.
- Docker se validó parcialmente en Fase 8.1; el E2E quedó pendiente por un error read-only del almacenamiento de Docker Desktop.
- No hay autenticación, como requiere el producto público. El rate limiting no sustituye cuotas globales ni protección DDoS del proveedor.

### Permisos de PostgreSQL y RLS

Usa un rol de inicio de sesión exclusivo de Flask, sin privilegios de administración ni `BYPASSRLS` y distinto del propietario de la tabla. Concédele `CONNECT` en la base, `USAGE` en el esquema, `SELECT` e `INSERT` en `predictions` y `USAGE` en la secuencia de su ID. Reserva el rol propietario para operaciones de esquema. Si habilitas RLS, establece políticas `SELECT` e `INSERT` para ese rol y valida ambas rutas antes de publicar. Evalúa revocar accesos a `anon`, `authenticated` y `PUBLIC` según otros consumidores. Si todo el proyecto usa PostgreSQL directo y ninguna aplicación usa REST o GraphQL de Supabase, se puede desactivar Data API desde **Integrations → Data API**. Las solicitudes públicas pasan por la misma identidad de Flask, por lo que el control del historial individual corresponde a la aplicación. [RLS de Supabase](https://supabase.com/docs/guides/database/postgres/row-level-security) · [Controles de Data API](https://supabase.com/docs/guides/api/securing-your-api)

## Fase 8.1: usuario, puertos y TLS

El contenedor ejecuta Gunicorn como usuario sin privilegios UID/GID 10001; los artefactos del modelo son legibles y de solo lectura. `/tmp` y la caché Keras se mantienen como ubicaciones temporales escribibles.

`docker-compose.yml` no publica puertos del host y deja que el proveedor de hosting enrute al puerto interno 5000. Para uso local, combina el Compose base con `docker-compose.local.yml`: `docker compose -f docker-compose.yml -f docker-compose.local.yml up --build`. La publicación local se limita a `127.0.0.1` y no impone esa restricción al despliegue.

Para PostgreSQL remoto, configura `SUPABASE_CA_CERT` como variable secreta con el PEM completo; esta fuente tiene prioridad y Vercel no necesita `SUPABASE_CA_FILE`. Docker/local puede conservar `SUPABASE_CA_FILE` como alternativa y mantiene el montaje de solo lectura en `/run/secrets/supabase-ca.crt`. Ambos caminos mantienen `sslmode=verify-full` y `sslrootcert`. No descargues el certificado durante el build ni lo copies a la imagen. Las pruebas PostgreSQL locales usan su URL dedicada y no requieren CA de produccion.

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

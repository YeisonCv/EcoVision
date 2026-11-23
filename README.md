# EcoVision

Aplicación web académica para clasificar residuos a partir de fotografías con un modelo MobileNetV2 entrenado. La interfaz permite cargar imágenes o capturarlas con cámara, consultar los registros persistidos y explorar estadísticas calculadas desde la base de datos.

## Tecnologías y arquitectura

- Python 3.11, Flask, TensorFlow/Keras, Pillow y NumPy.
- SQLAlchemy con PostgreSQL de Supabase para Docker; SQLite sigue disponible en ejecuciones locales sin DATABASE_URL.
- HTML, CSS y JavaScript sin framework frontend.
- Docker Compose y pytest.

`backend/prediction.py` valida y prepara imagenes; `database/` contiene SQLAlchemy; `app.py` integra API y paginas; `visualizacion/` contiene plantillas y recursos estaticos. La aplicacion no crea ni cambia el esquema al importarse o arrancar. Inicializa solo tablas faltantes mediante el comando explicito `flask --app app init-db`; conserva las tablas existentes.

## Modelo y clases

Se utiliza `modelo/waste_classifier.keras` tal como fue proporcionado, con entrada RGB de 224 × 224 y salida de 12 probabilidades. El modelo declara Keras 3.13.2 en sus metadatos; el artefacto no conserva la versión exacta de TensorFlow con que fue guardado. El contenedor fija TensorFlow CPU 2.20.0 y Keras 3.15.1, combinación recomendada actualmente en la matriz oficial de compatibilidad de Keras; el modelo guardado en Keras 3.13.2 aún requiere la prueba real para confirmar la carga de este artefacto específico. La inferencia entrega valores RGB [0, 255] y no vuelve a aplicar `preprocess_input`, pues el preprocesamiento se encuentra dentro del modelo.

Las clases se cargan exclusivamente desde `modelo/classes.txt`: battery, biological, brown-glass, cardboard, clothes, green-glass, metal, paper, plastic, shoes, trash y white-glass. `MODEL_DIR`, `MODEL_FILENAME` y `CLASSES_FILENAME` permiten configurar las rutas sin cambiar código. No se entrena ni descarga nada al iniciar.

## Requisitos

- Docker Desktop con Docker Compose, recomendado para Windows/macOS/Linux.
- Para instalación local: Python 3.11 y las dependencias de `requirements.txt`. Para ejecutar pruebas, instala también `requirements-dev.txt`. TensorFlow puede requerir recursos significativos.
- Para cámara, abrir la web en localhost o bajo HTTPS y permitir acceso al dispositivo.

## Ejecución con Docker

1. Copia `.env.example` como `.env` y completa `DATABASE_URL` con tu cadena de conexión PostgreSQL de Supabase. Compose falla si no está definida; no hay una base de datos local en el despliegue Docker.
2. Inicializa el esquema solo si corresponde y de forma explícita: `docker compose run --build --rm app flask --app app init-db`. Verifica el destino antes de ejecutarlo.
3. Arranca con `docker compose up --build app` y abre `http://localhost:5000`. Para detener, usa `Ctrl+C` y `docker compose down`.

### Supabase

Define `DATABASE_URL` con la URL PostgreSQL de Supabase (`postgresql+psycopg://usuario:contrasena@host:puerto/base`). Escapa caracteres especiales y no publiques la URL. Compose lee `.env` y exige `DATABASE_URL`; el archivo no se copia a la imagen. Confirma el destino antes de crear tablas. La inicialización sigue siendo manual con `docker compose run --build --rm app flask --app app init-db`; luego inicia Gunicorn con `docker compose up --build app`.

## Ejecución local

La aplicacion carga `.env` automaticamente desde la raiz. Si `DATABASE_URL` no existe ni en el entorno ni en `.env`, usa SQLite local (`ecovision.db`). Si la URL comienza con `postgresql+psycopg://`, usa PostgreSQL. Para SQLite, elimina o comenta `DATABASE_URL` en `.env`; para PostgreSQL, configurala explicitamente. Instala con `python -m pip install -r requirements.txt`, inicializa el esquema solo cuando corresponda con `python -m flask --app app init-db` y arranca con `python app.py`. El arranque normal no crea tablas.

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

Instala dependencias de pruebas con `python -m pip install -r requirements-dev.txt`. `python -m pytest` ejecuta pruebas unitarias y de integracion con SQLite en memoria y modelo simulado. Incluye pruebas de importacion sin creacion de tablas y de inicializacion explicita. La verificacion del modelo real se ejecuta con `python verify_model.py` o `ECOVISION_REAL_MODEL=1 python -m pytest tests/integration/test_real_model.py`. No ejecutes pruebas con el `DATABASE_URL` de produccion en el entorno.

## Seguridad y límites conocidos

- Tamaño máximo configurable con `MAX_IMAGE_MB` (1–25 MB; predeterminado 10 MB); valida formato, integridad y dimensiones.
- La confianza Softmax es una probabilidad del modelo, no garantía de acierto.
- Las fotos se procesan en memoria y no se conservan.
- La cámara requiere permiso del navegador y contexto seguro.
- La conexion Supabase se configura en el `.env` local, excluido de Git; verifica el destino antes de ejecutar `init-db`.
- La construccion y ejecucion en Docker no se verificaron en este entorno.
- No hay autenticación, como solicita el alcance académico.
